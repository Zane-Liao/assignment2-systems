import math
from typing import Optional
from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
import triton
import triton.language as tl

__all__ = [
    "FlashAttnAutogradFunction",
    "FlashTritonForwardAttnAutogradFunction",
    "TritonFlashAttentionAutogradFunction",
]


class FlashAttnAutogradFunction(torch.autograd.Function):
    """Naive Pytorch Implementation"""
    @staticmethod
    def forward(
        ctx,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        is_causal=False
        ):
        B, N, d = Q.shape
        
        # Define block_size
        B_q, B_k = 32, 32
        
        # Compute
        T_q = (N + B_q - 1) // B_q
        T_k = (N + B_k - 1) // B_k
        
        # $O_i = \sum_j \text{softmax}(S_{ij}) V_j$
        O = torch.zeros_like(Q)
        # $L_i = \log\left(\sum_j \exp(S_{ij})\right)$
        L = torch.zeros((B, N), device=Q.device)
        
        for b in range(B):
            for i in range(T_q):
                start_q = i * B_q
                end_q = min((i + 1) * B_q, N)
            
                # Load Q_i from global memory
                Q_i = Q[b, start_q : end_q, :]
            
                # Initialize O_i l_i m_i
                O_i = torch.zeros_like(Q_i)
                l_i = torch.zeros((Q_i.shape[0], 1))
                # -float('inf') ==> −∞
                m_i = torch.full((Q_i.shape[0], 1), float('-inf'), device=Q.device)

                for j in range(T_k):
                    start_k = j * B_k
                    end_k = min((j + 1) * B_k, N)
                
                    # Load K(j), V(j) from global memory
                    # K, V shape into ==> B_k x d
                    K_j = K[b, start_k : end_k, :]
                    V_j = V[b, start_k : end_k, :]
                
                    # mask
                    if is_causal and end_k > end_q:
                        break
                
                    # Compute tile of pre-softmax attention scores
                    # S_ij = (Q_i @ K_j.transpose(-2, -1)) / torch.sqrt(d)
                    S_ij = (Q_i @ K_j.T) / math.sqrt(d)
                
                    if is_causal:
                        row_idx = torch.arange(start_q, end_q, device=Q.device).view(-1, 1)
                        col_idx = torch.arange(start_k, end_k, device=Q.device).view(1, -1)
                    
                        mask = row_idx >= col_idx
                        S_ij = S_ij.masked_fill(~mask, float('-inf'))
                    
                    # Initial
                    m_ij = torch.max(S_ij, dim=1, keepdim=True).values
                    P_ij = torch.exp(S_ij - m_ij)
                    l_ij = torch.sum(P_ij, dim=1, keepdim=True)
                
                    m_new = torch.maximum(m_i, m_ij)
                
                    exp_m_diff = torch.exp(m_i - m_new)
                    exp_m_ij_diff = torch.exp(m_ij - m_new)
                
                    l_i = (exp_m_diff * l_i) + (exp_m_ij_diff * l_ij)
                
                    O_i = exp_m_diff * O_i
                    O_i = O_i + (exp_m_ij_diff * (P_ij @ V_j))
                
                    m_i = m_new
                    # End for
            
                O_i = O_i / l_i
                O[b, start_q : end_q, : ] = O_i
                L[b, start_q : end_q] = (m_i + torch.log(l_i)).view(-1)
                # End for
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        
        dQ, dK, dV = torch_flash_bwd(
            ctx, dO, Q, K, V, O, L, is_causal=is_causal
        )
        
        return dQ, dK, dV, None


def torch_flash_bwd(ctx, dO, Q, K, V, O, L, is_causal=None):
    B, N, d = Q.shape
        
    D = torch.sum(dO * O, dim=-1, keepdim=True)
        
    scale = 1.0 / math.sqrt(d)
        
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    # Define block_size
    B_q, B_k = 32, 32
        
    # Compute
    T_q = (N + B_q - 1) // B_q
    T_k = (N + B_k - 1) // B_k
        
    for b in range(B):
        for j in range(T_k):
            start_k = j * B_k
            end_k = min((j + 1) * B_k, N)
                
            K_j = K[b, start_k : end_k, :]
            V_j = V[b, start_k : end_k, :]
                
            dK_j = torch.zeros_like(K_j)
            dV_j = torch.zeros_like(V_j)
                
            for i in range(T_q):
                start_q = i * B_q
                end_q = min((i + 1)  * B_q, N)
                    
                Q_i = Q[b, start_q : end_q, :]
                O_i = O[b, start_q : end_q, :]
                dO_i = dO[b, start_q : end_q, :]
                # dQ_i = Q[b, start_q : end_q, :]
                L_i = L[b, start_q : end_q]
                D_i = D[b, start_q : end_q, :]
                    
                S_ij = Q_i @ K_j.T * scale
                    
                if is_causal:
                    row_idx = torch.arange(start_q, end_q, device=Q.device).view(-1, 1)
                    col_idx = torch.arange(start_k, end_k, device=Q.device).view(1, -1)
                    
                    mask = row_idx >= col_idx
                    S_ij = S_ij.masked_fill(~mask, float('-inf'))
                    
                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))
                    
                dV_j += P_ij.transpose(0, 1) @ dO_i

                dP_ij = dO_i @ V_j.transpose(0, 1)

                # sqrt(d) ==> scale
                dS_ij = P_ij * (dP_ij - D_i)
                    
                dQ_i = dS_ij @ K_j
                dQ_i = dQ_i * scale
                    
                # sqrt(d)
                dK_j += (dS_ij.transpose(0, 1) @ Q_i) * scale

                dQ[b, start_q : end_q, :] += dQ_i
                # End for

            dK[b, start_k : end_k, :] += dK_j
            dV[b, start_k : end_k, :] += dV_j
            # End for
        
    return dQ, dK, dV


class FlashTritonForwardAttnAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.ndim == K.ndim == V.ndim == 3
        assert Q.shape == K.shape == V.shape

        B, NQ, D = Q.shape
        NK = K.shape[1]
        D_PAD = next_power_of_2(D)
        scale = 1.0 / math.sqrt(D)

        Q_TILE = 32
        K_TILE = 32

        O = torch.empty((B, NQ, D), device=Q.device, dtype=torch.float32)
        LSE = torch.empty((B, NQ), device=Q.device, dtype=torch.float32)

        grid = (triton.cdiv(NQ, Q_TILE), B)

        flash_fwd_kernel[grid](
            Q_ptr=Q,
            K_ptr=K,
            V_ptr=V,
            O_ptr=O,
            LSE_ptr=LSE,
            stride_qb=Q.stride(0),
            stride_qq=Q.stride(1),
            stride_qd=Q.stride(2),
            stride_kb=K.stride(0),
            stride_kk=K.stride(1),
            stride_kd=K.stride(2),
            stride_vb=V.stride(0),
            stride_vk=V.stride(1),
            stride_vd=V.stride(2),
            stride_ob=O.stride(0),
            stride_oq=O.stride(1),
            stride_od=O.stride(2),
            stride_lsb=LSE.stride(0),
            stride_lsq=LSE.stride(1),
            N_QUERIES=NQ,
            N_KEYS=NK,
            scale=scale,
            D=D,
            D_PAD=D_PAD,
            Q_TILE_SIZE=Q_TILE,
            K_TILE_SIZE=K_TILE,
            is_causal=bool(is_causal),
        )

        ctx.save_for_backward(Q, K, V, O, LSE)
        ctx.is_causal = bool(is_causal)
        return O.to(Q.dtype)
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = torch_flash_bwd(ctx, dO, Q, K, V, O, L, is_causal=is_causal)
        return dQ, dK, dV, None


def next_power_of_2(x):
    return 1 << (x - 1).bit_length()

class TritonFlashAttentionAutogradFunction(torch.autograd.Function):
    """Naive version with TritonFlashAttention2 Implementation, Maybe It's a little slow"""
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.ndim == K.ndim == V.ndim == 3
        assert Q.shape == K.shape == V.shape

        B, NQ, D = Q.shape
        NK = K.shape[1]
        D_PAD = next_power_of_2(D)
        scale = 1.0 / math.sqrt(D)

        Q_TILE = 32
        K_TILE = 32

        O = torch.empty((B, NQ, D), device=Q.device, dtype=torch.float32)
        LSE = torch.empty((B, NQ), device=Q.device, dtype=torch.float32)

        grid = (triton.cdiv(NQ, Q_TILE), B)

        flash_fwd_kernel[grid](
            Q_ptr=Q,
            K_ptr=K,
            V_ptr=V,
            O_ptr=O,
            LSE_ptr=LSE,
            stride_qb=Q.stride(0),
            stride_qq=Q.stride(1),
            stride_qd=Q.stride(2),
            stride_kb=K.stride(0),
            stride_kk=K.stride(1),
            stride_kd=K.stride(2),
            stride_vb=V.stride(0),
            stride_vk=V.stride(1),
            stride_vd=V.stride(2),
            stride_ob=O.stride(0),
            stride_oq=O.stride(1),
            stride_od=O.stride(2),
            stride_lsb=LSE.stride(0),
            stride_lsq=LSE.stride(1),
            N_QUERIES=NQ,
            N_KEYS=NK,
            scale=scale,
            D=D,
            D_PAD=D_PAD,
            Q_TILE_SIZE=Q_TILE,
            K_TILE_SIZE=K_TILE,
            is_causal=bool(is_causal),
        )

        ctx.save_for_backward(Q, K, V, O, LSE)
        ctx.is_causal = bool(is_causal)
        return O.to(Q.dtype)

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, LSE = ctx.saved_tensors
        is_causal = ctx.is_causal

        B, NQ, D = Q.shape
        NK = K.shape[1]
        D_PAD = next_power_of_2(D)
        scale = 1.0 / math.sqrt(D)

        Q_TILE = 32
        K_TILE = 32

        dQ = torch.zeros((B, NQ, D), device=Q.device, dtype=torch.float32)
        dK = torch.zeros((B, NK, D), device=Q.device, dtype=torch.float32)
        dV = torch.zeros((B, NK, D), device=Q.device, dtype=torch.float32)

        grid = (triton.cdiv(NK, K_TILE), B)

        flash_bwd_kernel[grid](
            Q_ptr=Q,
            K_ptr=K,
            V_ptr=V,
            O_ptr=O,
            LSE_ptr=LSE,
            dO_ptr=dO.to(torch.float32),
            dQ_ptr=dQ,
            dK_ptr=dK,
            dV_ptr=dV,
            stride_qb=Q.stride(0),
            stride_qq=Q.stride(1),
            stride_qd=Q.stride(2),
            stride_kb=K.stride(0),
            stride_kk=K.stride(1),
            stride_kd=K.stride(2),
            stride_vb=V.stride(0),
            stride_vk=V.stride(1),
            stride_vd=V.stride(2),
            stride_ob=O.stride(0),
            stride_oq=O.stride(1),
            stride_od=O.stride(2),
            stride_lsb=LSE.stride(0),
            stride_lsq=LSE.stride(1),
            stride_dob=dO.stride(0),
            stride_doq=dO.stride(1),
            stride_dod=dO.stride(2),
            stride_dqb=dQ.stride(0),
            stride_dqq=dQ.stride(1),
            stride_dqd=dQ.stride(2),
            stride_dkb=dK.stride(0),
            stride_dkq=dK.stride(1),
            stride_dkd=dK.stride(2),
            stride_dvb=dV.stride(0),
            stride_dvq=dV.stride(1),
            stride_dvd=dV.stride(2),
            N_QUERIES=NQ,
            N_KEYS=NK,
            scale=scale,
            D=D,
            D_PAD=D_PAD,
            Q_TILE_SIZE=Q_TILE,
            K_TILE_SIZE=K_TILE,
            is_causal=is_causal,
        )

        return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    LSE_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lsb,
    stride_lsq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr, D_PAD: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    scale = tl.full((), scale, tl.float32)

    q_tile = tl.program_id(0)
    batch = tl.program_id(1)

    offs_m = q_tile * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    offs_d = tl.arange(0, D_PAD)
    mask_m = offs_m < N_QUERIES
    mask_d = offs_d < D

    q_ptrs = Q_ptr + batch*stride_qb + offs_m[:, None]*stride_qq + offs_d[None,:]*stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None,:], other=0.0).to(tl.float32)

    o = tl.zeros((Q_TILE_SIZE, D_PAD), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), -1.0e20, dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    for k_start in range(0, N_KEYS, K_TILE_SIZE):
        offs_n = k_start + tl.arange(0, K_TILE_SIZE)
        mask_n = offs_n < N_KEYS

        k_ptrs = K_ptr + batch*stride_kb + offs_n[:, None]*stride_kk + offs_d[None,:]*stride_kd
        v_ptrs = V_ptr + batch*stride_vb + offs_n[:, None]*stride_vk + offs_d[None,:]*stride_vd

        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None,:], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None,:], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scale
        mask = (~mask_m[:, None]) | (~mask_n[None, :])
        if is_causal:
            mask = mask | (offs_n[None,:] > offs_m[:, None])
        scores = tl.where(mask, -1.0e20, scores).to(tl.float32)

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m, m_ij)

        alpha = tl.where(mask_m, tl.exp(m - m_new), 0.0).to(tl.float32)
        p = tl.exp(scores - m_new[:, None]).to(tl.float32)
        p = tl.where(mask, 0.0, p).to(tl.float32)

        l = alpha * l + tl.sum(p, axis=1)
        o = alpha[:, None] * o + tl.dot(p, v)
        m = tl.where(mask_m, m_new, m)

    l_safe = tl.where(l > 0, l, 1.0)
    out = o / l_safe[:, None]
    lse = tl.where(l > 0, m + tl.log(l_safe), -1.0e20)

    o_ptrs = O_ptr + batch*stride_ob + offs_m[:, None]*stride_oq + offs_d[None,:]*stride_od
    lse_ptrs = LSE_ptr + batch*stride_lsb + offs_m*stride_lsq

    tl.store(o_ptrs, out, mask=mask_m[:, None] & mask_d[None,:])
    tl.store(lse_ptrs, lse, mask=mask_m)


@triton.jit
def flash_bwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    LSE_ptr,
    dO_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lsb,
    stride_lsq,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    stride_dkb,
    stride_dkq,
    stride_dkd,
    stride_dvb,
    stride_dvq,
    stride_dvd,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr, D_PAD: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    scale = tl.full((), scale, tl.float32)

    kv_tile = tl.program_id(0)
    batch = tl.program_id(1)

    offs_n = kv_tile*K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
    offs_d = tl.arange(0, D_PAD)
    mask_n = offs_n < N_KEYS
    mask_d = offs_d < D

    k_ptrs = K_ptr + batch*stride_kb + offs_n[:,None]*stride_kk + offs_d[None,:]*stride_kd
    v_ptrs = V_ptr + batch*stride_vb + offs_n[:,None]*stride_vk + offs_d[None,:]*stride_vd

    k = tl.load(k_ptrs, mask=mask_n[:,None] & mask_d[None,:], other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=mask_n[:,None] & mask_d[None,:], other=0.0).to(tl.float32)

    dK_acc = tl.zeros((K_TILE_SIZE, D_PAD), dtype=tl.float32)
    dV_acc = tl.zeros((K_TILE_SIZE, D_PAD), dtype=tl.float32)

    for q_start in range(0, N_QUERIES, Q_TILE_SIZE):
        offs_m = q_start + tl.arange(0, Q_TILE_SIZE)
        mask_m = offs_m < N_QUERIES

        q_ptrs = Q_ptr + batch*stride_qb + offs_m[:,None]*stride_qq + offs_d[None,:]*stride_qd
        o_ptrs = O_ptr + batch*stride_ob + offs_m[:,None]*stride_oq + offs_d[None,:]*stride_od
        do_ptrs = dO_ptr + batch*stride_dob + offs_m[:,None]*stride_doq + offs_d[None,:]*stride_dod
        lse_ptrs = LSE_ptr + batch*stride_lsb + offs_m*stride_lsq

        q = tl.load(q_ptrs, mask=mask_m[:,None] & mask_d[None,:], other=0.0).to(tl.float32)
        o = tl.load(o_ptrs, mask=mask_m[:,None] & mask_d[None,:], other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:,None] & mask_d[None,:], other=0.0).to(tl.float32)
        lse = tl.load(lse_ptrs, mask=mask_m, other=-1.0e20).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scale
        mask = (~mask_m[:,None]) | (~mask_n[None,:])
        if is_causal:
            mask = mask | (offs_n[None,:] > offs_m[:,None])
        scores = tl.where(mask, -1.0e20, scores).to(tl.float32)

        p = tl.exp(scores - lse[:,None]).to(tl.float32)
        p = tl.where(mask, 0.0, p).to(tl.float32)

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        delta = tl.sum(do * o, axis=1)[:, None].to(tl.float32)
        ds = (p * (dp - delta)).to(tl.float32)

        dV_acc += tl.dot(tl.trans(p), do).to(tl.float32)
        dK_acc += scale * tl.dot(tl.trans(ds), q).to(tl.float32)

        dQ_tile = scale * tl.dot(ds, k).to(tl.float32)
        dQ_ptrs = dQ_ptr + batch*stride_dqb + offs_m[:,None]*stride_dqq + offs_d[None,:]*stride_dqd
        tl.atomic_add(dQ_ptrs, dQ_tile, mask=mask_m[:,None] & mask_d[None,:])

    dK_ptrs = dK_ptr + batch*stride_dkb + offs_n[:,None]*stride_dkq + offs_d[None,:]*stride_dkd
    dV_ptrs = dV_ptr + batch*stride_dvb + offs_n[:,None]*stride_dvq + offs_d[None,:]*stride_dvd
    
    tl.store(dK_ptrs, dK_acc, mask=mask_n[:,None] & mask_d[None,:])
    tl.store(dV_ptrs, dV_acc, mask=mask_n[:,None] & mask_d[None,:])