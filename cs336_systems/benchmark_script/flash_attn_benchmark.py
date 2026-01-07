import timeit
import itertools
from typing import Callable, Optional
from einops import rearrange
import torch
import torch.nn.functional as F
from cs336_basics.modules.activation import scaled_dot_product_attention
from profiling_benchmark import *
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flash_attention import FlashAttnAutogradFunction, TritonFlashAttentionAutogradFunction
import triton
import triton.testing as testing


def my_scaled_dot_product_attention(Q, K, V):
    return scaled_dot_product_attention(Q, K, V)


def torch_scaled_dot_product_attention(Q, K, V, is_causal=False):
    return F.scaled_dot_product_attention(Q, K, V, is_causal)


def torch_flash_attn(Q, K, V, is_causal=False):
    return FlashAttnAutogradFunction.apply(Q, K, V, is_causal)


def half_triton_flash_attn(Q, K, V, is_causal=False):
    return TritonFlashAttentionAutogradFunction.apply(Q, K, V, is_causal)


def flash_benchmarking():
    B, H, N, D = 8, 16, 1024, 64
    q = torch.randn(B, H, N, D, device=get_device(), dtype=torch.float32)
    k = torch.randn(B, H, N, D, device=get_device(), dtype=torch.float32)
    v = torch.randn(B, H, N, D, device=get_device(), dtype=torch.float32)
    
    print("My Naive scaled_dot_product_attention Impl...")
    testing.do_bench(lambda: my_scaled_dot_product_attention(q, k, v))
    
    print("torch scaled_dot_product_attention Impl...")
    testing.do_bench(lambda: torch_scaled_dot_product_attention(q, k, v))
    
    print("My torch flash_attention2 Impl...")
    testing.do_bench(lambda: torch_flash_attn(q, k, v))
    
    print("flash_attention2 with Triton forward and torch backward Impl...")
    testing.do_bench(lambda: half_triton_flash_attn(q, k, v))

    
def flash_forward_backward():
    n_heads = 16

    d_head = 64
    
    sequence_length = 16384
    
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True
    )
    
    flash = torch.compile(TritonFlashAttentionAutogradFunction.apply)
    
    o = flash(q, k, v, True)
    
    loss = o.sum()
    
    loss.backward()
    
    results = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)

    print(results)
    

if __name__ == "__main__":
    flash_benchmarking()
    flash_forward_backward()