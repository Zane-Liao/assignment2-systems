import os
import json
import math
from typing import Optional
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.nn import init, Module, ModuleList, Parameter
from einops import rearrange
from dataclasses import dataclass
from .activation import Softmax, silu, scaled_dot_product_attention
from jaxtyping import Float, Int

__all__ = [
    "Embedding",
    "Linear",
    "RMSNorm",
    "SwiGLU",
    "RotaryPositionalEmbedding",
    "ScaledDotProductAttention",
    "MultiHeadSelfAttention",
    "TransformerBlock",
    "TransformerLM",
]


class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        linear transformation module.
    
        Parameters:
            in_features: int final dimension of the input
        
            out_features: int final dimension of the output
        
            device: torch.device | None = None Device to store the parameters on
        
            dtype: torch.dtype | None = None Data type of the parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        std = math.sqrt(2 / (in_features + out_features))
        init.trunc_normal_(
            self.weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std
            )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the linear transformation to the input.
        
        Parameter:
            x: torch.Tensor
        Return:
            torch.Tensor
        """
        output = x @ self.weight.T
        return output


class Embedding(Module):
    __constants__ = ['num_embeddings', 'embedding_dim']
    num_embeddings: int
    embedding_dim: int
    weight: Tensor
    
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        an embedding module.
        
        Parameters:
            num_embeddings: int Size of the vocabulary
            
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            
            device: torch.device | None = None Device to store the parameters on
            
            dtype: torch.dtype | None = None Data type of the parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        init.trunc_normal_(
            self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        
        Parameter:
            token_ids: torch.Tensor
        Return:
            torch.Tensor
        """
        return torch.embedding(self.weight, token_ids)


# I refer to torch concise implementation
class RMSNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module.
        
        Parameters:
            d_model: int Hidden dimension of the model
        
            eps: float = 1e-5 Epsilon value for numerical stability
        
            device: torch.device | None = None Device to store the parameters on
        
            dtype: torch.dtype | None = None Data type of the parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones(d_model, **factory_kwargs))
        
    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        
        Parameter:
            x: torch.Tensor
        Return:
            torch.Tensor
        """
        # In torch, x.float() => float32
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# position-wise feed-forward network
# Come from pytorch PR: Support Swiglu for Module and functional 144465
class SwiGLU(Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff) 
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


# I referred to some online Solution
class RotaryPositionalEmbedding(Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        RoPE module and create buffers if needed.
        
        Parameters:
            theta: float Θ value for the RoPE
            
            d_k: int dimension of query and key vectors
            
            max_seq_len: int Maximum sequence length that will be inputted
            
            device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        half_dim = d_k // 2
        self.theta = (1. / theta ** (torch.arange(0, half_dim, device=device) / half_dim))
        
        self.max_seq_len = torch.arange(max_seq_len, device=device)
        angle_max_seq_len = torch.einsum("i,j->ij", self.max_seq_len, self.theta)

        self.register_buffer("sin", torch.sin(angle_max_seq_len), persistent=False)
        self.register_buffer("cos", torch.cos(angle_max_seq_len), persistent=False)
    
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return
        a tensor of the same shape. Note that you should tolerate
        x with an arbitrary number of batch dimensions. You should assume
        that the token positions are a tensor of shape (..., seq_len)
        specifying the token positions of x along the sequence dimension.
        
        Parameters:
            x: torch.Tensor
            token_positions: torch.Tensor
        Return:
            torch.Tensor
        """
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]
        
        x_even = x[..., 0::2] 
        x_odd = x[..., 1::2]
        
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        
        return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)


class ScaledDotProductAttention(Module):
    """
    Your implementation should handle keys and queries of shape
    (batch_size, ..., seq_len, d_k) and values of shape
    (batch_size, ..., seq_len, d_v), where... represents any number of
    other batch-like dimensions (if provided). The implementation should
    return an output with the shape (batch_size, ..., d_v).
    See section 3.3 for a discussion on batch-like dimensions.
    Your implementation should also support an optional user-provided boolean
    mask of shape (seq_len, seq_len). The attention probabilities of 
    positions with a mask value of True should collectively sum to 1,
    and the attention probabilities of positions with a mask value of False
    should be zero. To test your implementation against our provided tests,
    you will need to implement the test adapter at 
    [adapters.run_scaled_dot_product_attention].
    """
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, mask):
        return scaled_dot_product_attention(query, key, value, mask)


class MultiHeadSelfAttention(Module):
    """
    Implement causal multi-head self-attention as a torch.nn.Module.
        We have (batch_size, seq_len, d_model)
        reshape (batch_size, seq_len, h, d_k)
        d_model == h * d_k ⟹ d_k = d_model // h
        
    Parameters:
        d_model: int Dimensionality of the Transformer block inputs

        num_heads: int Number of heads to use in multi-head self-attention

        Datermine whether rope_exist: (RotaryPositionalEmbedding exists) true or false
        
        theta: parameter for RotaryPositionalEmbedding

        max_seq_len: parameter for RotaryPositionalEmbedding
        
        rope_exist: parameter for RotaryPositionalEmbedding
    """
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 theta: float | None = None,
                 max_seq_len: int | None = None,
                 rope_exist: bool | None = None,
                 device=None,
                 dtype=None,
            ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv_proj = Linear(d_model, 3 * d_model, **factory_kwargs)
        self.o_proj = Linear(d_model, d_model, **factory_kwargs)
        
        self.rope_exist = rope_exist
        if self.rope_exist:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device
            )
        else:
            self.rope = None

    def forward(self,
                in_features: Tensor,
                token_positions: Optional[Tensor] = None,
               ) -> Tensor:
        
        batch_size, seq_len, _ = in_features.shape

        qkv = self.qkv_proj(in_features)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)

        if self.rope_exist:
            if token_positions is None:
                raise ValueError("token_positions must be provided when use_rope is True.")
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        
        output = scaled_dot_product_attention(q, k, v, ~causal_mask)
        
        return self.o_proj(rearrange(output, "b h t d -> b t (h d)"))


class TransformerBlock(Module):
    """
    Implement the pre-norm Transformer block as described in §3.5 and illustrated in Figure
        
    Parameters:
        d_model: int Dimensionality of the Transformer block inputs.
        
        num_heads: int Number of heads to use in multi-head self-attention.
        
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
    """
    def __init__(self,
                d_model: int,
                num_heads: int,
                d_ff: int,
                theta: float | None = None,
                max_seq_len: int | None = None,
                device=None,
                dtype=None,
                ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.rms_norm1 = RMSNorm(d_model, **factory_kwargs)
        # Test Case: MultiHeadSelfAttention <==> FlashAttention
        self.self_attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            rope_exist=True,
            **factory_kwargs,
            )

        self.rms_norm2 = RMSNorm(d_model, **factory_kwargs)
        self.ff = SwiGLU(d_model, d_ff)
    
    def forward(self,
                x: Tensor
                ):
        # Create token_positions
        seq_len = x.shape[1]
        token_positions = torch.arange(seq_len, device=x.device)

        attn_output = self.self_attn(self.rms_norm1(x),
                                         token_positions=token_positions
                                        )
        y = x + attn_output
        
        return y + self.ff(self.rms_norm2(y))


@dataclass
class GPT2Config:
    vocab_size: 50527
    context_length: 1024
    num_layers: 48
    d_model: 1600
    num_heads: 25
    d_ff: 6400


class TransformerLM(Module):
    """
    Time to put it all together! Implement the Transformer language model
    as described in §3.1 and illustrated in Figure 1. At minimum,
    your implementation should accept all the aforementioned 
    construction parameters for the Transformer block.
    
    Parameters:
        vocab_size: embedding matrix. int The size of the vocabulary,
        necessary for determining the dimensionality of the token
        
        context_length: int The maximum context length, necessary for
        determining the dimensionality of the position embedding matrix.
        
        num_layers: int The number of Transformer blocks to use.
    """
    def __init__(self,
                vocab_size: int,
                context_length: int,
                num_layers: int,
                d_model: int, 
                num_heads: int,
                d_ff: int, 
                rope_theta: float,
                ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.layers = ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    theta=rope_theta,
                    max_seq_len=context_length,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: Int[Tensor, "... sequence_length"]) -> Float[Tensor, "... sequence_length vocab_size"]:
        _, sequence_length = x.size()

        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_final(x)

        return self.lm_head(x)
    
    # Solution
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the lm_head parameters get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.lm_head.weight.numel()

        return n_params

    # Solution
    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ):
        """
        Args:
            x: LongTensor of shape `(1, sequence_length,)` or `(sequence_length, )`.
                Input IDs to condition on when generating.
            max_new_tokens: int
                Maximum number of tokens to generate.
            temperature: float
                Temperature to use during generation.
            top_k: int
                If provided, only sample from the `top_k` vocab items (by probability).
            eos_token_id: int
                If provided, stop generation when we generate this ID.

        Returns: A LongTensor of shape (max_new_tokens,) with the generated model output.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        original_sequence_length = x.size(-1)

        for _ in range(max_new_tokens):

            # Take the last `context_length` tokens if the input is
            # beyond the model's context length
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x

            # Get the logits from the model
            logits = self.forward(x)

            # Take the logits for the next token
            next_token_logits = logits[:, -1]

            # apply temperature scaling
            temperature_scaled_next_token_logits = next_token_logits / temperature

            # If top-k is provided, take the tokens with the highest score
            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )

                # Get the score of the kth item that we kept---items with lower scores should be masked.
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(topk_mask, float("-inf"))
            softmax = Softmax()
            next_token_probabilities = softmax(temperature_scaled_next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probabilities, 1)

            # End generation if we see the EOS token ID
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)

        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids
    
    # Solution
    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)

        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path)

        # Remove _orig_mod. prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        return model