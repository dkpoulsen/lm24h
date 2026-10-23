"""Qwen3-architecture decoder-only transformer (GQA + QK-norm + SwiGLU + RoPE + RMSNorm, tied embeddings).

Chosen so checkpoints export 1:1 to HF `Qwen3ForCausalLM` (vLLM / llama.cpp / transformers compatible).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
import torch, torch.nn as nn, torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    n_layer: int = 24
    d_model: int = 1024
    n_head: int = 16
    n_kv_head: int = 4
    head_dim: int = 64
    ffn_dim: int = 3072
    max_seq_len: int = 2048
    rope_theta: float = 100_000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    init_std: float = 0.02

    def n_params(self, embeddings=True) -> int:
        d, hd = self.d_model, self.head_dim
        attn = d * self.n_head * hd + 2 * d * self.n_kv_head * hd + self.n_head * hd * d + 2 * hd
        mlp = 3 * d * self.ffn_dim
        n = self.n_layer * (attn + mlp + 2 * d) + d
        emb = self.vocab_size * d * (1 if self.tie_embeddings else 2)
        return n + (emb if embeddings else 0)

    def flops_per_token(self, seq_len: int) -> float:
        """Training FLOPs/token (fwd+bwd = 6N matmul params incl. lm_head + causal attention)."""
        n_matmul = self.n_params(embeddings=False) + self.vocab_size * self.d_model
        attn = 6 * self.n_layer * self.n_head * self.head_dim * seq_len  # 12*L*H*hd*S/2 (causal)
        return 6 * n_matmul + attn

    def to_dict(self): return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        dt = x.dtype; x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def rope_tables(seq_len, head_dim, theta, device=None):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    f = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    return f.cos(), f.sin()  # (S, hd/2)


def apply_rope(x, cos, sin):
    # HF/Qwen3 "rotate_half" convention: first half / second half of head_dim
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s = cos[None, None], sin[None, None]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.q_proj = nn.Linear(c.d_model, c.n_head * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.d_model, c.n_kv_head * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.d_model, c.n_kv_head * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.n_head * c.head_dim, c.d_model, bias=False)
        self.q_norm = RMSNorm(c.head_dim, c.norm_eps)
        self.k_norm = RMSNorm(c.head_dim, c.norm_eps)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape; c = self.c
        q = self.q_proj(x).view(B, S, c.n_head, c.head_dim)
        k = self.k_proj(x).view(B, S, c.n_kv_head, c.head_dim)
        v = self.v_proj(x).view(B, S, c.n_kv_head, c.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos[:S], sin[:S]), apply_rope(k, cos[:S], sin[:S])
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.o_proj(o.transpose(1, 2).reshape(B, S, -1))


class MLP(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(c.d_model, c.ffn_dim, bias=False)
        self.up_proj = nn.Linear(c.d_model, c.ffn_dim, bias=False)
        self.down_proj = nn.Linear(c.ffn_dim, c.d_model, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_layernorm = RMSNorm(c.d_model, c.norm_eps)
        self.self_attn = Attention(c)
        self.post_attention_layernorm = RMSNorm(c.d_model, c.norm_eps)
        self.mlp = MLP(c)
    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Transformer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.d_model, c.norm_eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings: self.lm_head.weight = self.embed_tokens.weight
        cos, sin = rope_tables(c.max_seq_len, c.head_dim, c.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # GPT-2 style depth-scaled init on residual-output projections
        for n, p in self.named_parameters():
            if n.endswith("o_proj.weight") or n.endswith("down_proj.weight"):
                nn.init.normal_(p, 0.0, c.init_std / math.sqrt(2 * c.n_layer))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, self.c.init_std)

    def forward(self, idx, targets=None, ignore_index=-100):
        x = self.embed_tokens(idx)
        for blk in self.layers: x = blk(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        if targets is None:
            return self.lm_head(x)
        logits = self.lm_head(x).float()
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index)
