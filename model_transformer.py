from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    mlp_mult: int
    mlp_topk: int | None  # if None, just apply ReLU (no TopK)
    use_relu_before_topk: bool = True
    tie_weights: bool = True


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] or [B, T, ...]
        # Normalize by RMS along the last dimension
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Pre-compute cos and sin values
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pre-compute rotation matrices for efficiency
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.
        q, k: [B, nh, T, hs] (batch, num_heads, seq_len, head_dim)
        """
        seq_len = q.shape[2]
        
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        # Apply rotation: [x, y] -> [x*cos - y*sin, x*sin + y*cos]
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rot, k_rot
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)


class TopK(nn.Module):
    """Faster TopK: selects top k activations per token. If k is None, just applies ReLU."""
    def __init__(self, k: int | None, use_relu: bool = True):
        super().__init__()
        self.k = k
        self.use_relu = use_relu
        self.last_mean_active_per_token: float | None = None
        # Use register_buffer for statistics to ensure proper device placement
        self.register_buffer('_last_feature_counts', None, persistent=False)
        self.last_n_tokens: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        b, t, d = x.shape
        
        # If k is None, just apply ReLU and return (no TopK)
        if self.k is None:
            if self.use_relu:
                x = F.relu(x, inplace=False)
            self._last_feature_counts = torch.full((d,), b*t, dtype=torch.int64, device=x.device) if self.use_relu else torch.zeros(d, dtype=torch.int64, device=x.device)
            self.last_n_tokens = b * t
            self.last_mean_active_per_token = float((x != 0).sum().item() / max(1, b * t)) if self.use_relu else 0.0
            return x
        
        # Fast path for k=0
        if self.k == 0:
            self._last_feature_counts = torch.zeros(d, dtype=torch.int64, device=x.device)
            self.last_n_tokens = b * t
            self.last_mean_active_per_token = 0.0
            return torch.zeros_like(x)
        
        # Apply ReLU if needed (fused operation)
        if self.use_relu:
            x = F.relu(x, inplace=False)
        
        # Handle case when k >= d (keep everything)
        if self.k >= d:
            self._last_feature_counts = torch.full((d,), b*t, dtype=torch.int64, device=x.device)
            self.last_n_tokens = b * t
            self.last_mean_active_per_token = float(d)
            return x
        
        # Efficient topk with vectorized operations
        k_actual = self.k
        
        # Reshape for faster topk computation
        x_2d = x.view(-1, d)  # [B*T, D]
        
        # Get topk values and indices
        topk_vals, topk_idx = torch.topk(x_2d, k=k_actual, dim=1, sorted=False)
        
        # Create output using scatter_ with better memory access pattern
        y = torch.zeros_like(x_2d)
        y.scatter_(1, topk_idx, topk_vals)
        
        # Reshape back
        y = y.view(b, t, d)
        
        # Count feature occurrences efficiently
        flat_idx = topk_idx.flatten()
        self._last_feature_counts = torch.bincount(flat_idx, minlength=d).to(torch.int64)
        self.last_n_tokens = b * t
        
        # Compute statistics
        self.last_mean_active_per_token = k_actual
        
        return y
    
    @property
    def last_feature_counts(self) -> torch.Tensor | None:
        return self._last_feature_counts


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=True)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=True)
        
        # RoPE positional embedding
        self.rope = RoPE(self.head_dim, max_seq_len=cfg.block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(c, dim=2)

        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)  # [B, nh, T, hs]
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k)

        # Use fused scaled dot product attention when available (often much faster).
        # dropout_p=0.0 (we run no-dropout experiments), is_causal=True for autoregressive masking.
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)  # [B, nh, T, hs]
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        hidden = cfg.mlp_mult * cfg.n_embd
        self.fc1 = nn.Linear(cfg.n_embd, hidden, bias=True)
        self.act = TopK(
            k=cfg.mlp_topk,
            use_relu=cfg.use_relu_before_topk,
        )
        self.fc2 = nn.Linear(hidden, cfg.n_embd, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SparseTransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"sequence length {t} > block_size {self.cfg.block_size}")

        x = self.wte(idx)  # [B, T, C] - RoPE is applied in attention layers

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def batchtopk_stats(self) -> dict[str, float]:
        vals = []
        for blk in self.blocks:
            v = blk.mlp.act.last_mean_active_per_token
            if v is not None:
                vals.append(v)
        if not vals:
            return {}
        return {
            "batchtopk_mean_active_per_token_mean": float(sum(vals) / len(vals)),
            "batchtopk_mean_active_per_token_min": float(min(vals)),
            "batchtopk_mean_active_per_token_max": float(max(vals)),
        }


