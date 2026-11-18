"""GPT with explicit projections, causal attention, and tied character embeddings."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    n_layer: int = 6
    d_model: int = 384
    n_head: int = 6
    context_length: int = 256
    dropout: float = 0.1
    attention: str = "sdpa"

    def __post_init__(self):
        if min(self.vocab_size, self.n_layer, self.d_model, self.n_head, self.context_length) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        if not 0 <= self.dropout < 1 or self.attention not in ("sdpa", "manual"):
            raise ValueError("Invalid dropout or attention backend")

    def expected_parameters(self):
        d = self.d_model
        return d * (self.vocab_size + self.context_length) + self.n_layer * (12 * d**2 + 13 * d) + 2 * d


class CausalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.backend = cfg.attention
        self.rate = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.project = nn.Linear(cfg.d_model, cfg.d_model)
        self.residual_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, allowed=None):
        b, t, d = x.shape
        # Each projection becomes [batch, head, time, head_dim].
        q, k, v = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        dropout_p = self.rate if self.training else 0.0
        if self.backend == "sdpa":
            z = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed, is_causal=allowed is None, dropout_p=dropout_p,
            )
        else:
            if allowed is None:
                allowed = torch.ones((t, t), dtype=torch.bool, device=x.device).tril()
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~allowed, -torch.inf)
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
            z = F.dropout(probs, p=dropout_p, training=self.training) @ v
        return self.residual_dropout(self.project(z.transpose(1, 2).reshape(b, t, d)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.ln_attention = nn.LayerNorm(d, eps=1e-5)
        self.attention = CausalAttention(cfg)
        self.ln_mlp = nn.LayerNorm(d, eps=1e-5)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(cfg.dropout))

    def forward(self, x, allowed):
        x = x + self.attention(self.ln_attention(x), allowed)
        return x + self.mlp(self.ln_mlp(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.final_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.apply(self._init_weights)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Assign after initialization so the shared parameter gets only one model init.
        self.head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, ids: Tensor, *, valid_keys: Tensor | None = None, last_only=False):
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.config.context_length:
            raise ValueError(f"Expected [batch,time] with 1 <= time <= {self.config.context_length}, got {ids.shape}")
        t = ids.shape[1]
        pos = torch.arange(t, device=ids.device)
        allowed = None
        if valid_keys is not None:
            if valid_keys.shape != ids.shape or valid_keys.dtype != torch.bool:
                raise ValueError("valid_keys must be Boolean with the same shape as ids")
            causal = pos[:, None] >= pos[None, :]
            allowed = causal[None, None, :, :] & valid_keys[:, None, None, :]
            # Only padded query rows can lack eligible keys. Self-attention there
            # keeps the manual softmax finite; those rows never enter the loss.
            allowed = allowed | (~allowed.any(dim=-1, keepdim=True) & (pos[:, None] == pos[None, :]))
        # Ordinary right padding is already invisible to valid queries by causality.
        x = self.embedding_dropout(self.token_embedding(ids) + self.position_embedding(pos))
        for block in self.blocks:
            x = block(x, allowed)
        return self.head(self.final_norm(x[:, -1:, :] if last_only else x))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def cross_entropy_sum(logits, targets):
    if logits.shape[:-1] != targets.shape:
        raise ValueError("Targets must match all non-vocabulary dimensions of logits")
    return F.cross_entropy(logits.flatten(end_dim=-2), targets.reshape(-1),
                           ignore_index=-100, reduction="sum")
