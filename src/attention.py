from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import MistralConfig
from src.rope import RotaryEmbedding

class GroupedQueryAttention(nn.Module):

    def __init__(self, config: MistralConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads if config.use_gqa else config.num_heads
        self.head_dim = config.head_dim
        self.window_size = config.window_size
        self.use_sliding_window = config.use_sliding_window
        self.use_rope = config.use_rope
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f'num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})')
        self.kv_groups = self.num_heads // self.num_kv_heads
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            q, k = self.rope(q, k)
        k = k.repeat_interleave(self.kv_groups, dim=1)
        v = v.repeat_interleave(self.kv_groups, dim=1)
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        causal_mask = self._make_mask(seq_len, x.device)
        scores = scores + causal_mask
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch, seq_len, -1)
        return self.o_proj(context)

    def _make_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        causal = row_idx >= col_idx
        if self.use_sliding_window:
            window = row_idx - col_idx <= self.window_size
            allowed = causal & window
        else:
            allowed = causal
        mask = torch.where(allowed, torch.zeros_like(allowed, dtype=torch.float), torch.full_like(allowed, float('-inf'), dtype=torch.float))
        return mask.unsqueeze(0).unsqueeze(0)
