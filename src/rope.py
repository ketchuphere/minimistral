from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):

    def __init__(self, head_dim: int, rope_theta: float=10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        inv_freq = 1.0 / rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer('inv_freq', inv_freq)

    def _compute_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return (emb.cos(), emb.sin())

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        cos, sin = self._compute_cos_sin(seq_len, q.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q_rotated = q * cos + self._rotate_half(q) * sin
        k_rotated = k * cos + self._rotate_half(k) * sin
        return (q_rotated, k_rotated)
