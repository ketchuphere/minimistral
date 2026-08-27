from __future__ import annotations
import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float=1e-06) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_squared = x.pow(2)
        mean_squared = x_squared.mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(mean_squared + self.eps)
        return self.weight * x_normed
