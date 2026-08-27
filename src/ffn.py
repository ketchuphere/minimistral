from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import MistralConfig

class FeedForwardNetwork(nn.Module):

    def __init__(self, config: MistralConfig) -> None:
        super().__init__()
        self.use_swiglu = config.use_swiglu
        if self.use_swiglu:
            self.gate_proj = nn.Linear(config.hidden_size, config.ffn_hidden, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.ffn_hidden, bias=False)
            self.down_proj = nn.Linear(config.ffn_hidden, config.hidden_size, bias=False)
        else:
            self.fc1 = nn.Linear(config.hidden_size, config.ffn_hidden, bias=False)
            self.fc2 = nn.Linear(config.ffn_hidden, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            fused = gate * up
            return self.down_proj(fused)
        else:
            return self.fc2(F.relu(self.fc1(x)))
