from __future__ import annotations
import torch
import torch.nn as nn
from src.attention import GroupedQueryAttention
from src.config import MistralConfig
from src.ffn import FeedForwardNetwork
from src.normalization import RMSNorm

class MistralBlock(nn.Module):

    def __init__(self, config: MistralConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.ffn = FeedForwardNetwork(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = residual + x
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        return x

class MistralLM(nn.Module):

    def __init__(self, config: MistralConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MistralBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> int:
        return sum((p.numel() for p in self.parameters() if p.requires_grad))
