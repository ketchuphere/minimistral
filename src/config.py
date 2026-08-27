from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict

@dataclass
class MistralConfig:
    vocab_size: int = 50257
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 32
    ffn_hidden: int = 512
    max_seq_len: int = 512
    window_size: int = 64
    dropout: float = 0.1
    rope_theta: float = 10000.0
    use_gqa: bool = True
    use_sliding_window: bool = True
    use_rope: bool = True
    use_swiglu: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MistralConfig':
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

@dataclass
class TrainConfig:
    batch_size: int = 16
    seq_len: int = 128
    learning_rate: float = 0.0003
    max_steps: int = 1000
    eval_interval: int = 100
    eval_steps: int = 20
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    seed: int = 1337
    checkpoint_path: str = 'checkpoints/mistral_mini.pt'
    resume_from: str | None = None
    data_url: str = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    data_path: str = 'data/shakespeare.txt'
    train_val_split: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainConfig':
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)
