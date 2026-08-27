from __future__ import annotations
import os
import urllib.request
from typing import Tuple
import torch
from src.tokenizer import Tokenizer

def download_text(url: str, dest_path: str) -> str:
    os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
    if not os.path.exists(dest_path):
        urllib.request.urlretrieve(url, dest_path)
    with open(dest_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_and_split(raw_text: str, tokenizer: Tokenizer, train_val_split: float=0.9) -> Tuple[torch.Tensor, torch.Tensor]:
    all_token_ids = tokenizer.encode(raw_text)
    data = torch.tensor(all_token_ids, dtype=torch.long)
    split = int(train_val_split * len(data))
    train_data = data[:split]
    val_data = data[split:]
    return (train_data, val_data)

def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        raise ValueError(f'Dataset too small ({len(data)} tokens) for seq_len={seq_len}.')
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + seq_len] for s in starts])
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return (x.to(device), y.to(device))
