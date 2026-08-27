from __future__ import annotations
import math
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from src.config import MistralConfig
from src.dataset import get_batch
from src.model import MistralLM

@torch.no_grad()
def evaluate_split(model: MistralLM, config: MistralConfig, data: torch.Tensor, device: str, batch_size: int=16, seq_len: int=128, num_batches: int=50) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    for _ in range(num_batches):
        x, y = get_batch(data, batch_size, seq_len, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    return {'loss': mean_loss, 'perplexity': perplexity}

def plot_training_curve(eval_steps: List[int], train_eval_losses: List[float], val_eval_losses: List[float], output_path: str, train_losses: Optional[List[float]]=None) -> None:
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.figure(figsize=(10, 5))
    if train_losses:
        plt.plot(range(len(train_losses)), train_losses, color='steelblue', linewidth=1, alpha=0.25, label='Per-step train loss')
    plt.plot(eval_steps, train_eval_losses, marker='o', linewidth=2, label='Training Loss')
    plt.plot(eval_steps, val_eval_losses, marker='o', linewidth=2, label='Validation Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('MiniMistral Training Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
