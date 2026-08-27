from __future__ import annotations
import os
import random
from typing import Any, Dict, Optional
import numpy as np
import torch
from src.config import MistralConfig
from src.model import MistralLM

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def save_checkpoint(path: str, model: MistralLM, config: MistralConfig, optimizer: Optional[torch.optim.Optimizer]=None, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None, step: int=0, extra: Optional[Dict[str, Any]]=None) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload: Dict[str, Any] = {'model_state_dict': model.state_dict(), 'config': config.to_dict(), 'step': step}
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        payload['scheduler_state_dict'] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)

def load_checkpoint(path: str, device: str) -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)

def load_model_from_checkpoint(path: str, device: str) -> MistralLM:
    checkpoint = load_checkpoint(path, device)
    config = MistralConfig.from_dict(checkpoint['config'])
    model = MistralLM(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
