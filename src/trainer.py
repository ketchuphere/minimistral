from __future__ import annotations
import logging
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from src.config import MistralConfig, TrainConfig
from src.dataset import get_batch
from src.model import MistralLM
from src.utils import load_checkpoint, save_checkpoint
logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self, model: MistralLM, config: MistralConfig, train_config: TrainConfig, train_data: torch.Tensor, val_data: torch.Tensor, device: str) -> None:
        self.model = model
        self.config = config
        self.tc = train_config
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=train_config.max_steps)
        self.start_step = 0
        self.train_losses: List[float] = []
        self.train_eval_losses: List[float] = []
        self.val_eval_losses: List[float] = []
        self.eval_steps: List[int] = []
        if train_config.resume_from:
            self._resume(train_config.resume_from)

    def _resume(self, path: str) -> None:
        checkpoint = load_checkpoint(path, self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_step = checkpoint.get('step', 0)
        logger.info('Resumed training from %s at step %d', path, self.start_step)

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        self.model.eval()
        losses: Dict[str, float] = {}
        for split_name, split_data in [('train', self.train_data), ('val', self.val_data)]:
            split_losses = []
            for _ in range(self.tc.eval_steps):
                x, y = get_batch(split_data, self.tc.batch_size, self.tc.seq_len, self.device)
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                split_losses.append(loss.item())
            losses[split_name] = sum(split_losses) / len(split_losses)
        self.model.train()
        return losses

    def train(self) -> Dict[str, List[float]]:
        logger.info('Starting training...')
        self.model.train()
        for step in range(self.start_step, self.tc.max_steps):
            if step % self.tc.eval_interval == 0:
                losses = self.estimate_loss()
                self.train_eval_losses.append(losses['train'])
                self.val_eval_losses.append(losses['val'])
                self.eval_steps.append(step)
                logger.info('Step %4d | Train Loss: %.4f | Val Loss: %.4f', step, losses['train'], losses['val'])
            x, y = get_batch(self.train_data, self.tc.batch_size, self.tc.seq_len, self.device)
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
            self.train_losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.tc.grad_clip_norm)
            self.optimizer.step()
            self.scheduler.step()
        save_checkpoint(self.tc.checkpoint_path, self.model, self.config, optimizer=self.optimizer, scheduler=self.scheduler, step=self.tc.max_steps)
        logger.info('Training complete. Checkpoint saved to %s', self.tc.checkpoint_path)
        return {'train_losses': self.train_losses, 'train_eval_losses': self.train_eval_losses, 'val_eval_losses': self.val_eval_losses, 'eval_steps': self.eval_steps}
