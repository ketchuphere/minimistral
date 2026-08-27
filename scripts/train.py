from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import MistralConfig, TrainConfig
from src.dataset import download_text, load_and_split
from src.evaluation import plot_training_curve
from src.model import MistralLM
from src.tokenizer import Tokenizer
from src.trainer import Trainer
from src.utils import get_device, set_seed
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train MiniMistral.')
    parser.add_argument('--config', type=str, default='configs/mini_mistral.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume from.')
    parser.add_argument('--max_steps', type=int, default=None, help='Override max_steps (e.g. for a smoke test).')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--results_path', type=str, default='results/training_metrics_new.json')
    parser.add_argument('--plot_path', type=str, default='results/plots/training_loss.png')
    return parser.parse_args()

def load_config(path: str) -> tuple[MistralConfig, TrainConfig]:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    model_cfg = MistralConfig.from_dict(raw.get('model', {}))
    train_cfg = TrainConfig.from_dict(raw.get('training', {}))
    return (model_cfg, train_cfg)

def main() -> None:
    args = parse_args()
    model_cfg, train_cfg = load_config(args.config)
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.seq_len is not None:
        train_cfg.seq_len = args.seq_len
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate
    if args.seed is not None:
        train_cfg.seed = args.seed
    if args.checkpoint_path is not None:
        train_cfg.checkpoint_path = args.checkpoint_path
    if args.resume is not None:
        train_cfg.resume_from = args.resume
    set_seed(train_cfg.seed)
    device = get_device()
    logger.info('Using device: %s', device)
    tokenizer = Tokenizer()
    raw_text = download_text(train_cfg.data_url, train_cfg.data_path)
    train_data, val_data = load_and_split(raw_text, tokenizer, train_cfg.train_val_split)
    logger.info('Train tokens: %d | Val tokens: %d', len(train_data), len(val_data))
    model = MistralLM(model_cfg).to(device)
    logger.info('Model parameters: %s', f'{model.count_parameters():,}')
    trainer = Trainer(model, model_cfg, train_cfg, train_data, val_data, device)
    metrics = trainer.train()
    os.makedirs(os.path.dirname(args.results_path) or '.', exist_ok=True)
    with open(args.results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info('Saved training metrics to %s', args.results_path)
    if metrics['eval_steps']:
        plot_training_curve(metrics['eval_steps'], metrics['train_eval_losses'], metrics['val_eval_losses'], args.plot_path, train_losses=metrics['train_losses'])
        logger.info('Saved training curve plot to %s', args.plot_path)
if __name__ == '__main__':
    main()
