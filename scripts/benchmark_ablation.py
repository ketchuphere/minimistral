from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from src.config import MistralConfig, TrainConfig
from src.dataset import download_text, load_and_split
from src.evaluation import evaluate_split
from src.generation import generate
from src.model import MistralLM
from src.tokenizer import Tokenizer
from src.trainer import Trainer
from src.utils import get_device, set_seed
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
VARIANTS = {'full_mini_mistral': {}, 'without_gqa': {'use_gqa': False}, 'without_sliding_window': {'use_sliding_window': False}, 'without_rope': {'use_rope': False}, 'without_swiglu': {'use_swiglu': False}}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the MiniMistral ablation study.')
    parser.add_argument('--config', type=str, default='configs/mini_mistral.yaml')
    parser.add_argument('--max_steps', type=int, default=300, help='Steps per variant (default kept small for CPU feasibility).')
    parser.add_argument('--output', type=str, default='results/ablation_results.json')
    return parser.parse_args()

def load_config(path: str) -> tuple[MistralConfig, TrainConfig]:
    import yaml
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    model_cfg = MistralConfig.from_dict(raw.get('model', {}))
    train_cfg = TrainConfig.from_dict(raw.get('training', {}))
    return (model_cfg, train_cfg)

def main() -> None:
    args = parse_args()
    base_model_cfg, train_cfg = load_config(args.config)
    train_cfg.max_steps = args.max_steps
    train_cfg.eval_interval = max(args.max_steps // 5, 1)
    device = get_device()
    logger.info('Using device: %s | steps per variant: %d', device, args.max_steps)
    tokenizer = Tokenizer()
    raw_text = download_text(train_cfg.data_url, train_cfg.data_path)
    train_data, val_data = load_and_split(raw_text, tokenizer, train_cfg.train_val_split)
    results = {}
    for variant_name, overrides in VARIANTS.items():
        logger.info('\n=== Variant: %s (%s) ===', variant_name, overrides or 'no ablation, full model')
        set_seed(train_cfg.seed)
        variant_cfg = MistralConfig.from_dict({**base_model_cfg.to_dict(), **overrides})
        variant_train_cfg = TrainConfig.from_dict(train_cfg.to_dict())
        variant_train_cfg.checkpoint_path = f'checkpoints/ablation_{variant_name}.pt'
        variant_train_cfg.resume_from = None
        model = MistralLM(variant_cfg).to(device)
        num_params = model.count_parameters()
        trainer = Trainer(model, variant_cfg, variant_train_cfg, train_data, val_data, device)
        trainer.train()
        eval_results = evaluate_split(model, variant_cfg, val_data, device, batch_size=train_cfg.batch_size, seq_len=train_cfg.seq_len, num_batches=30)
        start = time.perf_counter()
        generate(model, tokenizer, 'JULIET:\n', device, max_new_tokens=50, temperature=0.8, top_p=0.9, seed=0)
        elapsed = time.perf_counter() - start
        results[variant_name] = {'config_overrides': overrides, 'parameter_count': num_params, 'final_val_loss': eval_results['loss'], 'perplexity': eval_results['perplexity'], 'generation_latency_seconds_50_tokens': round(elapsed, 4)}
        logger.info('%s -> val_loss: %.4f | perplexity: %.2f | params: %s', variant_name, eval_results['loss'], eval_results['perplexity'], f'{num_params:,}')
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info('\nSaved ablation results to %s', args.output)
if __name__ == '__main__':
    main()
