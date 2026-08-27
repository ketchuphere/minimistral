from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from src.generation import generate
from src.tokenizer import Tokenizer
from src.utils import get_device, load_model_from_checkpoint
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark MiniMistral.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/mistral_mini.pt')
    parser.add_argument('--prompt', type=str, default='JULIET:\n')
    parser.add_argument('--lengths', type=int, nargs='+', default=[20, 50, 100], help='Generation lengths (max_new_tokens) to benchmark.')
    parser.add_argument('--output', type=str, default='results/benchmark_results.json')
    return parser.parse_args()

def model_size_mb(model: torch.nn.Module) -> float:
    total_bytes = sum((p.numel() * p.element_size() for p in model.parameters()))
    return total_bytes / 1024 ** 2

def main() -> None:
    args = parse_args()
    device = get_device()
    logger.info('Using device: %s', device)
    model = load_model_from_checkpoint(args.checkpoint, device)
    tokenizer = Tokenizer()
    num_params = model.count_parameters()
    size_mb = model_size_mb(model)
    results = {'device': device, 'parameter_count': num_params, 'model_size_mb': round(size_mb, 3), 'generation_runs': []}
    if device == 'cuda':
        results['gpu_name'] = torch.cuda.get_device_name(0)
    for length in args.lengths:
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        generate(model, tokenizer, prompt=args.prompt, device=device, max_new_tokens=length, temperature=0.8, top_p=0.9, seed=0)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tokens_per_sec = length / elapsed if elapsed > 0 else float('inf')
        run_result = {'max_new_tokens': length, 'latency_seconds': round(elapsed, 4), 'tokens_per_second': round(tokens_per_sec, 2)}
        if device == 'cuda':
            run_result['peak_memory_mb'] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
        results['generation_runs'].append(run_result)
        logger.info('Length %4d | latency: %.3fs | tokens/sec: %.2f', length, elapsed, tokens_per_sec)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info('Saved benchmark results to %s', args.output)
if __name__ == '__main__':
    main()
