from __future__ import annotations
import argparse
import json
import logging
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import download_text, load_and_split
from src.evaluation import evaluate_split
from src.tokenizer import Tokenizer
from src.utils import get_device, load_model_from_checkpoint
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate MiniMistral.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/mistral_mini.pt')
    parser.add_argument('--data_url', type=str, default='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
    parser.add_argument('--data_path', type=str, default='data/shakespeare.txt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--num_batches', type=int, default=50)
    parser.add_argument('--output', type=str, default='results/evaluation_results.json')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = get_device()
    logger.info('Using device: %s', device)
    model = load_model_from_checkpoint(args.checkpoint, device)
    logger.info('Loaded model with %s parameters', f'{model.count_parameters():,}')
    tokenizer = Tokenizer()
    raw_text = download_text(args.data_url, args.data_path)
    train_data, val_data = load_and_split(raw_text, tokenizer, train_val_split=0.9)
    train_results = evaluate_split(model, model.config, train_data, device, args.batch_size, args.seq_len, args.num_batches)
    val_results = evaluate_split(model, model.config, val_data, device, args.batch_size, args.seq_len, args.num_batches)
    results = {'checkpoint': args.checkpoint, 'num_batches': args.num_batches, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'train': train_results, 'validation': val_results}
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info('Train      | loss: %.4f | perplexity: %.2f', train_results['loss'], train_results['perplexity'])
    logger.info('Validation | loss: %.4f | perplexity: %.2f', val_results['loss'], val_results['perplexity'])
    logger.info('Saved evaluation results to %s', args.output)
if __name__ == '__main__':
    main()
