from __future__ import annotations
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.generation import generate
from src.tokenizer import Tokenizer
from src.utils import get_device, load_model_from_checkpoint

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate text with MiniMistral.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/mistral_mini.pt')
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--greedy', action='store_true', help='Use deterministic greedy decoding.')
    parser.add_argument('--seed', type=int, default=None, help='Seed for deterministic sampling.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = get_device()
    model = load_model_from_checkpoint(args.checkpoint, device)
    tokenizer = Tokenizer()
    output = generate(model, tokenizer, prompt=args.prompt, device=device, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, greedy=args.greedy, seed=args.seed)
    print(output)
if __name__ == '__main__':
    main()
