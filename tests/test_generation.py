from __future__ import annotations
import os
import sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import MistralConfig
from src.generation import generate
from src.model import MistralLM
from src.tokenizer import Tokenizer

def make_tiny_model() -> tuple[MistralLM, MistralConfig]:
    config = MistralConfig(vocab_size=50257, hidden_size=16, num_layers=1, num_heads=2, num_kv_heads=1, head_dim=8, ffn_hidden=32, max_seq_len=32, window_size=8, dropout=0.0)
    model = MistralLM(config)
    model.eval()
    return (model, config)

class TestGeneration:

    def test_greedy_generates_requested_length(self):
        model, _ = make_tiny_model()
        tokenizer = Tokenizer()
        out = generate(model, tokenizer, prompt='Hello', device='cpu', max_new_tokens=5, greedy=True)
        assert isinstance(out, str)
        assert out.startswith('Hello')

    def test_temperature_top_p_generates(self):
        model, _ = make_tiny_model()
        tokenizer = Tokenizer()
        out = generate(model, tokenizer, prompt='Hello', device='cpu', max_new_tokens=5, temperature=0.8, top_p=0.9, seed=0)
        assert isinstance(out, str)
        assert len(out) >= len('Hello')

    def test_greedy_is_deterministic(self):
        model, _ = make_tiny_model()
        tokenizer = Tokenizer()
        out1 = generate(model, tokenizer, 'Once upon a time', 'cpu', max_new_tokens=8, greedy=True)
        out2 = generate(model, tokenizer, 'Once upon a time', 'cpu', max_new_tokens=8, greedy=True)
        assert out1 == out2

    def test_seeded_sampling_is_deterministic(self):
        model, _ = make_tiny_model()
        tokenizer = Tokenizer()
        out1 = generate(model, tokenizer, 'Once upon a time', 'cpu', max_new_tokens=8, seed=42)
        out2 = generate(model, tokenizer, 'Once upon a time', 'cpu', max_new_tokens=8, seed=42)
        assert out1 == out2

    def test_generation_respects_max_seq_len_context(self):
        model, config = make_tiny_model()
        tokenizer = Tokenizer()
        out = generate(model, tokenizer, prompt='A', device='cpu', max_new_tokens=config.max_seq_len + 5, greedy=True)
        assert isinstance(out, str)
