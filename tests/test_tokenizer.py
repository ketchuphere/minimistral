from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tokenizer import Tokenizer

class TestTokenizer:

    def test_vocab_size(self):
        tok = Tokenizer()
        assert tok.vocab_size == 50257

    def test_encode_returns_int_list(self):
        tok = Tokenizer()
        ids = tok.encode('Hello, world!')
        assert isinstance(ids, list)
        assert all((isinstance(i, int) for i in ids))
        assert len(ids) > 0

    def test_roundtrip(self):
        tok = Tokenizer()
        text = 'To be, or not to be, that is the question.'
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_empty_string(self):
        tok = Tokenizer()
        assert tok.encode('') == []
        assert tok.decode([]) == ''
