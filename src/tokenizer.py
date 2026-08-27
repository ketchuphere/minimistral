from __future__ import annotations
from typing import List
import tiktoken

class Tokenizer:

    def __init__(self, encoding_name: str='gpt2') -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name

    @property
    def vocab_size(self) -> int:
        return self._encoding.n_vocab

    def encode(self, text: str) -> List[int]:
        return self._encoding.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self._encoding.decode(token_ids)
