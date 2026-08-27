from __future__ import annotations
import torch
import torch.nn.functional as F
from src.model import MistralLM
from src.tokenizer import Tokenizer

@torch.no_grad()
def generate(model: MistralLM, tokenizer: Tokenizer, prompt: str, device: str, max_new_tokens: int=200, temperature: float=1.0, top_p: float=0.9, greedy: bool=False, seed: int | None=None) -> str:
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    max_seq_len = model.config.max_seq_len
    token_ids = tokenizer.encode(prompt)
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        context = tokens[:, -max_seq_len:]
        logits = model(context)
        next_logits = logits[:, -1, :]
        if greedy:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            scaled_logits = next_logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            to_remove = cumulative_probs - sorted_probs > top_p
            sorted_probs[to_remove] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, sampled_idx)
        tokens = torch.cat([tokens, next_token], dim=1)
    generated_ids = tokens[0].tolist()
    return tokenizer.decode(generated_ids)
