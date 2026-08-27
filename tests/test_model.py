from __future__ import annotations
import os
import sys
import pytest
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.attention import GroupedQueryAttention
from src.config import MistralConfig
from src.ffn import FeedForwardNetwork
from src.model import MistralBlock, MistralLM
from src.normalization import RMSNorm
from src.rope import RotaryEmbedding

@pytest.fixture
def tiny_config() -> MistralConfig:
    return MistralConfig(vocab_size=100, hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=8, ffn_hidden=64, max_seq_len=64, window_size=8, dropout=0.0)

class TestRMSNorm:

    def test_output_shape(self):
        norm = RMSNorm(hidden_size=16)
        x = torch.randn(2, 5, 16)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalizes_scale(self):
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(4, 10, 64) * 100
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.01)

class TestRoPE:

    def test_output_shapes(self):
        rope = RotaryEmbedding(head_dim=8)
        q = torch.randn(2, 4, 10, 8)
        k = torch.randn(2, 2, 10, 8)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_changes_values(self):
        rope = RotaryEmbedding(head_dim=8)
        q = torch.randn(1, 1, 5, 8)
        k = torch.randn(1, 1, 5, 8)
        q_rot, _ = rope(q, k)
        assert not torch.allclose(q_rot[:, :, 1:], q[:, :, 1:])

class TestAttention:

    def test_output_shape(self, tiny_config: MistralConfig):
        attn = GroupedQueryAttention(tiny_config)
        x = torch.randn(2, 10, tiny_config.hidden_size)
        out = attn(x)
        assert out.shape == x.shape

    def test_gqa_head_expansion(self, tiny_config: MistralConfig):
        attn = GroupedQueryAttention(tiny_config)
        assert attn.kv_groups == tiny_config.num_heads // tiny_config.num_kv_heads

    def test_invalid_head_config_raises(self, tiny_config: MistralConfig):
        tiny_config.num_heads = 5
        tiny_config.num_kv_heads = 2
        with pytest.raises(ValueError):
            GroupedQueryAttention(tiny_config)

    def test_sliding_window_masks_far_tokens(self, tiny_config: MistralConfig):
        tiny_config.window_size = 0
        attn = GroupedQueryAttention(tiny_config)
        mask = attn._make_mask(seq_len=6, device=torch.device('cpu'))
        row = mask[0, 0, 3]
        assert row[3].item() == 0.0
        assert row[2].item() == float('-inf')

    def test_causal_masks_future_tokens(self, tiny_config: MistralConfig):
        attn = GroupedQueryAttention(tiny_config)
        mask = attn._make_mask(seq_len=6, device=torch.device('cpu'))
        row = mask[0, 0, 2]
        assert row[3].item() == float('-inf')
        assert row[1].item() == 0.0

    def test_ablation_no_gqa_uses_full_heads(self, tiny_config: MistralConfig):
        tiny_config.use_gqa = False
        attn = GroupedQueryAttention(tiny_config)
        assert attn.num_kv_heads == tiny_config.num_heads

class TestFFN:

    def test_output_shape_swiglu(self, tiny_config: MistralConfig):
        ffn = FeedForwardNetwork(tiny_config)
        x = torch.randn(2, 10, tiny_config.hidden_size)
        out = ffn(x)
        assert out.shape == x.shape

    def test_output_shape_ablation_relu(self, tiny_config: MistralConfig):
        tiny_config.use_swiglu = False
        ffn = FeedForwardNetwork(tiny_config)
        x = torch.randn(2, 10, tiny_config.hidden_size)
        out = ffn(x)
        assert out.shape == x.shape

class TestModel:

    def test_block_forward_shape(self, tiny_config: MistralConfig):
        block = MistralBlock(tiny_config)
        x = torch.randn(2, 10, tiny_config.hidden_size)
        out = block(x)
        assert out.shape == x.shape

    def test_full_model_forward_shape(self, tiny_config: MistralConfig):
        model = MistralLM(tiny_config)
        tokens = torch.randint(0, tiny_config.vocab_size, (2, 10))
        logits = model(tokens)
        assert logits.shape == (2, 10, tiny_config.vocab_size)

    def test_weight_tying(self, tiny_config: MistralConfig):
        model = MistralLM(tiny_config)
        assert model.lm_head.weight is model.embedding.weight

    def test_parameter_count_positive(self, tiny_config: MistralConfig):
        model = MistralLM(tiny_config)
        assert model.count_parameters() > 0

    def test_loss_computable_and_finite(self, tiny_config: MistralConfig):
        import torch.nn.functional as F
        model = MistralLM(tiny_config)
        x = torch.randint(0, tiny_config.vocab_size, (2, 10))
        y = torch.randint(0, tiny_config.vocab_size, (2, 10))
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, tiny_config.vocab_size), y.view(-1))
        assert torch.isfinite(loss)

    def test_checkpoint_save_and_load(self, tiny_config: MistralConfig, tmp_path):
        from src.utils import load_model_from_checkpoint, save_checkpoint
        model = MistralLM(tiny_config)
        path = str(tmp_path / 'ckpt.pt')
        save_checkpoint(path, model, tiny_config, step=10)
        loaded = load_model_from_checkpoint(path, device='cpu')
        assert loaded.count_parameters() == model.count_parameters()
        tokens = torch.randint(0, tiny_config.vocab_size, (1, 5))
        with torch.no_grad():
            out1 = model(tokens)
            out2 = loaded(tokens)
        assert torch.allclose(out1, out2, atol=1e-05)
