from __future__ import annotations
import os
import sys
import pytest
from fastapi.testclient import TestClient
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.main import app, state
from src.config import MistralConfig
from src.model import MistralLM
from src.tokenizer import Tokenizer

@pytest.fixture
def client_with_model():
    config = MistralConfig(vocab_size=50257, hidden_size=16, num_layers=1, num_heads=2, num_kv_heads=1, head_dim=8, ffn_hidden=32, max_seq_len=32, window_size=8, dropout=0.0)
    model = MistralLM(config)
    model.eval()
    with TestClient(app) as client:
        state['model'] = model
        state['tokenizer'] = Tokenizer()
        state['device'] = 'cpu'
        yield client

class TestHealthEndpoint:

    def test_health_returns_healthy(self, client_with_model):
        response = client_with_model.get('/health')
        assert response.status_code == 200
        assert response.json() == {'status': 'healthy'}

class TestGenerateEndpoint:

    def test_generate_returns_text(self, client_with_model):
        response = client_with_model.post('/generate', json={'prompt': 'Hello', 'max_new_tokens': 5})
        assert response.status_code == 200
        body = response.json()
        assert 'generated_text' in body
        assert isinstance(body['generated_text'], str)

    def test_generate_rejects_empty_prompt(self, client_with_model):
        response = client_with_model.post('/generate', json={'prompt': '', 'max_new_tokens': 5})
        assert response.status_code == 422

    def test_generate_rejects_invalid_top_p(self, client_with_model):
        response = client_with_model.post('/generate', json={'prompt': 'Hello', 'top_p': 1.5})
        assert response.status_code == 422

    def test_generate_no_model_returns_503(self, client_with_model):
        state['model'] = None
        response = client_with_model.post('/generate', json={'prompt': 'Hello'})
        assert response.status_code == 503
