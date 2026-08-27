from __future__ import annotations
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.generation import generate
from src.tokenizer import Tokenizer
from src.utils import get_device, load_model_from_checkpoint
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
CHECKPOINT_PATH = os.environ.get('MINIMISTRAL_CHECKPOINT', 'checkpoints/mistral_mini.pt')
state: dict = {'model': None, 'tokenizer': None, 'device': None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_device()
    logger.info('Loading MiniMistral checkpoint from %s on %s ...', CHECKPOINT_PATH, device)
    if not os.path.exists(CHECKPOINT_PATH):
        logger.warning('Checkpoint not found at %s. /generate will return an error until a checkpoint exists (run `python scripts/train.py` first).', CHECKPOINT_PATH)
        state['model'] = None
    else:
        state['model'] = load_model_from_checkpoint(CHECKPOINT_PATH, device)
        logger.info('Model loaded: %s parameters.', f'{state['model'].count_parameters():,}')
    state['tokenizer'] = Tokenizer()
    state['device'] = device
    yield
    state['model'] = None
app = FastAPI(title='MiniMistral Inference API', description='Serves text generation from a Mistral-inspired small decoder-only language model implemented from scratch. Not affiliated with, and much smaller than, production Mistral models.', version='1.0.0', lifespan=lifespan)

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description='Text prompt to continue.')
    max_new_tokens: int = Field(100, ge=1, le=1024, description='Number of tokens to generate.')
    temperature: float = Field(0.8, gt=0.0, le=5.0, description='Sampling temperature.')
    top_p: float = Field(0.9, gt=0.0, le=1.0, description='Nucleus sampling threshold.')
    greedy: bool = Field(False, description='If true, use deterministic greedy decoding.')
    seed: Optional[int] = Field(None, description='Optional seed for deterministic sampling.')

class GenerateResponse(BaseModel):
    generated_text: str

class HealthResponse(BaseModel):
    status: str

@app.get('/health', response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status='healthy')

@app.post('/generate', response_model=GenerateResponse)
def generate_text(request: GenerateRequest) -> GenerateResponse:
    model = state['model']
    tokenizer = state['tokenizer']
    device = state['device']
    if model is None:
        raise HTTPException(status_code=503, detail=f"No model checkpoint loaded (expected at '{CHECKPOINT_PATH}'). Train a model first: python scripts/train.py --config configs/mini_mistral.yaml")
    try:
        text = generate(model, tokenizer, prompt=request.prompt, device=device, max_new_tokens=request.max_new_tokens, temperature=request.temperature, top_p=request.top_p, greedy=request.greedy, seed=request.seed)
    except Exception as exc:
        logger.exception('Generation failed')
        raise HTTPException(status_code=500, detail=f'Generation failed: {exc}') from exc
    return GenerateResponse(generated_text=text)
