import os

MODEL_TYPE = os.getenv('MODEL_TYPE', 'ollama')

if MODEL_TYPE == 'ollama':
    from .ollama_model import get_model, get_embedder
else:
    raise ValueError(f"Unsupported model type: {MODEL_TYPE}")
