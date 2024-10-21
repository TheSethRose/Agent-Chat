import os
from phi.model.ollama import Ollama
from phi.embedder.ollama import OllamaEmbedder

# Ollama configuration
OLLAMA_HOST: str = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

# Llama model ID
LLAMA_MODEL_ID: str = os.getenv('LLAMA_MODEL_ID', 'llama3.2')

# Embedder model
EMBEDDER_MODEL: str = os.getenv('EMBEDDER_MODEL', 'nomic-embed-text')

def get_model():
    """
    Creates and returns an instance of the Ollama model.
    """
    return Ollama(id=LLAMA_MODEL_ID, host=OLLAMA_HOST)

def get_embedder():
    """
    Creates and returns an instance of the OllamaEmbedder.
    """
    return OllamaEmbedder(
        model=EMBEDDER_MODEL,
        host=OLLAMA_HOST,
    )
