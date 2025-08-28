from abc import ABC, abstractmethod
import numpy as np

def as_passages(texts: list[str]) -> list[str]:
    """Agrega el prefijo 'passage:' (requerido por E5)."""
    return [f"passage: {t}" for t in texts]

def as_queries(texts: list[str]) -> list[str]:
    """Agrega el prefijo 'query:' (requerido por E5)."""
    return [f"query: {t}" for t in texts]

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        ...

class LocalE5Provider(EmbeddingProvider):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, normalize_embeddings=True))
