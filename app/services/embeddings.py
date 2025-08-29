# app/services/embeddings.py
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

def as_passages(texts: list[str], prefix_required: bool = True) -> list[str]:
    """Prefijo 'passage:' sólo si el provider lo requiere (E5)."""
    return [f"passage: {t}" if prefix_required else t for t in texts]

def as_queries(texts: list[str], prefix_required: bool = True) -> list[str]:
    """Prefijo 'query:' sólo si el provider lo requiere (E5)."""
    return [f"query: {t}" if prefix_required else t for t in texts]

class EmbeddingProvider(ABC):
    """Interfaz estable para cambiar de proveedor de embeddings sin tocar el resto."""
    dim: int
    requires_e5_prefix: bool = False  # E5 = True, Vertex = False

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        ...

class LocalE5Provider(EmbeddingProvider):
    """Embeddings locales con SentenceTransformers (E5)."""
    requires_e5_prefix = True

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True)  # float32
        return np.asarray(vecs, dtype=np.float32)

class VertexEmbeddingProvider(EmbeddingProvider):
    """
    Embeddings en Vertex AI (p. ej. 'text-embedding-004' o 'gemini-embedding-001').
    Requiere credenciales de GCP (en Cloud Run usar Service Account).
    """
    requires_e5_prefix = False

    def __init__(self, model_name: str = "text-embedding-004",
                 project: str | None = None, location: str = "us-central1"):
        from google.cloud import aiplatform
        aiplatform.init(project=project, location=location)
        # Carga perezosa del endpoint/modelo para embeddings
        self._model = aiplatform.TextEmbeddingModel.from_pretrained(model_name)
        # Detectar dimensión de salida sin “adivinar”
        sample = self._model.get_embeddings(["ping"])[0].values
        self.dim = len(sample)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # Vertex usualmente devuelve listas de floats sin normalizar
        embs = self._model.get_embeddings(texts)
        arr = np.asarray([e.values for e in embs], dtype=np.float32)
        # Normalizar L2 para usar distancia COSINE de forma consistente
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).astype(np.float32)
