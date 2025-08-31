from functools import lru_cache
from fastapi import Depends

from app.config import settings
from app.services.embeddings import (
    EmbeddingProvider, LocalE5Provider, VertexEmbeddingProvider
)
from app.services.vector_store import VectorIndex, QdrantIndex

# ----------------------------------------------------------------------
# Providers Factory (singleton via lru_cache)
# ----------------------------------------------------------------------

@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Return a singleton embedding provider based on settings."""
    if settings.provider_embeddings.lower() == "vertex":
        return VertexEmbeddingProvider(
            model_name=settings.embedding_model,
            project=settings.gcp_project,
            location=settings.gcp_location,
        )
    else:
        return LocalE5Provider(settings.embedding_model)


@lru_cache
def get_vector_index(
    provider: EmbeddingProvider = Depends(get_embedding_provider)
) -> VectorIndex:
    """Return a singleton vector index, ensuring collection exists."""
    vi = QdrantIndex(settings.qdrant_url, settings.qdrant_collection)
    try:
        vi.ensure_collection(dim=getattr(provider, "dim", 384))
    except Exception as e:
        print("WARN ensure_collection:", e)
    return vi
