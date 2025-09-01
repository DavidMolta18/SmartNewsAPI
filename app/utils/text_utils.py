import re
import os
from functools import lru_cache
from fastapi import Depends

from app.ingestion.quality import clean_text
from app.utils.patterns import SNIPPET_NOISE  # centralized snippet noise patterns
from app.config import settings
from app.services.embeddings import (
    EmbeddingProvider, LocalE5Provider, VertexEmbeddingProvider
)
from app.services.vector_store import VectorIndex, QdrantIndex

# =============================================================================
# Helpers for extracting clean snippets for UI/preview display.
# =============================================================================

def first_clean_sentence(text: str) -> str:
    """
    Extract the first clean sentence-like span to use as a preview/snippet.

    Rules:
      - Skip common boilerplate (ads, cookie banners, login prompts).
      - Must be at least 40 characters (avoid uselessly short spans).
      - Cap at 300 characters (avoid flooding the UI).
      - Fall back to the cleaned text (trimmed) if no suitable snippet is found.

    Args:
        text (str): Raw or cleaned text to extract snippet from.

    Returns:
        str: First suitable sentence for preview display.
    """
    if not text:
        return ""

    soft = clean_text(text)
    parts = re.split(r'(?<=[\.\?!])\s+', soft)

    for s in parts:
        s = s.strip()
        if len(s) >= 40 and not SNIPPET_NOISE.search(s):
            return s[:300]

    return soft.strip()[:300]


# =============================================================================
# Providers Factory (singleton via lru_cache)
# =============================================================================

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
    vi = QdrantIndex(
        settings.qdrant_url,
        settings.qdrant_collection,
        api_key=os.getenv("QDRANT_API_KEY"),  
    )
    try:
        vi.ensure_collection(dim=getattr(provider, "dim", 384))
    except Exception as e:
        print("WARN ensure_collection:", e)
    return vi
