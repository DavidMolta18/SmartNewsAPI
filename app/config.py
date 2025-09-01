from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # ---------------------------
    # Vector DB
    # ---------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "articles"
    qdrant_api_key: str | None = None   

    # ---------------------------
    # Embeddings
    # ---------------------------
    # local: intfloat/multilingual-e5-small
    # vertex: text-embedding-005 | text-multilingual-embedding-002 | gemini-embedding-001
    embedding_model: str = "intfloat/multilingual-e5-small"
    provider_embeddings: str = "local"   # "local" (CPU safe) | "vertex" (Vertex AI)

    # ---------------------------
    # Chunking
    # ---------------------------
    # simple: char-based sliding window
    # agentic: Gemini LLM semantic segmentation
    provider_chunking: str = "simple"

    # ---------------------------
    # Vector store provider
    # ---------------------------
    provider_vector: str = "qdrant"      # "qdrant" | "vertex" (future option)

    # ---------------------------
    # Vertex AI config
    # ---------------------------
    gcp_project: str | None = None
    gcp_location: str = "us-central1"

    # ---------------------------
    # Pydantic config
    # ---------------------------
    model_config = SettingsConfigDict(
        env_file=".env",        # load from .env automatically
        extra="ignore",         # ignore unexpected vars in .env
        case_sensitive=False    # accept PROVIDER_X or provider_x
    )

# Create singleton settings object
settings = Settings()
