# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Vector DB
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "articles"

    # Embeddings
    # local: intfloat/multilingual-e5-small
    # vertex: text-embedding-004 | gemini-embedding-001 
    embedding_model: str = "intfloat/multilingual-e5-small"
    provider_embeddings: str = "local"   # local | vertex

    # (Futuro) elegir otro vector store
    provider_vector: str = "qdrant"      # qdrant | vertex

    # Vertex config
    gcp_project: str | None = None
    gcp_location: str = "us-central1"

    class Config:
        env_file = ".env"

settings = Settings()
