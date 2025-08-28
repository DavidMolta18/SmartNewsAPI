from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "articles"
    embedding_model: str = "intfloat/multilingual-e5-small"
    provider_embeddings: str = "local"   # local | vertex
    provider_vector: str = "qdrant"      # qdrant | vertex
    class Config:
        env_file = ".env"

settings = Settings()
