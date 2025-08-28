from abc import ABC, abstractmethod
from typing import Any

class VectorIndex(ABC):
    @abstractmethod
    def ensure_collection(self, dim: int) -> None: ...
    @abstractmethod
    def upsert(self, ids: list[str], vectors, payloads: list[dict[str, Any]]) -> None: ...
    @abstractmethod
    def search(self, vector, top_k: int) -> list[dict[str, Any]]: ...

class QdrantIndex(VectorIndex):
    def __init__(self, url: str, collection: str):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(url=url)
        self.collection = collection

    def ensure_collection(self, dim: int) -> None:
        from qdrant_client.http import models as qm
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    def upsert(self, ids, vectors, payloads):
        from qdrant_client.http import models as qm
        self.client.upsert(
            collection_name=self.collection,
            points=qm.Batch(ids=ids, vectors=vectors, payloads=payloads),
        )

    def search(self, vector, top_k: int):
        hits = self.client.search(collection_name=self.collection, query_vector=vector, limit=top_k)
        # estandarizamos respuesta
        return [
            {"id": str(h.id), "score": float(h.score), "payload": h.payload}
            for h in hits
        ]
