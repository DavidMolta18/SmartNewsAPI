# app/services/vector_store.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional

class VectorIndex(ABC):
    @abstractmethod
    def ensure_collection(self, dim: int) -> None: ...
    @abstractmethod
    def upsert(self, ids: list[str], vectors, payloads: list[dict[str, Any]]) -> None: ...
    @abstractmethod
    def search(self, vector, top_k: int,
               qfilter: Any | None = None,
               hnsw_ef: Optional[int] = None) -> list[dict[str, Any]]: ...
    # Opcional: agrupación por payload (útil con chunking)
    def search_grouped(self, vector, top_k: int, group_by: str,
                       group_size: int = 1,
                       qfilter: Any | None = None,
                       hnsw_ef: Optional[int] = None) -> list[list[dict[str, Any]]]:
        raise NotImplementedError

class QdrantIndex(VectorIndex):
    def __init__(self, url: str, collection: str):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(url=url)
        self.collection = collection

    def _create_payload_indexes(self):
        """Crea índices de payload (idempotente)."""
        from qdrant_client.http import models as qm
        for field, ftype in [
            ("article_id", qm.PayloadSchemaType.KEYWORD),
            ("source", qm.PayloadSchemaType.KEYWORD),
            ("published_at_ts", qm.PayloadSchemaType.INTEGER),
        ]:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=ftype,
                )
            except Exception:
                # Ya existe o no soportado por la versión → ignorar
                pass

    def ensure_collection(self, dim: int) -> None:
        from qdrant_client.http import models as qm
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )
        # asegurar índices de payload (lo llamamos siempre; es idempotente)
        self._create_payload_indexes()

    def upsert(self, ids, vectors, payloads):
        from qdrant_client.http import models as qm
        # Qdrant acepta listas de float; asegurar tipo y forma
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        self.client.upsert(
            collection_name=self.collection,
            points=qm.Batch(ids=ids, vectors=vectors, payloads=payloads),
        )

    def search(self, vector, top_k: int,
               qfilter: Any | None = None,
               hnsw_ef: Optional[int] = None):
        from qdrant_client.http import models as qm
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        search_params = None
        if hnsw_ef is not None:
            search_params = qm.SearchParams(hnsw_ef=int(hnsw_ef), exact=False)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            query_filter=qfilter,
            search_params=search_params,
        )
        return [
            {"id": str(h.id), "score": float(h.score), "payload": h.payload}
            for h in hits
        ]

    # Agrupación (útil cuando indexas varios chunks por artículo)
    def search_grouped(self, vector, top_k: int, group_by: str,
                       group_size: int = 1,
                       qfilter: Any | None = None,
                       hnsw_ef: Optional[int] = None) -> list[list[dict[str, Any]]]:
        from qdrant_client.http import models as qm
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        search_params = None
        if hnsw_ef is not None:
            search_params = qm.SearchParams(hnsw_ef=int(hnsw_ef), exact=False)

        res = self.client.search_groups(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,             # número de grupos (p. ej. artículos)
            group_by=group_by,       # p. ej. "article_id"
            group_size=group_size,   # top hits por grupo (p. ej. mejores chunks)
            query_filter=qfilter,
            search_params=search_params,
        )
        groups = []
        for g in res.groups:
            groups.append([
                {"id": str(h.id), "score": float(h.score), "payload": h.payload}
                for h in g.hits
            ])
        return groups
