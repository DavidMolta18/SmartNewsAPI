from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional


class VectorIndex(ABC):
    """Abstract interface so we can swap vector backends without touching the app."""

    @abstractmethod
    def ensure_collection(self, dim: int) -> None:
        """Check that the collection exists, otherwise warn."""
        ...

    @abstractmethod
    def upsert(self, ids: list[str], vectors, payloads: list[dict[str, Any]]) -> None:
        """Insert or update vectors + payloads in batch."""
        ...

    @abstractmethod
    def search(
        self,
        vector,
        top_k: int,
        qfilter: Any | None = None,
        hnsw_ef: Optional[int] = None,
        with_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        """
        ANN search by vector. Optional:
          - qfilter: Qdrant filter object (pass-through)
          - hnsw_ef: runtime ef value for better recall on HNSW
          - with_vectors: return stored point vectors in results (if backend supports it)
        """
        ...

    def search_grouped(
        self,
        vector,
        top_k: int,
        group_by: str,
        group_size: int = 1,
        qfilter: Any | None = None,
        hnsw_ef: Optional[int] = None,
    ) -> list[list[dict[str, Any]]]:
        raise NotImplementedError


class QdrantIndex(VectorIndex):
    """
    Qdrant-backed vector index.
    Supports Qdrant Cloud (API key auth) or local Qdrant (no API key).
    """

    def __init__(self, url: str, collection: str, api_key: str | None = None):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False, timeout=30.0)
        self.collection = collection

    def _create_payload_indexes(self) -> None:
        """Ensure useful payload indexes exist (safe to skip on Cloud free)."""
        from qdrant_client.http import models as qm

        to_index: list[tuple[str, qm.PayloadSchemaType]] = [
            ("article_id", qm.PayloadSchemaType.KEYWORD),
            ("source", qm.PayloadSchemaType.KEYWORD),
            ("published_at_ts", qm.PayloadSchemaType.INTEGER),
        ]

        for field, ftype in to_index:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=ftype,
                )
            except Exception:
                pass

    def ensure_collection(self, dim: int) -> None:
        """Check if the collection exists in Qdrant Cloud, but don't try to create it."""
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            if self.collection not in existing:
                raise RuntimeError(
                    f"[ERROR] Collection '{self.collection}' not found in Qdrant Cloud. "
                    f"Please create it manually (dim={dim}, metric=cosine, vector_name='vector')."
                )
            else:
                print(f"[INFO] Collection '{self.collection}' is available.")
        except Exception as e:
            print("[WARN ensure_collection skipped]:", e)

    def upsert(self, ids, vectors, payloads):
        from qdrant_client.http import models as qm

        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()

        self.client.upsert(
            collection_name=self.collection,
            points=qm.Batch(ids=ids, vectors={"vector": vectors}, payloads=payloads),
        )

    def search(
        self,
        vector,
        top_k: int,
        qfilter: Any | None = None,
        hnsw_ef: Optional[int] = None,
        with_vectors: bool = False,
    ):
        from qdrant_client.http import models as qm

        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        search_params = None
        if hnsw_ef is not None:
            search_params = qm.SearchParams(hnsw_ef=int(hnsw_ef), exact=False)

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=qm.NamedVector(name="vector", vector=vector),
            limit=top_k,
            query_filter=qfilter,
            search_params=search_params,
            with_vectors=with_vectors,
        )

        return [
            {
                "id": str(h.id),
                "score": float(h.score),
                "payload": h.payload,
                "vector": (h.vector if with_vectors else None),
            }
            for h in hits
        ]

    def search_grouped(
        self,
        vector,
        top_k: int,
        group_by: str,
        group_size: int = 1,
        qfilter: Any | None = None,
        hnsw_ef: Optional[int] = None,
    ) -> list[list[dict[str, Any]]]:
        from qdrant_client.http import models as qm

        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        search_params = None
        if hnsw_ef is not None:
            search_params = qm.SearchParams(hnsw_ef=int(hnsw_ef), exact=False)

        res = self.client.search_groups(
            collection_name=self.collection,
            query_vector=qm.NamedVector(name="vector", vector=vector),
            limit=top_k,
            group_by=group_by,
            group_size=group_size,
            query_filter=qfilter,
            search_params=search_params,
        )

        groups: list[list[dict[str, Any]]] = []
        for g in res.groups:
            groups.append(
                [
                    {"id": str(h.id), "score": float(h.score), "payload": h.payload}
                    for h in g.hits
                ]
            )
        return groups
