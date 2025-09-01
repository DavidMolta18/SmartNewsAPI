# app/services/vector_store.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional


class VectorIndex(ABC):
    """Abstract interface so we can swap vector backends without touching the app."""

    @abstractmethod
    def ensure_collection(self, dim: int) -> None:
        """Create the collection if missing and make sure vector params & payload indexes exist."""
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
        with_vectors: bool = False,   # <-- NUEVO (opcional)
    ) -> list[dict[str, Any]]:
        """
        ANN search by vector. Optional:
          - qfilter: Qdrant filter object (pass-through)
          - hnsw_ef: runtime ef value for better recall on HNSW
          - with_vectors: return stored point vectors in results (if backend supports it)
        Returns a flat list of hits with id/score/payload and (optionally) vector.
        """
        ...

    # Optional convenience: server-side grouping (useful when chunking)
    def search_grouped(
        self,
        vector,
        top_k: int,
        group_by: str,
        group_size: int = 1,
        qfilter: Any | None = None,
        hnsw_ef: Optional[int] = None,
    ) -> list[list[dict[str, Any]]]:
        """
        Group hits by a payload field (e.g., "article_id") and return up to `top_k` groups,
        each with up to `group_size` hits. If unsupported by the backend, override/raise.
        """
        raise NotImplementedError


class QdrantIndex(VectorIndex):
    """
    Qdrant-backed vector index.
    - COSINE distance with L2-normalized embeddings (as produced by your providers)
    - Payload indexes on fields we filter/sort by
    - Optional runtime HNSW ef to trade recall/latency
    """

    def __init__(self, url: str, collection: str):
        from qdrant_client import QdrantClient

        # HTTP client is fine for local/dev; you can switch to gRPC with prefer_grpc=True
        self.client = QdrantClient(url=url)
        self.collection = collection

    # ---------- internal helpers ----------

    def _create_payload_indexes(self) -> None:
        """
        Create payload indexes (idempotent). We do it every boot since Qdrant ignores duplicates.
        Indexing these fields speeds up filters & grouping.
        """
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
                # Already exists or version without this feature — safe to ignore
                pass

    # ---------- VectorIndex API ----------

    def ensure_collection(self, dim: int) -> None:
        """Ensure collection exists with the right vector params + payload indexes."""
        from qdrant_client.http import models as qm

        try:
            existing = [c.name for c in self.client.get_collections().collections]
        except Exception:
            existing = []

        if self.collection not in existing:
            # Default HNSW config with cosine distance
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )

        # Make sure payload indexes are in place (idempotent)
        self._create_payload_indexes()

    def upsert(self, ids, vectors, payloads):
        """Upsert points in batch; accepts numpy arrays for vectors."""
        from qdrant_client.http import models as qm

        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()

        self.client.upsert(
            collection_name=self.collection,
            points=qm.Batch(ids=ids, vectors=vectors, payloads=payloads),
        )

    def search(
        self,
        vector,
        top_k: int,
        qfilter: Any | None = None,
        hnsw_ef: Optional[int] = None,
        with_vectors: bool = False,   # <-- NUEVO (opcional)
    ):
        """Standard ANN search with optional runtime ef override and optional vector return."""
        from qdrant_client.http import models as qm

        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        search_params = None
        if hnsw_ef is not None:
            # exact=False keeps ANN; set exact=True if you need brute-force rerank (slower)
            search_params = qm.SearchParams(hnsw_ef=int(hnsw_ef), exact=False)

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            query_filter=qfilter,
            search_params=search_params,
            with_vectors=with_vectors,   # <-- se lo pasamos al cliente
        )

        return [
            {
                "id": str(h.id),
                "score": float(h.score),
                "payload": h.payload,
                "vector": (h.vector if with_vectors else None),  # <-- devolvemos vector si se pidió
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
        """
        Use Qdrant's server-side grouping. This is great when you index multiple chunks
        per article and want the best chunk(s) for each top article in one call.
        """
        from qdrant_client.http import models as qm

        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        search_params = None
        if hnsw_ef is not None:
            search_params = qm.SearchParams(hnsw_ef=int(hnsw_ef), exact=False)

        res = self.client.search_groups(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,            # number of groups (e.g., articles)
            group_by=group_by,      # e.g., "article_id"
            group_size=group_size,  # how many hits per group (e.g., top 3 chunks)
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
