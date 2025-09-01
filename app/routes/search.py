from collections import defaultdict
from typing import List
from fastapi import APIRouter, Query, Depends

from app.services.embeddings import as_queries, EmbeddingProvider
from app.services.vector_store import VectorIndex
from app.utils.text_utils import SNIPPET_NOISE, first_clean_sentence
from app.dependencies import get_embedding_provider, get_vector_index
from app.schemas import SearchResponse, ArticleHit

router = APIRouter()


@router.get("/", response_model=SearchResponse)
def search_endpoint(
    q: str = Query(..., min_length=2, description="User text query (semantic)"),
    k: int = Query(5, ge=1, le=50, description="Max number of articles to return"),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_index: VectorIndex = Depends(get_vector_index),
) -> SearchResponse:
    """
    Semantic search.
    Pipeline:
      1) Embed the query string.
      2) ANN search in Qdrant with aggressive fanout.
      3) Group hits by article_id and rerank by best chunk score.
      4) Build article-level results with up to 3 clean snippets.
    """
    # 1) Embed query
    queries = as_queries(
        [q],
        prefix_required=getattr(embedding_provider, "requires_e5_prefix", False),
    )
    query_vec = embedding_provider.embed_texts(queries)[0]

    # 2) ANN search (fanout improves recall before article-level grouping)
    fanout = max(20, k * 8)
    try:
        hits = vector_index.search(vector=query_vec, top_k=fanout, hnsw_ef=256)
    except TypeError:
        # Older client versions without runtime ef override
        hits = vector_index.search(vector=query_vec, top_k=fanout)

    # 3) Group by article_id
    groups = defaultdict(list)
    for h in hits:
        aid = (h.get("payload") or {}).get("article_id")
        if aid:
            groups[aid].append(h)

    def group_score(hlist) -> float:
        return max(h["score"] for h in hlist) if hlist else 0.0

    ranked = sorted(groups.items(), key=lambda kv: group_score(kv[1]), reverse=True)[:k]

    # 4) Build response with clean snippets
    results: List[ArticleHit] = []
    for aid, hlist in ranked:
        hlist_sorted = sorted(hlist, key=lambda h: -h["score"])
        best = hlist_sorted[0]
        bestp = best.get("payload") or {}

        snippets, seen = [], set()
        # Look at top chunks for this article to extract diverse snippets
        for h in hlist_sorted[:8]:
            payload = h.get("payload", {}) or {}
            snip = payload.get("chunk_preview") or ""

            # If preview is noisy or empty, fall back to first clean sentence of head
            if not snip or SNIPPET_NOISE.search(snip):
                head = payload.get("chunk_text_head") or ""
                snip = first_clean_sentence(head) if head else ""

            # Final fallback: trimmed head
            if not snip and payload.get("chunk_text_head"):
                snip = (payload["chunk_text_head"][:280] + "…")

            snip = (snip or "").strip()
            if not snip or SNIPPET_NOISE.search(snip):
                continue
            key = snip.lower()
            if key in seen:
                continue
            seen.add(key)
            snippets.append(snip)
            if len(snippets) >= 3:
                break

        results.append(
            ArticleHit(
                article_id=aid,
                title=bestp.get("title") or "",
                url=bestp.get("url") or "",
                source=bestp.get("source") or "",
                published_at=bestp.get("published_at"),
                score=float(best.get("score", 0.0)),
                snippets=snippets,
            )
        )

    return SearchResponse(query=q, results=results)
