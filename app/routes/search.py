# app/routes/search.py
from collections import defaultdict
from fastapi import APIRouter, Query, Depends

from app.services.embeddings import as_queries, EmbeddingProvider
from app.services.vector_store import VectorIndex
from app.utils.text_utils import SNIPPET_NOISE, first_clean_sentence
from app.dependencies import get_embedding_provider, get_vector_index

router = APIRouter()

@router.get("/")
def search_endpoint(
    q: str = Query(..., min_length=2),
    k: int = Query(5, ge=1, le=50),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_index: VectorIndex = Depends(get_vector_index),
):
    """
    Semantic search endpoint.
    Pipeline:
    1. Embed query string into vector.
    2. ANN search in Qdrant with fanout.
    3. Group hits by article and rerank.
    4. Return article-level results with up to 3 clean snippets.
    """
    # Step 1: embed query
    queries = as_queries(
        [q],
        prefix_required=getattr(embedding_provider, "requires_e5_prefix", False),
    )
    query_vec = embedding_provider.embed_texts(queries)[0]

    # Step 2: ANN search with fanout
    FANOUT = max(20, k * 8)
    try:
        hits = vector_index.search(vector=query_vec, top_k=FANOUT, hnsw_ef=256)
    except TypeError:
        hits = vector_index.search(vector=query_vec, top_k=FANOUT)

    # Step 3: group by article
    groups = defaultdict(list)
    for h in hits:
        aid = h.get("payload", {}).get("article_id")
        if aid:
            groups[aid].append(h)

    def group_score(hlist): return max(h["score"] for h in hlist) if hlist else 0.0
    ranked = sorted(groups.items(), key=lambda kv: group_score(kv[1]), reverse=True)[:k]

    # Step 4: build response with clean snippets
    results = []
    for aid, hlist in ranked:
        hlist_sorted = sorted(hlist, key=lambda h: -h["score"])
        best = hlist_sorted[0]

        snippets, seen = [], set()
        for h in hlist_sorted[:8]:  # look at top chunks per article
            payload = h.get("payload", {}) or {}
            snip = payload.get("chunk_preview") or ""

            if not snip or SNIPPET_NOISE.search(snip):
                head = payload.get("chunk_text_head") or ""
                snip = first_clean_sentence(head) if head else ""

            if not snip and payload.get("chunk_text_head"):
                snip = (payload["chunk_text_head"][:280] + "…")

            snip = (snip or "").strip()
            if not snip or SNIPPET_NOISE.search(snip):
                continue
            if snip.lower() in seen:
                continue

            seen.add(snip.lower())
            snippets.append(snip)
            if len(snippets) >= 3:
                break

        results.append({
            "article_id": aid,
            "title": best["payload"].get("title"),
            "url": best["payload"].get("url"),
            "source": best["payload"].get("source"),
            "published_at": best["payload"].get("published_at"),
            "score": float(best["score"]),
            "snippets": snippets,
        })

    return {"query": q, "results": results}
