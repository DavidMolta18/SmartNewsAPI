# app/routes/indexing.py
import uuid, asyncio, random
from fastapi import APIRouter, Body, HTTPException, Depends

from app.config import settings
from app.services.chunking import agentic_chunk, split_text_by_chars
from app.ingestion.rss import fetch_feed, fetch_default
from app.ingestion.quality import clean_text, should_index
from app.utils.text_utils import first_clean_sentence
from app.utils.misc import to_epoch, embed_in_batches
from app.services.embeddings import as_passages, EmbeddingProvider
from app.services.vector_store import VectorIndex
from app.dependencies import get_embedding_provider, get_vector_index

router = APIRouter()

@router.post("/")
async def index_endpoint(
    payload: dict = Body(default={}),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_index: VectorIndex = Depends(get_vector_index),
):
    """
    Fetch, clean, chunk, embed, and index articles into Qdrant.

    Steps:
    1. Fetch RSS articles (from given feed or default feeds.json).
    2. Clean text and filter low-quality content.
    3. Split into chunks (agentic with Gemini or simple char-based).
    4. Embed chunks and upsert into the vector store.
    """
    provider_chunking = settings.provider_chunking.lower()
    feed_url = payload.get("feed_url")
    per_feed_target = int(payload.get("per_feed_target") or 2)
    max_scan = int(payload.get("max_scan") or 20)
    max_chunks_per_article = int(payload.get("max_chunks") or 3)

    # Step 1: fetch articles
    articles = await (
        fetch_feed(feed_url, want=per_feed_target, max_scan=max_scan)
        if feed_url else fetch_default(per_feed_target, max_scan)
    )
    if not articles:
        raise HTTPException(status_code=400, detail="No articles fetched")

    # Step 2: cleaning + quality filtering
    cleaned = []
    for a in articles:
        txt = clean_text(a.get("content") or (f"{a.get('title','')} {a.get('url','')}"))
        ok, _ = should_index(txt, min_chars=400, min_score=700.0)
        if ok:
            a["content"] = txt
            cleaned.append(a)

    if not cleaned:
        raise HTTPException(status_code=400, detail="All articles failed quality filters")

    # Step 3: process in batches
    BATCH_SIZE = 3
    total_chunks, total_articles = 0, 0

    for i in range(0, len(cleaned), BATCH_SIZE):
        batch = cleaned[i:i + BATCH_SIZE]
        chunk_texts, chunk_payloads, chunk_ids = [], [], []

        for a in batch:
            use_agentic = (provider_chunking == "agentic") and (len(a["content"]) >= 1500)

            if use_agentic:
                try:
                    chunks = agentic_chunk(a["content"], max_chunks=max_chunks_per_article)
                    await asyncio.sleep(random.uniform(0.15, 0.30))  # smooth traffic
                except Exception as e:
                    print(f"Agentic chunking failed for {a['title']}: {e}")
                    chunks = split_text_by_chars(a["content"], max_chars=2000, overlap=200)
            else:
                chunks = split_text_by_chars(a["content"], max_chars=2000, overlap=200)

            chunks = chunks[:max_chunks_per_article]

            # Build payloads for each chunk
            for idx, ch in enumerate(chunks):
                chunk_texts.append(ch)
                chunk_ids.append(str(uuid.uuid4()))
                chunk_payloads.append({
                    "article_id": a["id"],
                    "title": a["title"],
                    "url": a["url"],
                    "published_at": a["published_at"],
                    "published_at_ts": to_epoch(a.get("published_at")),
                    "source": a["source"],
                    "chunk_index": idx,
                    "chunk_preview": first_clean_sentence(ch),
                    "chunk_text_head": ch[:1200],
                    "chunk_mode": "agentic" if use_agentic else "simple",
                })

        if not chunk_texts:
            continue

        # Step 4: embed chunks
        passages = as_passages(
            chunk_texts,
            prefix_required=getattr(embedding_provider, "requires_e5_prefix", False),
        )
        vecs = embed_in_batches(passages, embedding_provider, batch_size=8)

        # Step 5: upsert into vector store
        vector_index.upsert(ids=chunk_ids, vectors=vecs, payloads=chunk_payloads)

        print(f"Indexed batch {i // BATCH_SIZE + 1}: {len(chunk_ids)} chunks from {len(batch)} articles")
        total_chunks += len(chunk_ids)
        total_articles += len(batch)

    return {
        "indexed_articles": total_articles,
        "indexed_chunks": total_chunks,
        "chunk_mode": settings.provider_chunking.lower()
    }
