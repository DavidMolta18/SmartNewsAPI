import uuid
import asyncio
import random
from typing import List, Dict
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
from app.schemas import IndexParams, IndexResponse

router = APIRouter()


@router.post("/", response_model=IndexResponse)
async def index_endpoint(
    params: IndexParams = Body(default=IndexParams()),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_index: VectorIndex = Depends(get_vector_index),
) -> IndexResponse:
    """
    Fetch, clean, chunk, embed, and index articles into the vector store.

    Steps:
      1) Fetch RSS items (single feed or default feeds.json).
      2) Clean and quality-filter textual content.
      3) Chunk either with agentic LLM segmentation or a safe char-based splitter.
      4) Embed chunks in batches and upsert into Qdrant.
    """
    provider_chunking = settings.provider_chunking.lower()

    # 1) Fetch
    if params.feed_url:
        articles = await fetch_feed(params.feed_url, want=params.per_feed_target, max_scan=params.max_scan)
    else:
        articles = await fetch_default(params.per_feed_target, params.max_scan)

    if not articles:
        raise HTTPException(status_code=400, detail="No articles fetched")

    # 2) Clean + filter
    cleaned: List[Dict] = []
    for a in articles:
        raw = a.get("content") or f"{a.get('title','')} {a.get('url','')}"
        txt = clean_text(raw)
        ok, _reason = should_index(txt, min_chars=400, min_score=700.0)
        if ok:
            a["content"] = txt
            cleaned.append(a)

    if not cleaned:
        raise HTTPException(status_code=400, detail="All articles failed quality filters")

    # 3) Batch processing (keeps memory and external calls under control)
    BATCH_SIZE = 3
    total_chunks = 0
    total_articles = 0

    for i in range(0, len(cleaned), BATCH_SIZE):
        batch = cleaned[i : i + BATCH_SIZE]
        chunk_texts: List[str] = []
        chunk_payloads: List[Dict] = []
        chunk_ids: List[str] = []

        for a in batch:
            content = a["content"]
            use_agentic = (provider_chunking == "agentic") and (len(content) >= 1500)

            if use_agentic:
                try:
                    chunks = agentic_chunk(content, max_chunks=params.max_chunks)
                    # Gentle pacing to avoid 429s on provider limits
                    await asyncio.sleep(random.uniform(0.15, 0.30))
                except Exception as e:
                    print(f"[index] Agentic chunking failed for '{a.get('title','')}' -> fallback: {e}")
                    chunks = split_text_by_chars(content, max_chars=2000, overlap=200)
            else:
                chunks = split_text_by_chars(content, max_chars=2000, overlap=200)

            chunks = chunks[: params.max_chunks]

            for idx, ch in enumerate(chunks):
                chunk_texts.append(ch)
                chunk_ids.append(str(uuid.uuid4()))
                chunk_payloads.append(
                    {
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
                    }
                )

        if not chunk_texts:
            continue

        # 4) Embed and upsert
        passages = as_passages(
            chunk_texts,
            prefix_required=getattr(embedding_provider, "requires_e5_prefix", False),
        )
        vecs = embed_in_batches(passages, embedding_provider, batch_size=8)
        vector_index.upsert(ids=chunk_ids, vectors=vecs, payloads=chunk_payloads)

        print(f"[index] Batch {i // BATCH_SIZE + 1}: {len(chunk_ids)} chunks from {len(batch)} articles")
        total_chunks += len(chunk_ids)
        total_articles += len(batch)

    return IndexResponse(
        indexed_articles=total_articles,
        indexed_chunks=total_chunks,
        chunk_mode=settings.provider_chunking.lower(),
    )
