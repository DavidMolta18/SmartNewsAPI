# app/main.py
from __future__ import annotations
from fastapi import FastAPI, Body, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services.embeddings import (
    EmbeddingProvider, LocalE5Provider, VertexEmbeddingProvider,
    as_passages, as_queries
)
from app.services.vector_store import VectorIndex, QdrantIndex
from app.ingestion.rss import fetch_feed, fetch_default

app = FastAPI(title="News Intelligence API")

# --- CORS básico (ajusta dominios en prod) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# --- Providers seleccionados por variables de entorno ---
if settings.provider_embeddings.lower() == "vertex":
    embedding_provider: EmbeddingProvider = VertexEmbeddingProvider(
        model_name=settings.embedding_model,
        project=settings.gcp_project,
        location=settings.gcp_location,
    )
else:
    embedding_provider = LocalE5Provider(settings.embedding_model)

# Por ahora sólo Qdrant como vector store
vector_index: VectorIndex = QdrantIndex(settings.qdrant_url, settings.qdrant_collection)

# Crear/asegurar colección con la dimensión real del provider
try:
    vector_index.ensure_collection(dim=getattr(embedding_provider, "dim", 384))
except Exception as e:
    print("WARN ensure_collection:", e)

@app.get("/health")
def health():
    return {"ok": True}

# ---------- /index ----------
@app.post("/index")
async def index_endpoint(payload: dict = Body(default={})):
    feed_url = payload.get("feed_url")
    max_items = int(payload.get("max_items") or 5)

    # 1) Obtener artículos
    articles = await (fetch_feed(feed_url, max_items) if feed_url else fetch_default(max_items))
    if not articles:
        raise HTTPException(status_code=400, detail="No se obtuvieron artículos")

    # 2) Preparar textos
    texts = [a["content"] or (a["title"] + " " + a["url"]) for a in articles]
    passages = as_passages(texts, prefix_required=getattr(embedding_provider, "requires_e5_prefix", False))

    # 3) Embeddings
    vecs = embedding_provider.embed_texts(passages)

    # 4) Upsert en índice vectorial
    ids = [a["id"] for a in articles]
    payloads = [{
        "title": a["title"],
        "url": a["url"],
        "published_at": a["published_at"],
        "source": a["source"],
    } for a in articles]
    vector_index.upsert(ids=ids, vectors=vecs, payloads=payloads)

    return {"indexed": len(articles)}

# ---------- /search ----------
@app.get("/search")
def search_endpoint(q: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=50)):
    queries = as_queries([q], prefix_required=getattr(embedding_provider, "requires_e5_prefix", False))
    query_vec = embedding_provider.embed_texts(queries)[0]

    hits = vector_index.search(vector=query_vec, top_k=k)
    results = [{
        "id": h["id"],
        "score": h["score"],
        "title": h["payload"].get("title"),
        "url": h["payload"].get("url"),
        "published_at": h["payload"].get("published_at"),
        "source": h["payload"].get("source"),
    } for h in hits]

    return {"query": q, "results": results}
