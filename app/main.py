from fastapi import FastAPI, Body, HTTPException, Query
from app.config import settings
from app.services.embeddings import (
    LocalE5Provider, EmbeddingProvider, as_passages, as_queries
)
from app.services.vector_store import QdrantIndex, VectorIndex
from app.ingestion.rss import fetch_feed, fetch_default

app = FastAPI(title="News Intelligence API")

# --- Providers (por ahora, local + qdrant) ---
embedding_provider: EmbeddingProvider = LocalE5Provider(settings.embedding_model)
vector_index: VectorIndex = QdrantIndex(settings.qdrant_url, settings.qdrant_collection)

# Crear colección en arranque (dim del modelo local e5-small = 384)
try:
    vector_index.ensure_collection(dim=384)
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

    # 1. Obtener artículos (RSS)
    articles = await (fetch_feed(feed_url, max_items) if feed_url else fetch_default(max_items))
    if not articles:
        raise HTTPException(status_code=400, detail="No se obtuvieron artículos")

    # 2. Preparar textos con prefijo E5
    texts = [a["content"] or (a["title"] + " " + a["url"]) for a in articles]
    passages = as_passages(texts)

    # 3. Crear embeddings
    vecs = embedding_provider.embed_texts(passages)

    # 4. Subir a Qdrant
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
def search_endpoint(q: str = Query(..., min_length=2), k: int = 5):
    # 1. Embedding de la consulta con prefijo query
    query_vec = embedding_provider.embed_texts(as_queries([q]))[0]

    # 2. Buscar en Qdrant
    hits = vector_index.search(vector=query_vec, top_k=k)

    # 3. Normalizar salida
    results = [{
        "id": h["id"],
        "score": h["score"],
        "title": h["payload"].get("title"),
        "url": h["payload"].get("url"),
        "published_at": h["payload"].get("published_at"),
        "source": h["payload"].get("source"),
    } for h in hits]
    return {"query": q, "results": results}
