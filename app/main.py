# app/main.py
from __future__ import annotations
from fastapi import FastAPI, Body, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uuid

from app.config import settings
from app.services.embeddings import (
    EmbeddingProvider, LocalE5Provider, VertexEmbeddingProvider,
    as_passages, as_queries
)
from app.services.vector_store import VectorIndex, QdrantIndex
from app.ingestion.rss import fetch_feed, fetch_default
from app.ingestion.quality import clean_text, should_index



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

    # 2) Limpiar + filtrar textos (doble seguro)
    cleaned_articles, cleaned_texts = [], []
    for a in articles:
        txt = a.get("content") or (f"{a.get('title','')} {a.get('url','')}")
        txt = clean_text(txt)

        ok, reason = should_index(txt, min_chars=400, min_score=700.0)
        if not ok:
            # Si quieres ver qué se descartó:
            # print(f"SKIP index [{a.get('source')}] {a.get('title','')[:80]} -> {reason}")
            continue

        cleaned_articles.append(a)
        cleaned_texts.append(txt)

    if not cleaned_articles:
        raise HTTPException(status_code=400, detail="Todos los artículos fueron descartados por baja calidad")

    # 3) Debug: ver lo que realmente se embebe
    print("\n=== ARTÍCULOS A EMBED (post-filtro) ===")
    for art, txt in zip(cleaned_articles, cleaned_texts):
        preview = (txt[:400] + "...") if len(txt) > 400 else txt
        print(f"[{art['source']}] {art['title'][:80]}")
        print(f"LEN={len(txt)} | Published: {art['published_at']}")
        print(preview, "\n")

    # 4) IDs y payloads
    ids = [str(uuid.uuid4()) for _ in cleaned_articles]

    def to_epoch(ts_iso: str | None) -> int | None:
        if not ts_iso:
            return None
        try:
            # ISO con Z / +00:00
            return int(dt.datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp())
        except Exception:
            return None

    payloads = [{
        "article_id": a["id"],
        "title": a["title"],
        "url": a["url"],
        "published_at": a["published_at"],
        "published_at_ts": to_epoch(a.get("published_at")),  # ⬅️ útil para ordenar/filtrar
        "source": a["source"],
    } for a in cleaned_articles]

    # 5) Embeddings con prefijo si aplica (E5)
    passages = as_passages(
        cleaned_texts,
        prefix_required=getattr(embedding_provider, "requires_e5_prefix", False)
    )

    # 6) Generar embeddings
    vecs = embedding_provider.embed_texts(passages)

    # 7) Upsert en índice vectorial
    vector_index.upsert(ids=ids, vectors=vecs, payloads=payloads)

    return {"indexed_articles": len(cleaned_articles)}



# ---------- /search ----------
@app.get("/search")
def search_endpoint(q: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=50)):
    queries = as_queries(
        [q],
        prefix_required=getattr(embedding_provider, "requires_e5_prefix", False)
    )
    query_vec = embedding_provider.embed_texts(queries)[0]

    hits = vector_index.search(vector=query_vec, top_k=k)
    results = [{
        "id": h["id"],
        "score": h["score"],
        "title": h["payload"].get("title"),
        "url": h["payload"].get("url"),
        "published_at": h["payload"].get("published_at"),
        "source": h["payload"].get("source"),
        "article_id": h["payload"].get("article_id"),
    } for h in hits]

    return {"query": q, "results": results}
