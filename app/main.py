from __future__ import annotations
import os, json, re, uuid, datetime as dt, asyncio, time, random
from typing import List
import numpy as np

from fastapi import FastAPI, Body, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services.embeddings import (
    EmbeddingProvider, LocalE5Provider, VertexEmbeddingProvider,
    as_passages, as_queries
)
from app.services.vector_store import VectorIndex, QdrantIndex
from app.ingestion.rss import fetch_feed, fetch_default
from app.ingestion.quality import clean_text, should_index

# Mutable flag to avoid 'global' assignment issues in some reloaders
_VERTEX_READY = {"ok": False}

app = FastAPI(title="News Intelligence API")

# -----------------------------------------------------------------------------
# CORS (relax for demo; restrict origins in prod)
# -----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# -----------------------------------------------------------------------------
# Embedding provider selection (local E5 or Vertex)
# -----------------------------------------------------------------------------
if settings.provider_embeddings.lower() == "vertex":
    embedding_provider: EmbeddingProvider = VertexEmbeddingProvider(
        model_name=settings.embedding_model,
        project=settings.gcp_project,
        location=settings.gcp_location,
    )
else:
    embedding_provider = LocalE5Provider(settings.embedding_model)

# -----------------------------------------------------------------------------
# Vector store (Qdrant)
# -----------------------------------------------------------------------------
vector_index: VectorIndex = QdrantIndex(settings.qdrant_url, settings.qdrant_collection)

# Ensure the collection exists with the proper vector size
try:
    vector_index.ensure_collection(dim=getattr(embedding_provider, "dim", 384))
except Exception as e:
    print("WARN ensure_collection:", e)


@app.get("/health")
def health():
    return {"ok": True}


# =============================================================================
# ---------------------- Chunking / Embedding Utilities -----------------------
# =============================================================================

def _split_text_by_chars(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """
    Simple sliding window chunker by character length.
    This is the "safe" fallback: cheap, deterministic, no external calls.
    """
    text = text or ""
    max_chars = max(500, max_chars)
    overlap = max(0, min(overlap, max_chars // 2))
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        out.append(text[i:j])
        if j == n:
            break
        i = j - overlap
    return out


def _to_epoch(ts_iso: str | None) -> int | None:
    """Convert ISO8601 to epoch seconds (useful for sorting/filtering)."""
    if not ts_iso:
        return None
    try:
        return int(dt.datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def _is_429_error(err: Exception) -> bool:
    """Best-effort detector for quota/capacity errors without importing SDK types."""
    s = str(err).lower()
    return ("429" in s) or ("resource exhausted" in s) or ("quota exceeded" in s)


def _embed_in_batches(texts: List[str], batch_size: int = 8, max_retries: int = 3) -> np.ndarray:
    """
    Batch embeddings to avoid request/token limits (esp. in Vertex).
    Includes simple retry with exponential backoff to soften transient 429s.
    """
    if not texts:
        return np.zeros((0, getattr(embedding_provider, "dim", 384)), dtype=np.float32)

    arrs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]

        attempt = 0
        while True:
            try:
                arrs.append(embedding_provider.embed_texts(chunk))
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries or not _is_429_error(e):
                    # Give up if it's not a 429-like error or we've retried too many times
                    raise
                # Exponential backoff with jitter
                sleep_s = min(1.5 ** attempt + random.uniform(0, 0.3), 6.0)
                print(f"Embedding batch retry {attempt} after 429 (~{sleep_s:.2f}s)...")
                time.sleep(sleep_s)  # ok to block here; this is not in the request path

    return np.vstack(arrs)


# --- Light snippet cleaner (defensive UI polish) -----------------------------

_SNIPPET_NOISE = re.compile(
    r"(suscripci[oó]n|inicia\s+sesi[oó]n|aceptar\s+cookies?|cookies?|"
    r"pol[ií]tica\s+de\s+cookies?|cookie\s?policy|privacy\s+policy|"
    r"newsletter|bolet[ií]n|haz\s+clic|haga\s+clic|contin[uú]a\s+leyendo|"
    r"ver\s+m[aá]s|solo\s+puedes\s+acceder|tu\s+suscripci[oó]n\s+se\s+est[aá]\s+usando|"
    r"dispositivo|publicidad|anuncios?|"
    r"al\s+continuar\s+navegando|configura\s+tus\s+preferencias|"
    r"aceptas\s+el\s+uso\s+de\s+cookies?)",
    re.IGNORECASE
)

def _first_clean_sentence(text: str) -> str:
    """
    Pick the first sentence-like span that doesn't look like boilerplate.
    Keep it short (<= 300 chars) for UI.
    """
    if not text:
        return ""

    soft = clean_text(text)
    parts = re.split(r'(?<=[\.\?!])\s+', soft)
    for s in parts:
        s = s.strip()
        if len(s) >= 40 and not _SNIPPET_NOISE.search(s):
            return s[:300]
    return soft.strip()[:300]


# --- Helpers to sanitize LLM JSON output -------------------------------------

_CODE_FENCE_LINE = re.compile(r"^\s*```.*$", re.IGNORECASE)

def _strip_code_fences(s: str) -> str:
    """
    Remove Markdown code fences (``` or ```json ...) from a string.
    """
    if not s:
        return ""
    return "\n".join(line for line in s.splitlines() if not _CODE_FENCE_LINE.match(line)).strip()

def _extract_first_json_object(s: str) -> str | None:
    """
    Extract the first JSON object from a string.
    Pragmatic approach: take substring from first '{' to last '}'.
    """
    if not s:
        return None
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return s[start:end + 1].strip()

def _trim_for_llm(text: str, max_chars: int = 8000) -> str:
    """Cap article text before sending to the LLM to reduce cost and errors."""
    return (text or "")[:max_chars]


# --- Agentic chunking (Gemini) -----------------------------------------------

def _ensure_vertexai_initialized():
    """
    Initialize Vertex AI only if needed (lazy).
    Using a mutable dict avoids 'global' assignment issues.
    """
    if _VERTEX_READY["ok"]:
        return
    try:
        from vertexai import init as vertexai_init
        project = settings.gcp_project
        location = settings.gcp_location or "us-central1"
        if not project:
            raise RuntimeError("Missing GCP project for Vertex AI (settings.gcp_project).")
        vertexai_init(project=project, location=location)
        _VERTEX_READY["ok"] = True
    except Exception as e:
        print("WARN Vertex AI init failed (agentic chunking may not work):", e)


def agentic_chunk(
    text: str,
    max_chunks: int = 5,
    fallback_simple: bool = True,
    simple_max_chars: int = 2000,
    simple_overlap: int = 200,
    retries: int = 2
) -> List[str]:
    """
    Use Gemini to produce semantic chunks. If it fails (quota/JSON/etc),
    fallback to the simple char-based splitter with overlap.
    Includes JSON sanitization and retry/backoff for transient 429s.
    """
    try:
        _ensure_vertexai_initialized()
        from vertexai.generative_models import GenerativeModel

        # Hard cap input to keep the model concise and cheap
        text_lite = _trim_for_llm(text, max_chars=8000)

        system = (
            "You are a JSON-only machine. "
            "You must segment long-form news articles into coherent contiguous chunks. "
            f"Return at most {max_chunks} chunks, each around 400–600 tokens. "
            "Remove cookie banners, subscription notices, ads, and social widgets. "
            "IMPORTANT: Output must be a single valid JSON object, no explanations, no markdown, no code fences."
        )

        user = (
            "Produce exactly this JSON structure:\n"
            '{ "chunks": ["chunk_1", "chunk_2", "..."] }\n\n'
            "Rules:\n"
            " - Output only one JSON object.\n"
            " - Do not include any commentary.\n"
            " - Do not use markdown fences.\n"
            " - Do not truncate chunks mid-sentence if possible.\n\n"
            f"Article:\n{text_lite}"
        )


        model = GenerativeModel("gemini-2.5-flash-lite")

        attempt = 0
        while True:
            attempt += 1
            try:
                resp = model.generate_content([system, user])
                raw = (resp.text or "").strip()

                # Debug: show raw output (first 500 chars)
                print("RAW Gemini response >>>", raw[:500])

                # Remove code fences and try to extract the first JSON object
                cleaned = _strip_code_fences(raw)
                candidate = _extract_first_json_object(cleaned) or cleaned

                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as je:
                    print("Gemini returned invalid JSON:", je)
                    print("Raw preview >>>", raw[:200])
                    parsed = {"chunks": []}

                chunks = parsed.get("chunks", [])
                chunks = [c.strip() for c in chunks if isinstance(c, str) and len(c.strip()) >= 80]

                if not chunks:
                    raise ValueError("Empty/invalid chunks from LLM")

                return chunks[:max_chunks]

            except Exception as e:
                # Retry only for 429-ish errors; otherwise break immediately
                if attempt <= retries and _is_429_error(e):
                    backoff = min(1.5 ** attempt + random.uniform(0, 0.3), 6.0)
                    print(f"Agentic 429, retry {attempt} in ~{backoff:.2f}s...")
                    time.sleep(backoff)
                    continue
                # Any other error → break to outer fallback
                raise

    except Exception as e:
        print("Agentic chunking failed -> falling back to simple splitter:", e)
        if fallback_simple:
            simple_chunks = _split_text_by_chars(text, max_chars=simple_max_chars, overlap=simple_overlap)
            return simple_chunks[:max_chunks]
        return [text]


# =============================================================================
# --------------------------------- /index ------------------------------------
# =============================================================================
@app.post("/index")
async def index_endpoint(payload: dict = Body(default={})):
    """
    Ingest RSS articles, clean them, chunk (simple or agentic), embed and upsert into Qdrant.

    Body params:
      - feed_url (optional): only index that feed; otherwise use default feeds.json
      - per_feed_target (int): how many good articles to keep per feed (default 2)
      - max_scan (int): how many items per feed to scan to reach the target (default 20)
      - max_chunks (int): cap chunks per article (default 3)
    """
    provider_chunking = settings.provider_chunking.lower()

    feed_url = payload.get("feed_url")
    per_feed_target = int(payload.get("per_feed_target") or 2)
    max_scan = int(payload.get("max_scan") or 20)
    max_chunks_per_article = int(payload.get("max_chunks") or 3)

    # 1) Pull articles
    articles = await (fetch_feed(feed_url, want=per_feed_target, max_scan=max_scan)
                      if feed_url else fetch_default(per_feed_target, max_scan))
    if not articles:
        raise HTTPException(status_code=400, detail="No articles fetched")

    # 2) Extra cleaning + quality guard
    cleaned = []
    for a in articles:
        txt = clean_text(a.get("content") or (f"{a.get('title','')} {a.get('url','')}"))
        ok, _ = should_index(txt, min_chars=400, min_score=700.0)
        if ok:
            a["content"] = txt
            cleaned.append(a)

    if not cleaned:
        raise HTTPException(status_code=400, detail="All articles failed quality filters")

    # --- Process in batches to avoid 429s (tune batch size if needed) ---
    BATCH_SIZE = 3  # smaller batches reduce LLM pressure
    total_chunks, total_articles = 0, 0

    for i in range(0, len(cleaned), BATCH_SIZE):
        batch = cleaned[i:i + BATCH_SIZE]

        SIMPLE_MAX_CHARS = 2000
        SIMPLE_OVERLAP = 200

        chunk_texts, chunk_payloads, chunk_ids = [], [], []

        for a in batch:
            # Prefer simple splitter for short articles; agentic for longer ones
            use_agentic = (provider_chunking == "agentic") and (len(a["content"]) >= 1500)

            if use_agentic:
                try:
                    chunks = agentic_chunk(a["content"], max_chunks=max_chunks_per_article)
                    # Small async pause between agentic calls to smooth traffic
                    await asyncio.sleep(random.uniform(0.15, 0.30))
                except Exception as e:
                    print(f"Agentic chunking failed for {a['title']}: {e}")
                    chunks = _split_text_by_chars(a["content"], max_chars=SIMPLE_MAX_CHARS, overlap=SIMPLE_OVERLAP)
            else:
                chunks = _split_text_by_chars(a["content"], max_chars=SIMPLE_MAX_CHARS, overlap=SIMPLE_OVERLAP)

            chunks = chunks[:max_chunks_per_article]

            # Build payloads
            for idx, ch in enumerate(chunks):
                chunk_texts.append(ch)
                chunk_ids.append(str(uuid.uuid4()))
                chunk_payloads.append({
                    "article_id": a["id"],
                    "title": a["title"],
                    "url": a["url"],
                    "published_at": a["published_at"],
                    "published_at_ts": _to_epoch(a.get("published_at")),
                    "source": a["source"],
                    "chunk_index": idx,
                    "chunk_preview": _first_clean_sentence(ch),
                    # keep a head of raw text to recover a clean snippet at search time
                    "chunk_text_head": ch[:1200],
                    "chunk_mode": "agentic" if use_agentic else "simple",
                })

        if not chunk_texts:
            continue

        # 4) Embeddings (with retry/backoff inside)
        passages = as_passages(
            chunk_texts,
            prefix_required=getattr(embedding_provider, "requires_e5_prefix", False)
        )
        vecs = _embed_in_batches(passages, batch_size=8)

        # 5) Upsert
        vector_index.upsert(ids=chunk_ids, vectors=vecs, payloads=chunk_payloads)

        print(f"Indexed batch {i // BATCH_SIZE + 1}: {len(chunk_ids)} chunks from {len(batch)} articles")
        total_chunks += len(chunk_ids)
        total_articles += len(batch)

    return {
        "indexed_articles": total_articles,
        "indexed_chunks": total_chunks,
        "chunk_mode": settings.provider_chunking.lower()
    }


# =============================================================================
# -------------------------------- /search ------------------------------------
# =============================================================================
@app.get("/search")
def search_endpoint(q: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=50)):
    """
    Semantic search: query -> embedding -> ANN -> group by article -> top-k articles.
    Returns article-level hits with a few representative snippets from the top chunks.
    """
    # 1) Encode query
    queries = as_queries(
        [q],
        prefix_required=getattr(embedding_provider, "requires_e5_prefix", False)
    )
    query_vec = embedding_provider.embed_texts(queries)[0]

    # 2) Fan-out (if your VectorIndex ignores hnsw_ef, it's fine)
    FANOUT = max(20, k * 8)
    try:
        hits = vector_index.search(vector=query_vec, top_k=FANOUT, hnsw_ef=256)  # type: ignore[arg-type]
    except TypeError:
        # For VectorIndex implementations that don't accept hnsw_ef
        hits = vector_index.search(vector=query_vec, top_k=FANOUT)

    # 3) Group by article
    from collections import defaultdict
    groups = defaultdict(list)
    for h in hits:
        aid = h.get("payload", {}).get("article_id")
        if aid:
            groups[aid].append(h)

    def group_score(hlist):
        return max(h["score"] for h in hlist) if hlist else 0.0

    ranked = sorted(groups.items(), key=lambda kv: group_score(kv[1]), reverse=True)[:k]

    # 4) Response (skip noisy snippets; try alternative chunk text head if needed)
    results = []
    for aid, hlist in ranked:
        hlist_sorted = sorted(hlist, key=lambda h: -h["score"])
        best = hlist_sorted[0]

        snippets = []
        seen_snips = set()

        # scan a few more than 3 to find up to 3 clean, distinct snippets
        for h in hlist_sorted[:8]:
            payload = h.get("payload", {}) or {}

            # 1) try stored preview (already cleaned)
            snip = payload.get("chunk_preview") or ""

            # 2) if preview looks noisy/empty, try from raw chunk head
            if not snip or _SNIPPET_NOISE.search(snip):
                head = payload.get("chunk_text_head") or ""
                snip = _first_clean_sentence(head) if head else ""

            # 3) final fallback: trimmed head (last resort, still avoids obvious noise)
            if not snip and payload.get("chunk_text_head"):
                snip = (payload["chunk_text_head"][:280] + "…")

            snip = (snip or "").strip()
            if not snip or _SNIPPET_NOISE.search(snip):
                continue

            # avoid duplicates
            key = snip.lower()
            if key in seen_snips:
                continue
            seen_snips.add(key)
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
