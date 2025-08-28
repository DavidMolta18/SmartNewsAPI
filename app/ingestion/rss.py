# app/ingestion/rss.py
from __future__ import annotations
import hashlib, datetime as dt
import feedparser, httpx, trafilatura

# Feeds que usaremos como ejemplo
DEFAULT_FEEDS = [
    "https://www.bbc.com/mundo/index.xml",
    "https://www.reuters.com/world/rss",
    "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/america/portada"
]

import uuid

def _make_id(url: str) -> str:
    """Genera un UUID5 a partir de la URL del artículo (válido para Qdrant)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))


def _to_iso(parsed_time):
    """Convierte un parsed_time de feedparser a ISO8601."""
    if parsed_time:
        return dt.datetime(*parsed_time[:6], tzinfo=dt.timezone.utc).isoformat()
    return None

async def fetch_feed(feed_url: str, max_items: int = 10) -> list[dict]:
    """Lee un feed RSS y retorna artículos normalizados."""
    feed = feedparser.parse(feed_url)
    out = []
    async with httpx.AsyncClient(timeout=10) as client:
        for entry in feed.entries[:max_items]:
            url = entry.get("link")
            if not url:
                continue
            title = entry.get("title", "").strip()
            summary = (entry.get("summary") or "").strip()
            published = _to_iso(getattr(entry, "published_parsed", None)) \
                        or _to_iso(getattr(entry, "updated_parsed", None))

            # Intentamos extraer texto completo de la página
            content = summary
            try:
                resp = await client.get(url, follow_redirects=True)
                if resp.status_code == 200:
                    extracted = trafilatura.extract(resp.text, include_comments=False)
                    if extracted and len(extracted) > 200:
                        content = extracted.strip()
            except Exception:
                pass

            out.append({
                "id": _make_id(url),
                "url": url,
                "title": title,
                "content": content or title,
                "published_at": published,
                "source": feed.feed.get("title", "rss"),
                "lang": None,
            })
    return out

async def fetch_default(max_items_per_feed: int = 5) -> list[dict]:
    """Obtiene artículos de los feeds por defecto."""
    articles = []
    for f in DEFAULT_FEEDS:
        articles.extend(await fetch_feed(f, max_items_per_feed))
    return articles
