# app/ingestion/rss.py
from __future__ import annotations
import os, json, pathlib, asyncio
import uuid, datetime as dt
import feedparser, httpx, trafilatura
from app.ingestion.quality import clean_text, should_index

# ---------- Config y utilidades ----------
def _make_id(url: str) -> str:
    """Genera un UUID5 a partir de la URL del artículo (válido para Qdrant)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))

def _to_iso(parsed_time):
    """Convierte un parsed_time de feedparser a ISO8601."""
    if parsed_time:
        return dt.datetime(*parsed_time[:6], tzinfo=dt.timezone.utc).isoformat()
    return None

def load_default_feeds() -> list[str]:
    """
    Carga la lista de feeds desde un archivo JSON.
    - Por defecto: app/feeds.json
    - Se puede sobreescribir con la variable FEEDS_FILE
    Estructura del JSON:
    { "feeds": ["https://...", "https://..."] }
    """
    feeds_path = os.getenv("FEEDS_FILE")
    if feeds_path:
        path = pathlib.Path(feeds_path)
    else:
        path = pathlib.Path(__file__).resolve().parent.parent / "feeds.json"

    if not path.exists():
        # Fallback defensivo: lista mínima para no romper
        return [
            "https://www.bbc.com/mundo/index.xml",
            "https://www.reuters.com/world/rss",
            "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/america/portada",
        ]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    feeds = data.get("feeds", [])
    # Sanitizar: quitar duplicados y vacíos
    seen, out = set(), []
    for u in feeds:
        u = (u or "").strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out

HEADERS = {
    # UA “real” ayuda a que algunos medios sirvan HTML completo
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept-Language": "es-419,es;q=0.9,en;q=0.8",
}

async def _fetch_page(client: httpx.AsyncClient, url: str) -> str | None:
    """Descarga HTML con reintentos y follow_redirects."""
    for attempt in range(3):
        try:
            resp = await client.get(url, headers=HEADERS, follow_redirects=True, timeout=10.0)
            if resp.status_code == 200 and resp.text:
                return resp.text
        except Exception:
            await asyncio.sleep(0.5 * (attempt + 1))
    return None

def _truncate(text: str, max_chars: int = 8000) -> str:
    """Recorta texto largo para cumplir límite de Vertex (~20k tokens)."""
    return text[:max_chars] if text else text

def _extract_with_trafilatura(html: str, url: str | None = None) -> str | None:
    """Extracción robusta con trafilatura; devuelve texto o None."""
    if not html:
        return None
    try:
        txt = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            favor_precision=True
        )
        if txt and len(txt) > 200:
            return txt.strip()
    except Exception:
        pass
    return None

# ---------- Ingesta ----------
async def fetch_feed(feed_url: str, max_items: int = 10) -> list[dict]:
    """Lee un feed RSS y retorna artículos normalizados (con filtro de calidad)."""
    feed = feedparser.parse(feed_url)
    out = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        for entry in feed.entries[:max_items]:
            url = entry.get("link")
            if not url:
                continue

            title = (entry.get("title") or "").strip()
            summary = (entry.get("summary") or "").strip()
            published = _to_iso(getattr(entry, "published_parsed", None)) \
                        or _to_iso(getattr(entry, "updated_parsed", None))
            source_name = feed.feed.get("title", "rss")

            # 1) Intentar cuerpo completo desde el RSS (content:encoded)
            content = None
            try:
                content_list = entry.get("content") or []
                if content_list:
                    # Muchos feeds (WordPress) traen el body HTML aquí
                    html_cand = (content_list[0].get("value") or "").strip()
                    if html_cand:
                        content = _extract_with_trafilatura(html_cand, url=None)
            except Exception:
                pass

            # 2) Si no hay cuerpo decente, extraer desde la página
            if not content or len(content) < 400:
                html = await _fetch_page(client, url)
                if html:
                    content = _extract_with_trafilatura(html, url=url)

            # 3) Fallback final: summary / title
            if not content or len(content) < 200:
                content = summary or title

            # 4) Limpieza + filtro de calidad (descarta basura)
            content = clean_text(content or "")
            ok, reason = should_index(content, min_chars=400, min_score=700.0)
            if not ok:
                # Puedes subir a logging.debug si quieres
                # print(f"SKIP [{source_name}] {title[:60]} -> {reason}")
                continue

            out.append({
                "id": _make_id(url),
                "url": url,
                "title": title,
                "content": _truncate(content, max_chars=8000),
                "published_at": published,
                "source": source_name,
                "lang": None,
            })

    return out

async def fetch_default(max_items_per_feed: int = 5) -> list[dict]:
    """Obtiene artículos de los feeds por defecto (paralelo + dedupe)."""
    feeds = load_default_feeds()
    tasks = [fetch_feed(f, max_items_per_feed) for f in feeds]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles = []
    for r in results:
        if isinstance(r, list):
            articles.extend(r)

    # Deduplicar por id (misma noticia puede aparecer en múltiples feeds)
    seen, dedup = set(), []
    for a in articles:
        if a["id"] not in seen:
            seen.add(a["id"])
            dedup.append(a)
    return dedup
