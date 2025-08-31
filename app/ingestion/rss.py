# app/ingestion/rss.py
from __future__ import annotations
import os, json, pathlib, asyncio, uuid, datetime as dt
from typing import List, Tuple
import urllib.parse as urlparse

import feedparser
import httpx
import trafilatura

from app.ingestion.quality import clean_text, should_index

# -----------------------------------------------------------------------------
# Utilities & config
# -----------------------------------------------------------------------------

def _make_id(url: str) -> str:
    """Build a stable UUID5 from the article URL (works well with vector DBs)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))


def _to_iso(parsed_time) -> str | None:
    """Convert a feedparser time struct to ISO-8601."""
    if parsed_time:
        return dt.datetime(*parsed_time[:6], tzinfo=dt.timezone.utc).isoformat()
    return None


def _normalize_url(u: str | None) -> str | None:
    """
    Normalize URLs to reduce duplicates and stabilize IDs:
    - force https schema if possible
    - drop common tracking query params (utm_*, fbclid, gclid, ...).
    - strip trailing slash
    """
    if not u:
        return u
    try:
        p = urlparse.urlsplit(u.strip())
        scheme = "https" if p.scheme in ("http", "https") else "https"

        # Keep only non-tracking params
        tracking_prefixes = ("utm_", "_hs", "mc_", "fbclid", "gclid", "igshid")
        q = urlparse.parse_qsl(p.query, keep_blank_values=False)
        q_clean = [(k, v) for (k, v) in q if not k.lower().startswith(tracking_prefixes)]
        query = urlparse.urlencode(q_clean, doseq=True)

        path = p.path.rstrip("/")
        norm = urlparse.urlunsplit((scheme, p.netloc.lower(), path, query, ""))

        return norm
    except Exception:
        return u


def load_default_feeds() -> list[str]:
    """
    Load feed URLs from a JSON file.
    Default path: app/feeds.json
    Overridable via env var: FEEDS_FILE
    Expected JSON shape:
      { "feeds": ["https://...", "https://..."] }
    """
    feeds_path = os.getenv("FEEDS_FILE")
    path = pathlib.Path(feeds_path) if feeds_path else pathlib.Path(__file__).resolve().parent.parent / "feeds.json"

    if not path.exists():
        # Conservative fallback so the system doesn't break
        return [
            "https://www.bbc.com/mundo/index.xml",
            "https://www.reuters.com/world/rss",
            "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/america/portada",
        ]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    feeds = data.get("feeds", [])
    seen, out = set(), []
    for u in feeds:
        u = (u or "").strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


# A real-ish UA helps some publishers serve full HTML instead of blocked views.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-419,es;q=0.9,en;q=0.8",
}


async def _fetch_page(client: httpx.AsyncClient, url: str) -> str | None:
    """Download HTML with retries and redirects."""
    for attempt in range(3):
        try:
            resp = await client.get(url, headers=HEADERS, follow_redirects=True, timeout=10.0)
            if resp.status_code == 200 and resp.text:
                return resp.text
        except Exception:
            await asyncio.sleep(0.5 * (attempt + 1))
    return None


def _truncate(text: str, max_chars: int = 8000) -> str:
    """
    Hard cut long texts to keep token budgets sane before LLM/embeddings.
    8k chars is ~2–3k tokens depending on language (rough ballpark).
    """
    return text[:max_chars] if text else text


def _extract_with_trafilatura(html: str, url: str | None = None) -> str | None:
    """
    Robust article extraction via trafilatura.
    Returns cleaned text or None if extraction is not good enough.
    """
    if not html:
        return None
    try:
        txt = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            favor_precision=True,
        )
        if txt and len(txt) > 200:
            return txt.strip()
    except Exception:
        pass
    return None


def _pick_summary_or_title(summary: str, title: str) -> str:
    """Last-resort content if we couldn't fetch a decent body."""
    s = (summary or "").strip()
    t = (title or "").strip()
    return s if len(s) >= 40 else (t or s)


# -----------------------------------------------------------------------------
# Ingestion
# -----------------------------------------------------------------------------

async def fetch_feed(feed_url: str, want: int = 2, max_scan: int = 20) -> list[dict]:
    """
    Read an RSS feed and return up to `want` high-quality articles.
    If the first items are weak, keep scanning up to `max_scan`.
    """
    feed = feedparser.parse(feed_url)
    out: list[dict] = []
    if not getattr(feed, "entries", None):
        return out

    async with httpx.AsyncClient(timeout=10.0) as client:
        for entry in feed.entries[:max_scan]:
            if len(out) >= want:
                break

            raw_url = entry.get("link")
            if not raw_url:
                continue

            url = _normalize_url(raw_url)
            title = (entry.get("title") or "").strip()
            summary = (entry.get("summary") or "").strip()
            published = _to_iso(getattr(entry, "published_parsed", None)) \
                        or _to_iso(getattr(entry, "updated_parsed", None))
            source_name = feed.feed.get("title", "rss")

            # (1) Try full body from feed (content:encoded). Many WordPress-like feeds include HTML here.
            content: str | None = None
            try:
                content_list = entry.get("content") or []
                if content_list:
                    html_cand = (content_list[0].get("value") or "").strip()
                    if html_cand:
                        content = _extract_with_trafilatura(html_cand, url=None)
            except Exception:
                # non-fatal; just keep going
                pass

            # (2) If feed body isn’t good enough, fetch the page and extract main content
            if not content or len(content) < 400:
                html = await _fetch_page(client, url)
                if html:
                    content = _extract_with_trafilatura(html, url=url)

            # (3) Last fallback: use summary/title
            if not content or len(content) < 200:
                content = _pick_summary_or_title(summary, title)

            # (4) Early normalization + quality filter before LLM/chunking
            content = clean_text(content or "")
            # Trim super long content to reduce downstream token consumption
            content = _truncate(content, max_chars=8000)

            ok, reason = should_index(content, min_chars=400, min_score=700.0)
            if not ok:
                # Skip low-quality or boilerplate-only items
                continue

            out.append({
                "id": _make_id(url),
                "url": url,
                "title": title,
                "content": content,
                "published_at": published,
                "source": source_name,
                "lang": None,
            })

    return out


async def fetch_default(per_feed_target: int = 2, max_scan: int = 20) -> list[dict]:
    """
    Try to collect `per_feed_target` good articles per feed.
    If some feeds under-deliver, others are not blocked.
    """
    feeds = load_default_feeds()
    tasks = [fetch_feed(f, want=per_feed_target, max_scan=max_scan) for f in feeds]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles: list[dict] = []
    for r in results:
        if isinstance(r, list):
            articles.extend(r)

    # De-dup by normalized URL-based id (same story might appear across different feeds)
    seen, dedup = set(), []
    for a in articles:
        if a["id"] not in seen:
            seen.add(a["id"])
            dedup.append(a)
    return dedup
