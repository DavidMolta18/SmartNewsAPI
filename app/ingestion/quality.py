# app/ingestion/quality.py
from __future__ import annotations
import re
from typing import List, Tuple
from html import unescape
from app.utils.patterns import RE_BOILER  # central boilerplate regex

# =============================================================================
# Conservative text cleaner for full news articles.
# Goal: remove obvious boilerplate (cookies, subscriptions, banners) while
# preserving the editorial body intact. This runs BEFORE LLMs, to reduce
# wasted tokens without risking important content loss.
# =============================================================================

DEBUG = False  # Toggle verbose debug logs for development

# --- Try optional HTML parser ---
try:
    from bs4 import BeautifulSoup  # type: ignore
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False

# --- Regex helpers ---
_NAVLIKE_LINE = re.compile(
    r"^(?:menu|men[uú]|home|inicio|buscar|search|portada|última hora)$",
    re.IGNORECASE,
)
_RE_LINK = re.compile(r"https?://|www\.", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+")
_RE_BLANKS = re.compile(r"\n{3,}")   # collapse 3+ newlines into 2
_RE_SPACES = re.compile(r"[ \t]{2,}")  # collapse multiple spaces/tabs
_TAG_RE = re.compile(r"<[^>]+>")  # generic HTML tag stripper
_CDATA_OPEN = re.compile(r"<!\[CDATA\[", re.IGNORECASE)
_CDATA_CLOSE = re.compile(r"\]\]>", re.IGNORECASE)

# Elimina bloques completos de medios incrustados comunes que no aportan texto
_REMOVE_BLOCK_TAGS = re.compile(
    r"</?(?:script|style|noscript|iframe|figure|figcaption|video|audio|source|picture|svg|canvas|form|input|button)[^>]*>",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# HTML / markup stripping
# ----------------------------------------------------------------------
def strip_html(raw: str) -> str:
    """
    Remove CDATA wrappers, HTML tags, embedded media blocks and normalize entities.
    Prefer BeautifulSoup when available; fallback to regex otherwise.
    """
    if not raw:
        return ""

    # Quita CDATA
    raw = _CDATA_OPEN.sub("", raw)
    raw = _CDATA_CLOSE.sub("", raw)

    # A veces el feed trae bloques de media/players dentro de content:encoded
    raw = _REMOVE_BLOCK_TAGS.sub(" ", raw)

    if _HAS_BS4:
        try:
            soup = BeautifulSoup(raw, "lxml")  # lxml es rápido y robusto si está disponible
            # Eliminamos también script/style por si el parser los conserva
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(" ")  # separador de espacios para no pegar palabras
        except Exception:
            # Fallback duro a regex si algo falla en bs4/lxml
            text = _TAG_RE.sub(" ", raw)
    else:
        text = _TAG_RE.sub(" ", raw)

    # Decodifica &amp; &quot; etc
    text = unescape(text)

    # Limpieza básica de espacios
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    # Restituye saltos lógicos básicos donde había párrafos
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _dedupe_adjacent_lines(lines: List[str]) -> List[str]:
    """Remove consecutive duplicate lines (common in sticky banners)."""
    out, last = [], None
    for l in lines:
        if l and l != last:
            out.append(l)
        last = l
    return out


def _is_shouty(s: str, min_len: int = 10, max_upper_ratio: float = 0.7) -> bool:
    """Detect lines that are mostly uppercase (typical of banners/headers)."""
    letters = [c for c in s if c.isalpha()]
    if len(letters) < min_len:
        return False
    uppers = sum(1 for c in letters if c.isupper())
    return (uppers / max(1, len(letters))) >= max_upper_ratio


def _normalize(text: str) -> str:
    """Normalize newlines, whitespace, and invisible characters."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ").replace("\u200B", "")
    lines = [l.strip() for l in text.split("\n")]
    lines = _dedupe_adjacent_lines(lines)
    return "\n".join(lines).strip()


def _strip_navlike_and_boiler_lines(lines: List[str]) -> List[str]:
    """
    Remove short nav-like labels (e.g., "Inicio") and boilerplate one-liners.
    Conservative: only drop lines that are clearly noise.
    """
    out = []
    for l in lines:
        if not l:
            continue
        if len(l) <= 18 and _NAVLIKE_LINE.match(l):
            continue
        if RE_BOILER.search(l) and len(l) <= 200:
            continue
        if len(l) <= 120 and _is_shouty(l):
            continue
        out.append(l)
    return out


def _sentences(text: str) -> List[str]:
    """
    Split into sentence-like chunks. If very few terminators, fall back to
    paragraph-level chunks to avoid over-fragmentation.
    """
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    if len(parts) <= 2:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [p.strip() for p in parts if p and p.strip()]


def _filter_sentences(sentences: List[str]) -> List[str]:
    """Keep only sentences that look like real prose (not banners/noise)."""
    out = []
    for s in sentences:
        if len(s) < 25:
            continue
        if RE_BOILER.search(s):
            continue
        if _is_shouty(s):
            continue
        out.append(s)
    return out

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def clean_text(txt: str) -> str:
    """
    Conservative cleaner for raw article text.
    Pipeline (now with HTML stripping first):
      0) strip_html: remove CDATA/HTML/tags/entities
      1) Normalize whitespace and line endings
      2) Drop nav-like labels and obvious boilerplate lines
      3) Lightly filter sentences (short, shouty, boiler-like)
      4) Reassemble into coherent paragraphs

    If cleaning is too strict and most content is lost, fall back to
    lightly normalized text.
    """
    if not txt:
        return ""

    # --- NUEVO: limpia HTML/CDATA/etiquetas antes de todo ---
    txt = strip_html(txt)

    txt = _normalize(txt)
    lines = txt.split("\n")

    lines = _strip_navlike_and_boiler_lines(lines)

    text = "\n".join(lines)
    text = _RE_BLANKS.sub("\n\n", text).strip()

    sents = _sentences(text)
    sents = _filter_sentences(sents)

    if len(sents) < 3 and len(text) > 0:
        fallback = _RE_SPACES.sub(" ", text).strip()
        if DEBUG:
            print("[quality.clean_text] Fallback to lightly-normalized text")
        return fallback

    cleaned = " ".join(sents)
    cleaned = _RE_SPACES.sub(" ", cleaned).strip()
    return cleaned


def is_boilerplate(txt: str) -> bool:
    """
    Consider text boilerplate if multiple boiler patterns are found.
    Threshold = 2 to avoid false positives on legit mentions.
    """
    if not txt:
        return True
    matches = RE_BOILER.findall(txt)
    return len(matches) >= 2


def quality_score(txt: str) -> float:
    """
    Lightweight heuristic for text quality:
      + Score for length and prose signals
      - Penalty for excessive links (spam-like)
    """
    if not txt:
        return 0.0

    n = len(txt)
    bonus = 0
    if re.search(r"[\.!?…。]", txt):
        bonus += 300
    if "\n\n" in txt or re.search(r"\s{2,}\n", txt):
        bonus += 300

    links = len(_RE_LINK.findall(txt))
    penalty = 150 * links

    tokens = re.findall(r"\w+|\S", txt)
    link_ratio = links / max(1, len(tokens))
    if link_ratio > 0.05:
        penalty += int(800 * (link_ratio - 0.05))

    score = max(0, n + bonus - penalty)
    if DEBUG:
        print(f"[quality.quality_score] len={n} links={links} bonus={bonus} penalty={penalty} => {score}")
    return score


def should_index(txt: str, min_chars: int = 400, min_score: float = 700.0) -> Tuple[bool, str]:
    """
    Decide whether a cleaned text is good enough to index.
    Conservative thresholds: prefer recall, since LLM/chunkers refine later.
    """
    if not txt:
        return False, "empty"
    if is_boilerplate(txt):
        return False, "boilerplate"
    if len(txt) < min_chars:
        return False, f"too_short({len(txt)})"

    score = quality_score(txt)
    if score < min_score:
        return False, f"low_score({int(score)})"

    return True, "ok"
