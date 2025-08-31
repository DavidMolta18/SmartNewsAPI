# app/ingestion/quality.py
from __future__ import annotations
import re
from typing import List, Tuple

# =============================================================================
# Lightweight, conservative text cleaning for news articles.
# Goal: remove obvious boilerplate (cookies, subscriptions, UX banners)
# while keeping the editorial body intact. This runs BEFORE the LLM so we
# reduce token waste without risking dropping important content.
# =============================================================================

# Toggle extra debug prints when tuning (keep False for prod).
DEBUG = False

# --- Common boilerplate / paywall / subscription patterns seen across outlets ---
# Keep this list conservative. Fine-grained cleanup should be left to the LLM or a
# downstream agentic step. Here we only filter *obvious* junk.
_BOILERPLATE_PATTERNS = [
    # cookies / privacy
    r"\b(?:aceptar|accept)(?:\s+all)?\s+cookies\b",
    r"\bpol[ií]tica\s+de\s+cookies\b",
    r"\bcookie\s?policy\b",
    r"\bprivacy\s+policy\b",

    # login / subscription / paywall
    r"\bsuscr[ií]bete\b", r"\breg[ií]strate\b", r"\binicia\s+sesi[oó]n\b",
    r"\blector\s+premium\b", r"\bplanes?\s+de\s+suscripci[oó]n\b",
    r"\bsolo\s+puedes\s+acceder.*dispositivo\b",
    r"\b¿?quieres\s+añadir\s+otro\s+usuario\b",
    r"\bcompartir\s+tu\s+cuenta\b",
    r"\btu\s+suscripci[oó]n\s+se\s+est[aá]\s+usando\s+en\s+otro\s+dispositivo\b",

    # promos / CTAs / UX banners
    r"\bboletines?\b", r"\bnewsletter\b",
    r"\bhaz\s+clic\s+aqu[ií]\b", r"\bcontin[uú]a\s+leyendo\b",
    r"\bver\s+m[aá]s\b", r"\blee\s+tambi[eé]n\b",
    r"\bpublicidad\b", r"\banuncios\b",

    # social widgets / follow-us
    r"\bs[ií]guenos\s+en\s+(?:facebook|x|twitter|instagram|tiktok)\b",

    # generic consent / preferences
    r"\bal\s+continuar\s+navegando\b",
    r"\bconfigura\s+tus\s+preferencias\b",
]

# Lines that are typical nav/menu labels (single-word or very short items)
_NAVLIKE_LINE = re.compile(
    r"^(?:menu|men[uú]|home|inicio|buscar|search|portada|última hora)$",
    re.IGNORECASE,
)

# One big regex to detect boilerplate anywhere (fast coarse check)
_RE_BOILER = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)

# Basic URL / link detection
_RE_LINK = re.compile(r"https?://|www\.", re.IGNORECASE)

# Sentence-ish splitter: split on ., ?, ! followed by whitespace/newline.
_SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+")

# Collapse 3+ blank lines into just 2 (preserve paragraphs but remove noise)
_RE_BLANKS = re.compile(r"\n{3,}")

# Replace repeated spaces/tabs with single spaces
_RE_SPACES = re.compile(r"[ \t]{2,}")

# Remove repeated identical lines (common in sticky banners)
def _dedupe_adjacent_lines(lines: List[str]) -> List[str]:
    out, last = [], None
    for l in lines:
        if l and l != last:
            out.append(l)
        last = l
    return out

def _is_shouty(s: str, min_len: int = 10, max_upper_ratio: float = 0.7) -> bool:
    """Detect lines that are mostly uppercase (often headers/banners)."""
    letters = [c for c in s if c.isalpha()]
    if len(letters) < min_len:
        return False
    uppers = sum(1 for c in letters if c.isupper())
    return (uppers / max(1, len(letters))) >= max_upper_ratio

def _normalize(text: str) -> str:
    """Normalize newlines, whitespace and common invisible characters."""
    if not text:
        return ""
    # Normalize Windows/Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace non-breaking spaces and odd unicode spaces
    text = text.replace("\u00A0", " ").replace("\u200B", "")
    # Strip trailing spaces per line
    lines = [l.strip() for l in text.split("\n")]
    # De-duplicate exact adjacent lines (sticky banners repeated)
    lines = _dedupe_adjacent_lines(lines)
    return "\n".join(lines).strip()

def _strip_navlike_and_boiler_lines(lines: List[str]) -> List[str]:
    """
    Remove menu-like one-liners and lines that are clearly boilerplate.
    Conservative: we only drop a line if it is short or matches obvious noise.
    """
    out = []
    for l in lines:
        if not l:
            continue
        # very short nav-like labels (e.g., "Inicio", "Buscar")
        if len(l) <= 18 and _NAVLIKE_LINE.match(l):
            continue
        # drop small boilerplate banners/prompts
        if _RE_BOILER.search(l) and len(l) <= 200:
            continue
        # drop shouty headings that are likely layout noise
        if len(l) <= 120 and _is_shouty(l):
            continue
        out.append(l)
    return out

def _sentences(text: str) -> List[str]:
    """
    Split into sentence-like chunks. If the text has few terminators,
    fallback to paragraph-level chunks to avoid over-fragmentation.
    """
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    if len(parts) <= 2:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [p.strip() for p in parts if p and p.strip()]

def _filter_sentences(sentences: List[str]) -> List[str]:
    """
    Keep sentences that look like editorial prose (not banners, not shouty),
    and that are long enough to be meaningful.
    """
    out = []
    for s in sentences:
        if len(s) < 25:
            continue
        if _RE_BOILER.search(s):
            continue
        if _is_shouty(s):
            continue
        out.append(s)
    return out

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def clean_text(txt: str) -> str:
    """
    Conservative cleaner (runs before the LLM):
      1) normalize whitespace/newlines and remove duplicated sticky lines
      2) drop nav-like lines and small boiler banners
      3) lightly filter at sentence level (short/shouty/boiler)
      4) reassemble into coherent paragraphs
    If filtering was too strict and we lost most content, we fall back to the
    lightly normalized text to preserve recall.
    """
    if not txt:
        return ""

    # Step 1: normalize and split
    txt = _normalize(txt)
    lines = [l for l in txt.split("\n")]

    # Step 2: remove trivial nav-like labels / small boilerplate lines
    lines = _strip_navlike_and_boiler_lines(lines)

    # Collapse excessive blank lines (keep paragraph breaks)
    text = "\n".join(lines)
    text = _RE_BLANKS.sub("\n\n", text).strip()

    # Step 3: sentence-level conservative filtering
    sents = _sentences(text)
    sents = _filter_sentences(sents)

    # Step 4: fallback if we were too strict
    if len(sents) < 3 and len(text) > 0:
        fallback = _RE_SPACES.sub(" ", text).strip()
        if DEBUG:
            print("[quality.clean_text] Fallback to lightly-normalized text")
        return fallback

    # Step 5: reassemble
    cleaned = " ".join(sents)
    cleaned = _RE_SPACES.sub(" ", cleaned).strip()
    return cleaned

def is_boilerplate(txt: str) -> bool:
    """
    Consider text boilerplate if multiple boiler patterns appear.
    Threshold kept at 2 to avoid false positives on legit mentions.
    """
    if not txt:
        return True
    matches = _RE_BOILER.findall(txt)
    return len(matches) >= 2

def quality_score(txt: str) -> float:
    """
    Simple heuristic: length + prose signals - link penalties.
    Intentionally lightweight; the goal is to catch extremes early.
    """
    if not txt:
        return 0.0

    n = len(txt)

    # Prose signals: sentence terminators and paragraph breaks
    bonus = 0
    if re.search(r"[\.!?…。]", txt):
        bonus += 300
    if "\n\n" in txt or re.search(r"\s{2,}\n", txt):
        bonus += 300

    # Link penalty (URLs rarely belong to article body in bulk)
    links = len(_RE_LINK.findall(txt))
    penalty = 150 * links

    # Very high link density penalty (defensive)
    tokens = re.findall(r"\w+|\S", txt)
    link_ratio = links / max(1, len(tokens))
    if link_ratio > 0.05:  # >5% tokens are links looks spammy
        penalty += int(800 * (link_ratio - 0.05))

    score = max(0, n + bonus - penalty)
    if DEBUG:
        print(f"[quality.quality_score] len={n} links={links} bonus={bonus} penalty={penalty} => {score}")
    return score

def should_index(txt: str, min_chars: int = 400, min_score: float = 700.0) -> Tuple[bool, str]:
    """
    Decide if a cleaned text is worth indexing.
    Keep thresholds modest; we prefer recall here because the LLM/chunker
    will further refine content. This is just a first guardrail.
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
