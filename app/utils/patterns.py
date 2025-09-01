# app/utils/patterns.py
"""
Centralized regex patterns for boilerplate detection in news text.
Used by quality.py (document-level cleaning) and text_utils.py (snippet cleanup).
"""

import re

# --- Generic HTML/CDATA noise often seen in RSS feeds (e.g., La FM, WP, etc.)
RE_HTML_TAG = re.compile(r"<[^>]+>")  # remove any remaining HTML tag
RE_CDATA_BLOCK = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL | re.IGNORECASE)

# --- Base patterns seen across news outlets (Spanish + some English)
COMMON_PATTERNS = [
    # cookies / privacy
    r"\b(?:aceptar|accept)(?:\s+all)?\s+cookies\b",
    r"\bpol[ií]tica\s+de\s+cookies\b",
    r"\bcookie\s?policy\b",
    r"\bprivacy\s+policy\b",

    # login / subscription / paywall
    r"\bsuscr[ií]bete\b",
    r"\breg[ií]strate\b",
    r"\binicia\s+sesi[oó]n\b",
    r"\blector\s+premium\b",
    r"\bplanes?\s+de\s+suscripci[oó]n\b",
    r"\bsolo\s+puedes\s+acceder.*dispositivo\b",
    r"\btu\s+suscripci[oó]n\s+se\s+est[aá]\s+usando\s+en\s+otro\s+dispositivo\b",

    # promos / CTAs / UX banners / link-outs
    r"\bboletines?\b",
    r"\bnewsletter\b",
    r"\bhaz\s+clic\s+aqu[ií]\b",
    r"\bhaga\s+clic\b",
    r"\bcontin[uú]a\s+leyendo\b",
    r"\bver\s+m[aá]s\b",
    r"\blee\s+tambi[eé]n\b",
    r"\blea\s+(?:tambi[eé]n|adem[aá]s|aqu[ií])\b",
    r"\ble\s+puede\s+interesar\b",
    r"\bvea\s+(?:tambi[eé]n|aqu[ií])\b",
    r"\brelacionad[oa]s?:?\b",
    r"\bpublicidad\b",
    r"\banuncios?\b",

    # social widgets / share
    r"\bs[ií]guenos\s+en\s+(?:facebook|x|twitter|instagram|tiktok)\b",
    r"\b(compartir|comp[aá]rtelo)\s+en\s+(?:facebook|x|twitter|instagram|tiktok)\b",

    # generic consent
    r"\bal\s+continuar\s+navegando\b",
    r"\bconfigura\s+tus\s+preferencias\b",

    # media embeds / captions
    r"\bver\s+video\b",
    r"\bvideo:\b",
    r"\byoutube\.com\/watch\?v=",
]

# Compile reusable regexes
RE_BOILER = re.compile("|".join(COMMON_PATTERNS), re.IGNORECASE)

# For snippet filtering (lighter, UI-facing) — can be more aggressive
# For snippet filtering (lighter, UI-facing)
SNIPPET_NOISE = re.compile("|".join([
    # subscriptions / logins / consent
    r"suscripci[oó]n",                
    r"suscr[ií]b[ei]te",              
    r"inicia\s+sesi[oó]n",
    r"aceptar\s+cookies?",
    r"cookies?",
    r"privacy\s+policy",

    # promos / CTAs
    r"newsletter",
    r"\bbolet[ií]n\b",                
    r"haz\s+clic",
    r"haga\s+clic",
    r"contin[uú]a\s+leyendo",
    r"ver\s+m[aá]s",
    r"lee\s+tambi[eé]n",
    r"lea\s+(?:tambi[eé]n|adem[aá]s|aqu[ií])",
    r"le\s+puede\s+interesar",
    r"vea\s+(?:tambi[eé]n|aqu[ií])",
    r"relacionad[oa]s?:?",

    # device / ads / prefs
    r"solo\s+puedes\s+acceder",
    r"dispositivo",
    r"publicidad",
    r"anuncios?",
    r"configura\s+tus\s+preferencias",

    # media embeds / consent banners
    r"ver\s+video",
    r"youtube\.com\/watch\?v=",
    r"si\s+contin[uú]a\s+navegando",
]), re.IGNORECASE)

