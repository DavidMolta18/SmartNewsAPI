# app/utils/patterns.py
"""
Centralized regex patterns for boilerplate detection in news text.
This avoids duplication between quality.py (document-level cleaning)
and text_utils.py (snippet-level cleanup).
"""

import re

# --- Base patterns seen across news outlets ---
COMMON_PATTERNS = [
    # cookies / privacy
    r"\b(?:aceptar|accept)(?:\s+all)?\s+cookies\b",
    r"\bpol[ií]tica\s+de\s+cookies\b",
    r"\bcookie\s?policy\b",
    r"\bprivacy\s+policy\b",

    # login / subscription / paywall
    r"\bsuscr[ií]bete\b", r"\breg[ií]strate\b", r"\binicia\s+sesi[oó]n\b",
    r"\blector\s+premium\b", r"\bplanes?\s+de\s+suscripci[oó]n\b",
    r"\bsolo\s+puedes\s+acceder.*dispositivo\b",
    r"\btu\s+suscripci[oó]n\s+se\s+est[aá]\s+usando\s+en\s+otro\s+dispositivo\b",

    # promos / CTAs / UX banners
    r"\bboletines?\b", r"\bnewsletter\b",
    r"\bhaz\s+clic\s+aqu[ií]\b", r"\bhaga\s+clic\b",
    r"\bcontin[uú]a\s+leyendo\b", r"\bver\s+m[aá]s\b",
    r"\blee\s+tambi[eé]n\b",
    r"\bpublicidad\b", r"\banuncios?\b",

    # social widgets
    r"\bs[ií]guenos\s+en\s+(?:facebook|x|twitter|instagram|tiktok)\b",

    # generic consent
    r"\bal\s+continuar\s+navegando\b",
    r"\bconfigura\s+tus\s+preferencias\b",
]

# Compile reusable regexes
RE_BOILER = re.compile("|".join(COMMON_PATTERNS), re.IGNORECASE)

# For snippet filtering (lighter, UI-facing)
SNIPPET_NOISE = re.compile("|".join([
    r"suscripci[oó]n", r"inicia\s+sesi[oó]n", r"aceptar\s+cookies?",
    r"cookies?", r"privacy\s+policy", r"newsletter", r"haz\s+clic",
    r"contin[uú]a\s+leyendo", r"ver\s+m[aá]s",
    r"solo\s+puedes\s+acceder", r"dispositivo", r"publicidad", r"anuncios?",
    r"configura\s+tus\s+preferencias"
]), re.IGNORECASE)
