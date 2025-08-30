# app/ingestion/quality.py
from __future__ import annotations
import re

# Patrones típicos de basura/boilerplate en medios
_BOILERPLATE_PATTERNS = [
    r"este sitio usa cookies", r"pol[ií]tica de cookies", r"aceptar cookies",
    r"suscr[ií]bete", r"reg[ií]strate", r"inicia sesi[oó]n", r"boletines? el tiempo",
    r"newsletter", r"publicidad", r"anuncios", r"cookie ?policy", r"sign up",
    r"subscribe", r"accept (all )?cookies", r"privacy policy",
]

_RE_BOILER = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)

def clean_text(txt: str) -> str:
    if not txt:
        return ""
    # Normalizaciones básicas
    txt = txt.replace("\r", "\n")
    # Quitar espacios repetidos y líneas super cortas de navegación
    lines = [l.strip() for l in txt.split("\n")]
    # descartar líneas de navegación/menu típicas
    lines = [l for l in lines if len(l) > 2 and not re.match(r"^(menu|home|inicio|buscar|search)$", l, re.I)]
    # colapsar múltiples saltos
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return cleaned

def is_boilerplate(txt: str) -> bool:
    if not txt:
        return True
    # Si hay muchos matches de boilerplate, marcar como basura
    matches = _RE_BOILER.findall(txt)
    return len(matches) >= 2

def quality_score(txt: str) -> float:
    """Heurística simple: longitud + señales de prosa."""
    if not txt:
        return 0.0
    n = len(txt)
    # señales de texto “real”: puntos, párrafos
    bonus = 0
    if "." in txt or "。" in txt: bonus += 300
    if "\n\n" in txt: bonus += 300
    # penaliza texto con demasiados enlaces
    links = len(re.findall(r"https?://|www\.", txt))
    penalty = 150 * links
    return max(0, n + bonus - penalty)

def should_index(txt: str, min_chars: int = 400, min_score: float = 700.0) -> tuple[bool, str]:
    """Decide si vale la pena indexar y explica por qué."""
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
