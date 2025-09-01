# app/utils/text_utils.py
import re
from app.ingestion.quality import clean_text
from app.utils.patterns import SNIPPET_NOISE  # centralized snippet noise patterns

# =============================================================================
# Helpers for extracting clean snippets for UI/preview display.
# =============================================================================

def first_clean_sentence(text: str) -> str:
    """
    Extract the first clean sentence-like span to use as a preview/snippet.

    Rules:
      - Skip common boilerplate (ads, cookie banners, login prompts).
      - Must be at least 40 characters (avoid uselessly short spans).
      - Cap at 300 characters (avoid flooding the UI).
      - Fall back to the cleaned text (trimmed) if no suitable snippet is found.

    Args:
        text (str): Raw or cleaned text to extract snippet from.

    Returns:
        str: First suitable sentence for preview display.
    """
    if not text:
        return ""

    soft = clean_text(text)
    parts = re.split(r'(?<=[\.\?!])\s+', soft)

    for s in parts:
        s = s.strip()
        if len(s) >= 40 and not SNIPPET_NOISE.search(s):
            return s[:300]

    return soft.strip()[:300]
