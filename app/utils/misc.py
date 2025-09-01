import datetime as dt
import time, random
from typing import List
import numpy as np
from fastapi import HTTPException  

def to_epoch(ts_iso: str | None) -> int | None:
    """Convert ISO8601 timestamp (str) to epoch seconds."""
    if not ts_iso:
        return None
    try:
        return int(dt.datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def is_429_error(err: Exception) -> bool:
    """Detect quota/capacity errors (HTTP 429-like) without SDK-specific imports."""
    s = str(err).lower()
    return (
        "429" in s
        or "resource exhausted" in s
        or "quota exceeded" in s
        or "exceeded for aiplatform" in s  
        or "too many requests" in s         


def embed_in_batches(
    texts: List[str],
    embedding_provider,
    batch_size: int = 10,
    max_retries: int = 5,
) -> np.ndarray:
    """
    Embed texts in small batches with retries/backoff to avoid 429s.
    Returns stacked np.ndarray of embeddings.
    If quota is fully exhausted, raise 503 instead of crashing with 500.
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

                # Not a 429-like error → propagate original error (FastAPI will return 500)
                if not is_429_error(e):
                    raise

                # Quota/rate limit exceeded after max retries → return 503
                if attempt > max_retries:
                    raise HTTPException(
                        status_code=503,
                        detail="Embedding quota exhausted or rate limit hit. Try again later."
                    )

                # Retry with exponential backoff + jitter
                sleep_s = min(1.5 ** attempt + random.uniform(0, 0.3), 6.0)
                print(f"[WARN] Embedding batch retry {attempt} after 429 (~{sleep_s:.2f}s)...")
                time.sleep(sleep_s)

    return np.vstack(arrs)
