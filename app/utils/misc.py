# app/utils/misc.py
import datetime as dt
import time, random
from typing import List
import numpy as np


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
    return ("429" in s) or ("resource exhausted" in s) or ("quota exceeded" in s)


def embed_in_batches(
    texts: List[str],
    embedding_provider,
    batch_size: int = 8,
    max_retries: int = 3,
) -> np.ndarray:
    """
    Embed texts in small batches with retries/backoff to avoid 429s.
    Returns stacked np.ndarray of embeddings.
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
                if attempt > max_retries or not is_429_error(e):
                    raise
                sleep_s = min(1.5 ** attempt + random.uniform(0, 0.3), 6.0)
                print(f"[WARN] Embedding batch retry {attempt} after 429 (~{sleep_s:.2f}s)...")
                time.sleep(sleep_s)

    return np.vstack(arrs)
