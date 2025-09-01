from __future__ import annotations
import json, re, time, random
from typing import List
from app.config import settings

_VERTEX_READY = {"ok": False}


def split_text_by_chars(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """
    Simple sliding window chunker by character length.
    This is the "safe" fallback: cheap, deterministic, no external calls.
    """
    text = text or ""
    max_chars = max(500, max_chars)
    overlap = max(0, min(overlap, max_chars // 2))
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        out.append(text[i:j])
        if j == n:
            break
        i = j - overlap
    return out


def _ensure_vertexai_initialized():
    """
    Initialize Vertex AI only if needed (lazy).
    Using a mutable dict avoids 'global' assignment issues.
    """
    if _VERTEX_READY["ok"]:
        return
    try:
        from vertexai import init as vertexai_init
        project = settings.gcp_project
        location = settings.gcp_location or "us-central1"
        if not project:
            raise RuntimeError("Missing GCP project for Vertex AI (settings.gcp_project).")
        vertexai_init(project=project, location=location)
        _VERTEX_READY["ok"] = True
    except Exception as e:
        print("WARN Vertex AI init failed (agentic chunking may not work):", e)


def agentic_chunk(
    text: str,
    max_chunks: int = 5,
    fallback_simple: bool = True,
    simple_max_chars: int = 2000,
    simple_overlap: int = 200,
    retries: int = 2
) -> List[str]:
    """
    Use Gemini to produce semantic chunks. If it fails (quota/JSON/etc),
    fallback to the simple char-based splitter with overlap.
    Includes JSON sanitization and retry/backoff for transient 429s.
    """
    try:
        _ensure_vertexai_initialized()
        from vertexai.generative_models import GenerativeModel

        # Hard cap input
        text_lite = (text or "")[:8000]

        system = (
            "You are a JSON-only machine. "
            "You must segment long-form news articles into coherent contiguous chunks. "
            f"Return at most {max_chunks} chunks, each around 400–600 tokens. "
            "Remove cookie banners, subscription notices, ads, and social widgets. "
            "IMPORTANT: Output must be a single valid JSON object, no explanations, no markdown, no code fences."
        )

        user = (
            "Produce exactly this JSON structure:\n"
            '{ "chunks": ["chunk_1", "chunk_2", "..."] }\n\n'
            "Rules:\n"
            " - Output only one JSON object.\n"
            " - Do not include any commentary.\n"
            " - Do not use markdown fences.\n"
            " - Do not truncate chunks mid-sentence if possible.\n\n"
            f"Article:\n{text_lite}"
        )

        model = GenerativeModel("gemini-2.5-flash-lite")

        attempt = 0
        while True:
            attempt += 1
            try:
                resp = model.generate_content([system, user])
                raw = (resp.text or "").strip()
                print("RAW Gemini response >>>", raw[:500])

                # Sanitize JSON output
                cleaned = "\n".join(line for line in raw.splitlines() if not line.strip().startswith("```"))
                start, end = cleaned.find("{"), cleaned.rfind("}")
                candidate = cleaned[start:end + 1] if start != -1 and end != -1 else cleaned

                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as je:
                    print("Gemini returned invalid JSON:", je)
                    print("Raw preview >>>", raw[:200])
                    parsed = {"chunks": []}

                chunks = parsed.get("chunks", [])
                chunks = [c.strip() for c in chunks if isinstance(c, str) and len(c.strip()) >= 80]

                if not chunks:
                    raise ValueError("Empty/invalid chunks from LLM")

                return chunks[:max_chunks]

            except Exception as e:
                if attempt <= retries and ("429" in str(e).lower() or "resource exhausted" in str(e).lower()):
                    backoff = min(1.5 ** attempt + random.uniform(0, 0.3), 6.0)
                    print(f"Agentic 429, retry {attempt} in ~{backoff:.2f}s...")
                    time.sleep(backoff)
                    continue
                raise

    except Exception as e:
        print("Agentic chunking failed -> falling back to simple splitter:", e)
        if fallback_simple:
            return split_text_by_chars(text, max_chars=simple_max_chars, overlap=simple_overlap)[:max_chunks]
        return [text]
