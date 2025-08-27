from fastapi import APIRouter
from pydantic import BaseModel, HttpUrl
from typing import Optional
from app.services.embeddings import embed_text
from app.db.repo import upsert_article

router = APIRouter()

class IndexIn(BaseModel):
    title: str
    content: str
    url: Optional[HttpUrl] = None
    source: Optional[str] = None
    published_at: Optional[str] = None  # ISO8601
    lang: Optional[str] = "es"

@router.post("/index")
def index_article(payload: IndexIn):
    text = f"{payload.title}\n\n{payload.content}"
    vec = embed_text(text)
    article_id = upsert_article(
        title=payload.title,
        content=payload.content,
        url=str(payload.url) if payload.url else None,
        source=payload.source,
        published_at=payload.published_at,
        lang=payload.lang,
        embedding=vec,
    )
    return {"status": "indexed", "id": article_id}
