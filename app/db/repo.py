import uuid
from typing import Optional
from sqlalchemy import text
from app.db.engine import SessionLocal

# Inserta o actualiza por URL y retorna el id (UUID) del artículo
def upsert_article(
    *,
    title: str,
    content: str,
    embedding: list[float],
    url: Optional[str] = None,
    source: Optional[str] = None,
    published_at: Optional[str] = None,  # ISO8601 o None
    lang: Optional[str] = "es",
) -> str:
    new_id = str(uuid.uuid4())

    sql = text("""
        INSERT INTO articles (id, title, content, url, source, published_at, lang, embedding)
        VALUES (:id, :title, :content, :url, :source, :published_at, :lang, :embedding)
        ON CONFLICT (url) DO UPDATE
        SET title = EXCLUDED.title,
            content = EXCLUDED.content,
            source = EXCLUDED.source,
            published_at = EXCLUDED.published_at,
            lang = EXCLUDED.lang,
            embedding = EXCLUDED.embedding
        RETURNING id;
    """)

    params = {
        "id": new_id,
        "title": title,
        "content": content,
        "url": url,
        "source": source,
        "published_at": published_at,
        "lang": lang,
        "embedding": embedding,
    }

    with SessionLocal() as s:
        row = s.execute(sql, params).first()
        s.commit()
        return str(row[0])  # UUID (el de inserción o el existente si fue update)
