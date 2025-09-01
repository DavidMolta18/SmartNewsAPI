from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# -----------------------------
# /index request/response
# -----------------------------
class IndexParams(BaseModel):
    feed_url: Optional[str] = Field(
        default=None,
        description="Single RSS feed to index. If omitted, use default feeds.json.",
    )
    per_feed_target: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Number of good articles to capture per feed.",
    )
    max_scan: int = Field(
        default=20,
        ge=5,
        le=100,
        description="How many RSS items to scan to find good articles.",
    )
    max_chunks: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max number of chunks per article.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "feed_url": None,
                    "per_feed_target": 3,
                    "max_scan": 6,
                    "max_chunks": 3,
                }
            ]
        }
    }


class IndexResponse(BaseModel):
    indexed_articles: int = Field(..., ge=0, description="Number of articles processed")
    indexed_chunks: int = Field(..., ge=0, description="Total chunks indexed")
    chunk_mode: str = Field(..., description="Chunking strategy used: 'agentic' or 'simple'")


# -----------------------------
# Common article hit payload
# -----------------------------
class ArticleHit(BaseModel):
    article_id: str
    title: str
    url: str
    source: str
    published_at: Optional[str] = Field(default=None)
    score: Optional[float] = Field(
        default=None, description="Score of the best chunk for this article"
    )
    snippets: List[str] = Field(default_factory=list)


# -----------------------------
# /search response
# -----------------------------
class SearchResponse(BaseModel):
    query: str
    results: List[ArticleHit]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "inflation in Colombia",
                    "results": [
                        {
                            "article_id": "a1b2c3",
                            "title": "Inflation falls to 7.5%",
                            "url": "https://example.com/news",
                            "source": "El Tiempo",
                            "published_at": "2025-08-31T10:00:00+00:00",
                            "score": 0.83,
                            "snippets": [
                                "Year-over-year inflation dropped for the fourth consecutive month...",
                                "Food prices showed the largest correction...",
                            ],
                        }
                    ],
                }
            ]
        }
    }
