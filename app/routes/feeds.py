from fastapi import APIRouter
from app.ingestion.rss import load_default_feeds

router = APIRouter()

@router.get("/")
def list_feeds():
    """
    Return the list of default RSS feeds configured in feeds.json.
    Useful for debugging and ensuring the system is reading feeds correctly.
    """
    return {"feeds": load_default_feeds()}
