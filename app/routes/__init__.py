from fastapi import APIRouter
from . import health, indexing, search, feeds

# Central API router
api_router = APIRouter()

# Mount sub-routers with prefixes + tags
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(indexing.router, prefix="/index", tags=["indexing"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(feeds.router, prefix="/feeds", tags=["feeds"])