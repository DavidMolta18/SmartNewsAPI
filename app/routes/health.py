from fastapi import APIRouter

# Router for health-related endpoints
router = APIRouter()

@router.get("/")
def health():
    """
    Health check endpoint.
    Returns a simple JSON object to verify the service is alive.
    """
    return {"ok": True}
