# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import api_router

# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------
app = FastAPI(title="News Intelligence API")

# -----------------------------------------------------------------------------
# Global CORS config
# (Relaxed for demo; restrict origins in production)
# -----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# -----------------------------------------------------------------------------
# Register API routers
# -----------------------------------------------------------------------------
app.include_router(api_router)
