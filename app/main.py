from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from app.api.routes import router

app = FastAPI(title="News Intelligence API", version="0.1.0")
app.include_router(router, prefix="/api/v1")


if not getattr(app.state, "metrics_instrumented", False):
    Instrumentator().instrument(app).expose(app)
    app.state.metrics_instrumented = True

@app.get("/health")
def health():
    return {"status": "ok"}
