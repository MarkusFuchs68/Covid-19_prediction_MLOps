from fastapi import FastAPI
from app.models import router as models_router

app = FastAPI()

app.include_router(models_router, prefix="/api/models", tags=["models"])


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}
