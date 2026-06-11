"""FastAPI application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.app.api import router as predict_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AgriMap API",
    description="CNN-ViT satellite land geospatial classification API",
    version="0.1.0",
)

# CORS — allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include prediction routes
app.include_router(predict_router)


@app.get("/health", response_model=dict)
async def health_check() -> dict:
    """Health check endpoint — returns GPU and model status."""
    import torch

    from api.app.services import ModelService

    try:
        model_svc = ModelService()
        model_loaded = model_svc.model is not None
    except Exception:
        model_loaded = False

    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "model_loaded": model_loaded,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.on_event("startup")
async def startup_event() -> None:
    """Warm up the model on startup."""
    logger.info("🚀 AgriMap API starting up...")
    from api.app.services import ModelService

    try:
        ModelService()
        logger.info("✅ Model loaded and GPU warmed up")
    except Exception as exc:
        logger.error("❌ Model loading failed: %s", exc)
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
