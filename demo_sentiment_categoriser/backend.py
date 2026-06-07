"""
FastAPI backend for the Sentiment Intelligence Engine.

Endpoints:
  POST /api/v1/analyze        – Analyze a single text
  POST /api/v1/analyze/batch  – Analyze multiple texts (CSV upload or JSON)
  GET  /api/v1/history        – Get recent analysis history
  GET  /api/v1/health         – Health check with dependency status

Architecture:
  Request → FastAPI → SentimentService → GeminiClient → Structured Response
                                       ↘ Mock fallback (DEMO_MODE=true)
"""
import sys
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is on path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from shared.config import settings
from shared.gemini_client import GeminiClient
from demo_sentiment_categoriser.models import (
    AnalyzeRequest, AnalyzeResponse, BatchAnalyzeResponse,
    HealthResponse, SentimentResult,
)
from demo_sentiment_categoriser.services import SentimentService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Lazy-init service (works with both lifespan and TestClient) ---

_gemini_client: GeminiClient | None = None
_sentiment_service: SentimentService | None = None


def get_gemini() -> GeminiClient:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


def get_service() -> SentimentService:
    global _sentiment_service
    if _sentiment_service is None:
        _sentiment_service = SentimentService(get_gemini())
    return _sentiment_service


# --- FastAPI App ---

app = FastAPI(
    title="Sentiment Intelligence Engine",
    description=(
        "Enterprise-grade sentiment analysis powered by Google Gemini. "
        "Analyzes customer feedback with aspect-based sentiment, entity extraction, "
        "urgency scoring, and batch processing capabilities."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and dependency status."""
    client = get_gemini()
    svc = get_service()
    return HealthResponse(
        gemini_status="live" if client.is_live else "demo_mode",
        pytorch_status="ready" if svc.pytorch_client.is_ready else "not_loaded",
        demo_mode=settings.effective_demo_mode,
    )


@app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze a single text for sentiment, aspects, entities, and urgency.

    Supports dynamic selection of backends: GEMINI, OLLAMA, LOCAL_PYTORCH, or HEURISTIC.
    """
    svc = get_service()
    client = get_gemini()
    try:
        result = await svc.analyze(request.text, request.context, backend=request.backend)
        
        active_backend = (request.backend or settings.MODEL_BACKEND).upper()
        if active_backend == "LOCAL_PYTORCH":
            model_used = "local-pytorch"
        elif active_backend == "HEURISTIC":
            model_used = "heuristic-mock"
        elif active_backend == "OLLAMA":
            model_used = settings.OLLAMA_MODEL
        else:
            model_used = settings.GEMINI_MODEL if client.is_live else "heuristic-mock"

        return AnalyzeResponse(
            result=result,
            model_used=model_used,
            demo_mode=settings.effective_demo_mode,
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze/batch", response_model=BatchAnalyzeResponse, tags=["Analysis"])
async def analyze_batch(items: list[AnalyzeRequest]):
    """
    Analyze multiple texts in a single request.
    Returns individual results plus aggregate statistics.
    """
    svc = get_service()
    try:
        texts = [{"text": item.text, "context": item.context} for item in items]
        backend_override = items[0].backend if items else None
        results = await svc.analyze_batch(texts, backend=backend_override)
        aggregate = svc.get_aggregate_stats(results)
        return BatchAnalyzeResponse(
            results=results,
            total_processed=len(results),
            aggregate=aggregate,
            demo_mode=settings.effective_demo_mode,
        )
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/history", response_model=list[SentimentResult], tags=["Analysis"])
async def get_history():
    """Get recent analysis history (last 50 results, in-memory)."""
    return get_service().get_history()


# --- Run ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=8001)
