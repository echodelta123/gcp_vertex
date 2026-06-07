"""
FastAPI backend for the Semantic Recommendation Engine.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from shared.config import settings
from shared.gemini_client import GeminiClient
from shared.mock_data import generate_product_catalog
from demo_recommendation_engine.models import (
    RecommendRequest, RecommendResponse, IngestResponse,
    HealthResponse, Product,
)
from demo_recommendation_engine.services import RecommenderService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_service: RecommenderService | None = None


def get_service() -> RecommenderService:
    global _service
    if _service is None:
        gemini = GeminiClient()
        _service = RecommenderService(gemini)
        _service.ingest_catalog()
    return _service


app = FastAPI(
    title="Semantic Recommendation Engine",
    description="AI-powered product recommendations using vector search and Gemini explanations.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health():
    svc = get_service()
    return HealthResponse(
        catalog_size=svc.get_catalog_size(),
        demo_mode=settings.effective_demo_mode,
    )


@app.post("/api/v1/recommend", response_model=RecommendResponse, tags=["Recommendations"])
async def recommend(request: RecommendRequest):
    """Get personalized product recommendations based on natural language query."""
    svc = get_service()
    try:
        results = await svc.recommend(request.query, request.top_k)
        return RecommendResponse(
            query=request.query,
            recommendations=results,
            total_catalog_size=svc.get_catalog_size(),
            demo_mode=settings.effective_demo_mode,
        )
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["Data"])
async def ingest_catalog():
    """Re-ingest the product catalog into the vector store."""
    svc = get_service()
    count = svc.ingest_catalog()
    return IngestResponse(products_ingested=count, embeddings_created=count,
                          message=f"Successfully ingested {count} products")


@app.get("/api/v1/catalog", response_model=list[Product], tags=["Data"])
async def get_catalog():
    """List all products in the catalog."""
    return [Product(**p) for p in get_service().get_catalog()]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=8002)
