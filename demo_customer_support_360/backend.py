"""FastAPI backend for Customer 360 Intelligence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from shared.config import settings
from shared.gemini_client import GeminiClient
from demo_customer_support_360.models import (
    PersonaRequest, PersonaResponse, HealthResponse, CustomerProfile, Interaction,
)
from demo_customer_support_360.services import Customer360Service

logging.basicConfig(level=logging.INFO)

_service: Customer360Service | None = None


def get_service() -> Customer360Service:
    global _service
    if _service is None:
        _service = Customer360Service(GeminiClient())
    return _service


app = FastAPI(
    title="Customer 360 Intelligence", version="1.0.0",
    description="RAG-powered customer persona synthesis from interaction history.",
)

# Include MuleSoft stub router
from demo_customer_support_360.mulesoft_stub import router as mulesoft_router
app.include_router(mulesoft_router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        customers_loaded=len(get_service().get_profiles()),
        demo_mode=settings.effective_demo_mode,
    )


@app.post("/api/v1/persona", response_model=PersonaResponse, tags=["Intelligence"])
async def generate_persona(request: PersonaRequest):
    """Generate a comprehensive customer persona using RAG + Gemini."""
    try:
        profile, persona, rag_count = await get_service().generate_persona(request.customer_id)
        return PersonaResponse(
            customer=profile, persona=persona,
            interactions_analyzed=rag_count, rag_sources_used=rag_count,
            demo_mode=settings.effective_demo_mode,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/customers", response_model=list[CustomerProfile], tags=["Data"])
async def list_customers():
    return get_service().get_profiles()


@app.get("/api/v1/customers/{customer_id}/timeline", response_model=list[Interaction], tags=["Data"])
async def get_timeline(customer_id: str):
    interactions = get_service().get_interactions(customer_id)
    return [Interaction(**i) for i in interactions]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=8003)
