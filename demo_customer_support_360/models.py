"""Pydantic models for Customer 360 Intelligence."""
from pydantic import BaseModel, Field


class CustomerProfile(BaseModel):
    id: str
    name: str
    segment: str
    ltv: float = 0.0
    tenure_months: int = 0


class Interaction(BaseModel):
    interaction_id: str
    customer_id: str
    type: str
    summary: str
    date: str
    agent: str | None = None
    satisfaction_score: int | None = None


class PersonaRequest(BaseModel):
    customer_id: str = Field(..., description="Customer ID to generate persona for")
    model_config = {"json_schema_extra": {"examples": [{"customer_id": "CUST-1001"}]}}


class PersonaInsight(BaseModel):
    summary: str = ""
    segment: str = ""
    lifetime_value_tier: str = ""
    preferences: list[str] = []
    pain_points: list[str] = []
    sentiment_trend: str = "STABLE"
    churn_risk: str = "LOW"
    recommendations: list[str] = []
    key_interactions: list[dict] = []


class PersonaResponse(BaseModel):
    success: bool = True
    customer: CustomerProfile | None = None
    persona: PersonaInsight | None = None
    interactions_analyzed: int = 0
    rag_sources_used: int = 0
    demo_mode: bool = False


class IngestResponse(BaseModel):
    success: bool = True
    interactions_ingested: int = 0
    chunks_created: int = 0
    message: str = ""


class HealthResponse(BaseModel):
    status: str = "healthy"
    demo: str = "Customer 360 Intelligence"
    version: str = "1.0.0"
    customers_loaded: int = 0
    rag_store_status: str = "ready"
    demo_mode: bool = False
