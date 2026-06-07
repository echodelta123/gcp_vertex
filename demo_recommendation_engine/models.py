"""
Pydantic models for the Recommendation Engine.
"""
from pydantic import BaseModel, Field


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    description: str
    tags: list[str] = []
    similarity_score: float | None = None


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Natural language preference or product description")
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations to return")

    model_config = {"json_schema_extra": {"examples": [
        {"query": "I need lightweight running shoes for daily training", "top_k": 5}
    ]}}


class SimilarRequest(BaseModel):
    product_id: str = Field(..., description="Product ID to find similar items for")
    top_k: int = Field(5, ge=1, le=20)


class RecommendationResult(BaseModel):
    product: Product
    explanation: str = ""
    match_reasons: list[str] = []


class RecommendResponse(BaseModel):
    success: bool = True
    query: str = ""
    recommendations: list[RecommendationResult] = []
    total_catalog_size: int = 0
    demo_mode: bool = False


class IngestResponse(BaseModel):
    success: bool = True
    products_ingested: int = 0
    embeddings_created: int = 0
    message: str = ""


class HealthResponse(BaseModel):
    status: str = "healthy"
    demo: str = "Semantic Recommendation Engine"
    version: str = "1.0.0"
    catalog_size: int = 0
    vector_store_status: str = "ready"
    demo_mode: bool = False
