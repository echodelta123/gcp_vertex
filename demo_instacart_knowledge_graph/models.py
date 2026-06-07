"""Pydantic models for Graph Knowledge Explorer."""
from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    properties: dict = {}


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    confidence: float = 0.0


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500,
                       description="Natural language question about the product graph")
    model_config = {"json_schema_extra": {"examples": [
        {"query": "What products are frequently bought with Organic Bananas?"}
    ]}}


class QueryResponse(BaseModel):
    success: bool = True
    natural_language_query: str = ""
    generated_cypher: str = ""
    explanation: str = ""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    raw_results: list[dict] = []
    demo_mode: bool = False


class GraphStats(BaseModel):
    total_products: int = 0
    total_categories: int = 0
    total_aisles: int = 0
    total_relationships: int = 0
    relationship_types: list[str] = []


class HealthResponse(BaseModel):
    status: str = "healthy"
    demo: str = "Knowledge Graph Explorer"
    version: str = "1.0.0"
    graph_status: str = "ready"
    demo_mode: bool = False
