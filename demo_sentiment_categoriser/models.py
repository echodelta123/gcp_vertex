"""
Pydantic models for the Sentiment Intelligence Engine.
Defines request/response schemas with validation and OpenAPI docs.
"""
from pydantic import BaseModel, Field
from enum import Enum


class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"


class UrgencyLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# --- Request Models ---

class AnalyzeRequest(BaseModel):
    """Single text analysis request."""
    text: str = Field(..., min_length=5, max_length=5000, description="Text to analyze")
    context: str | None = Field(None, description="Optional business context (e.g., 'product review', 'support ticket')")
    backend: str | None = Field(None, description="Optional backend override (e.g., 'GEMINI', 'OLLAMA', 'LOCAL_PYTORCH', 'HEURISTIC')")

    model_config = {"json_schema_extra": {"examples": [
        {"text": "The product quality is amazing but shipping took forever.", "context": "product review", "backend": "LOCAL_PYTORCH"}
    ]}}


class BatchAnalyzeRequest(BaseModel):
    """Batch analysis request with multiple texts."""
    items: list[AnalyzeRequest] = Field(..., min_length=1, max_length=100)


# --- Response Models ---

class AspectSentiment(BaseModel):
    aspect: str = Field(..., description="Aspect being evaluated (e.g., 'product quality', 'shipping')")
    sentiment: SentimentLabel
    score: float = Field(..., ge=0.0, le=1.0)

# IAB‑Tech compatible aspect representation (flat, includes per‑aspect confidence)
class IABAspectSentiment(BaseModel):
    aspect: str = Field(..., description="Aspect name (e.g., 'design', 'durability')")
    sentiment: SentimentLabel
    score: float = Field(..., ge=0.0, le=1.0, description="Aspect‑level confidence score (0‑1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Duplicate of score for explicit IAB compliance")


class Entity(BaseModel):
    name: str
    type: str = Field(..., description="Entity type: PRODUCT, PERSON, ORG, FEATURE, ISSUE")


class SentimentResult(BaseModel):
    sentiment: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    aspects: list[AspectSentiment] = []
    iab_aspects: list[IABAspectSentiment] = []
    entities: list[Entity] = []
    key_phrases: list[str] = []
    urgency: UrgencyLevel = UrgencyLevel.LOW
    summary: str = ""
    source_text: str = ""


class AnalyzeResponse(BaseModel):
    success: bool = True
    result: SentimentResult
    model_used: str = ""
    demo_mode: bool = False


class BatchAnalyzeResponse(BaseModel):
    success: bool = True
    results: list[SentimentResult] = []
    total_processed: int = 0
    aggregate: dict = Field(default_factory=dict, description="Aggregate statistics")
    demo_mode: bool = False


class TrendDataPoint(BaseModel):
    date: str
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    mixed: int = 0
    avg_score: float = 0.0


class HealthResponse(BaseModel):
    status: str = "healthy"
    demo: str = "Sentiment Intelligence Engine"
    version: str = "1.0.0"
    gemini_status: str = "connected"
    pytorch_status: str = "not_loaded"
    demo_mode: bool = False
