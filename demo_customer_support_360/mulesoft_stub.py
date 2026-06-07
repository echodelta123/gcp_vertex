from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from shared.gemini_client import GeminiClient
from demo_customer_support_360.services import Customer360Service

router = APIRouter(prefix="/mulesoft", tags=["MuleSoft Integration"])

class MuleRequest(BaseModel):
    """Payload that a MuleSoft flow would send."""
    customer_id: str
    include_history: bool = True
    language: str = "en"

class MuleResponse(BaseModel):
    """Minimal response MuleSoft would consume."""
    persona_summary: str
    churn_risk: str
    next_best_action: list[str]

# Create a service instance (same as backend singleton but separate for stub)
_service = Customer360Service(GeminiClient())

@router.post("/persona", response_model=MuleResponse)
async def mule_persona(req: MuleRequest):
    try:
        profile, persona, _ = await _service.generate_persona(req.customer_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if profile is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return MuleResponse(
        persona_summary=persona.summary,
        churn_risk=persona.churn_risk,
        next_best_action=persona.recommendations,
    )
