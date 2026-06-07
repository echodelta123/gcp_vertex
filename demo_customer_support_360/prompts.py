"""Prompt templates for Customer 360 RAG persona synthesis."""

PERSONA_SYSTEM_INSTRUCTION = """You are a Customer Intelligence Analyst at a major fashion retail brand (H&M Group).
Your job is to synthesize customer interaction data into actionable intelligence.
You analyze purchases, Twitter complaints, email tickets, phone support logs, and surveys to build
a complete customer understanding. Be specific, data-driven, and actionable."""

PERSONA_SYNTHESIS_PROMPT = """Analyze these customer interactions retrieved from our RAG system and generate a comprehensive customer persona.

CUSTOMER: {customer_name} (ID: {customer_id})
SEGMENT: {segment} | TENURE: {tenure_months} months | LTV: ${ltv:.2f}

RETRIEVED INTERACTION HISTORY:
{interactions_text}

Generate a structured customer persona as valid JSON:
{{
  "summary": "<2-3 sentence executive summary of this customer>",
  "segment": "<refined segment based on behavior>",
  "lifetime_value_tier": "HIGH | MEDIUM | LOW",
  "preferences": ["<preference 1>", "<preference 2>", "..."],
  "pain_points": ["<pain point 1>", "<pain point 2>", "..."],
  "sentiment_trend": "IMPROVING | STABLE | DECLINING",
  "churn_risk": "LOW | MEDIUM | HIGH",
  "recommendations": ["<action item 1>", "<action item 2>", "..."],
  "key_interactions": [
    {{"date": "<date>", "type": "<type>", "summary": "<brief summary>"}}
  ]
}}"""


def build_persona_prompt(customer: dict, interactions: list[dict]) -> str:
    interactions_text = "\n".join([
        f"  [{i['date'][:10]}] ({i['type'].upper()}) {i['summary']}"
        for i in interactions
    ])
    return PERSONA_SYNTHESIS_PROMPT.format(
        customer_name=customer.get("name", "Unknown"),
        customer_id=customer.get("id", ""),
        segment=customer.get("segment", "Unknown"),
        tenure_months=customer.get("tenure_months", 0),
        ltv=customer.get("ltv", 0),
        interactions_text=interactions_text,
    )
