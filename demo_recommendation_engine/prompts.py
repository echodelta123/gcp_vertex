"""
Prompt templates for the Recommendation Engine.
"""

RECOMMENDATION_SYSTEM_INSTRUCTION = """You are an AI product recommendation specialist.
Given a user's query and a list of semantically similar products from our catalog,
explain why each product is a good match. Be specific about feature matches,
use cases, and value propositions. Keep explanations concise (2-3 sentences each)."""

RECOMMENDATION_EXPLANATION_PROMPT = """A customer said: "{query}"

Based on semantic search, these products were found as top matches:
{products_json}

For each product, provide a brief, specific explanation of why it matches the customer's needs.
Return valid JSON array:
[
  {{
    "product_id": "<id>",
    "explanation": "<2-3 sentence explanation of why this matches>",
    "match_reasons": ["<reason1>", "<reason2>"]
  }}
]"""


def build_explanation_prompt(query: str, products: list[dict]) -> str:
    import json
    products_summary = json.dumps([
        {"product_id": p["product_id"], "name": p["name"],
         "description": p["description"], "category": p["category"]}
        for p in products
    ], indent=2)
    return RECOMMENDATION_EXPLANATION_PROMPT.format(
        query=query, products_json=products_summary
    )
