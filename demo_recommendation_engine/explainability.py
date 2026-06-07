"""Explainability module for the recommendation engine demo.

This module isolates the Gemini‑based explanation logic that was previously
embedded in `services.py`.  It provides a thin async interface that can be
called from `RecommenderService`.
"""

import logging
from typing import List, Dict

from shared.gemini_client import GeminiClient
from shared.config import settings

logger = logging.getLogger(__name__)


class Explainability:
    """Generate human‑readable explanations for a list of products.

    Parameters
    ----------
    gemini: GeminiClient
        The shared Gemini client used for LLM calls.  It automatically
        falls back to mock explanations when ``DEMO_MODE`` is true.
    """

    def __init__(self, gemini: GeminiClient) -> None:
        self._gemini = gemini

    async def generate_explanations(
        self, query: str, products: List[Dict]
    ) -> Dict[str, Dict]:
        """Return a mapping of ``product_id`` → explanation data.

        The method builds a prompt using the templates in
        ``demo_recommendation_engine/prompts.py`` and calls the Gemini API.
        In demo mode a deterministic mock is returned.
        """
        if not products:
            return {}

        # Build the prompt – we reuse the existing prompt template.
        from .prompts import build_explanation_prompt, RECOMMENDATION_SYSTEM_INSTRUCTION
        prompt = build_explanation_prompt(query, products)

        try:
            # Call the sync generate_json on GeminiClient
            data = self._gemini.generate_json(
                prompt, system_instruction=RECOMMENDATION_SYSTEM_INSTRUCTION
            )
            # Convert list response to mapping of product_id -> data
            if isinstance(data, list):
                return {item["product_id"]: item for item in data if "product_id" in item}
            return data
        except Exception as exc:  # pragma: no cover – fallback is exercised in tests
            logger.exception("Failed to generate explanations, falling back to mock.")
            # Simple deterministic mock based on product name.
            mock = {}
            for p in products:
                pid = p["product_id"]
                mock[pid] = {
                    "explanation": f"Matches your interest in '{query}'.",
                    "match_reasons": [p.get("category", ""), *p.get("tags", [])[:2]],
                }
            return mock
