"""Recommender Service – thin façade delegating to VectorSearch and Explainability modules.

RecommenderService orchestrates vector search and Gemini explanations.
It now composes two dedicated modules for better separation of concerns.
"""

import logging
from shared.gemini_client import GeminiClient
from shared.config import settings
from demo_recommendation_engine.models import RecommendationResult, Product
from demo_recommendation_engine.vector_search import VectorSearch
from demo_recommendation_engine.explainability import Explainability
from demo_recommendation_engine.pytorch_client import LocalTwoTowerClient

logger = logging.getLogger(__name__)

class RecommenderService:
    def __init__(self, gemini: GeminiClient):
        self._gemini = gemini
        self._vector = VectorSearch(gemini)
        self._explain = Explainability(gemini)
        self.pytorch_client = LocalTwoTowerClient()

    def get_pytorch_recommendations(
        self, user_id: int, user_features: list[float] = None, top_k: int = 5
    ) -> dict:
        """Fetch recommendations using the local Two-Tower PyTorch model."""
        return self.pytorch_client.predict(user_id, user_features, top_k)

    def train_pytorch_model(self, epochs: int = 5) -> dict:
        """Run local training on synthetic data for the Two-Tower model."""
        return self.pytorch_client.train(epochs)

    def ingest_catalog(self) -> int:
        # Ingest using VectorSearch which handles mock fallback internally
        return self._vector.ingest_catalog()

    async def recommend(self, query: str, top_k: int = 5) -> list[RecommendationResult]:
        # 1. Semantic search (or fallback) to get candidate products
        matched_products = self._vector.search(query, top_k)
        if not matched_products:
            return []
        # 2. Generate explanations via Explainability module
        explanations = await self._explain.generate_explanations(query, matched_products)
        # 3. Assemble results
        results = []
        for product_dict in matched_products:
            pid = product_dict["product_id"]
            expl = explanations.get(pid, {})
            results.append(
                RecommendationResult(
                    product=Product(**product_dict),
                    explanation=expl.get("explanation", f"Matches your interest in '{query}'."),
                    match_reasons=expl.get("match_reasons", [product_dict.get("category", ""), *product_dict.get("tags", [])[:2]]),
                )
            )
        return results

    def get_catalog(self) -> list[dict]:
        return self._vector.get_catalog()

    def get_catalog_size(self) -> int:
        return len(self._vector.get_catalog())
