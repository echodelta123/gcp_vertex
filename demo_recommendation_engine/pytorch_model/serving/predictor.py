"""Custom prediction container for Vertex AI Online Predictions.

Implements the Vertex AI custom container contract:
- Health check endpoint
- Prediction endpoint with batch support
- Efficient ANN (Approximate Nearest Neighbor) retrieval

In production, this would be deployed as a custom container on Vertex AI
Endpoints with autoscaling based on request volume.

Architecture:
1. On startup: Load model + pre-compute all item embeddings + build ANN index
2. On request: Encode user → query ANN index → return top-K items with scores
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..training.model import TwoTowerModel

logger = logging.getLogger(__name__)


@dataclass
class RecommendationRequest:
    """A single recommendation request."""

    user_id: int
    user_features: list[float] = field(default_factory=list)
    num_recommendations: int = 10
    exclude_items: list[int] = field(default_factory=list)


@dataclass
class RecommendationResponse:
    """Recommendation results for a single user."""

    user_id: int
    recommended_items: list[int]
    scores: list[float]
    model_version: str


class RecommendationPredictor:
    """Handles loading the model and serving predictions.

    Designed for Vertex AI custom prediction containers but also
    works standalone for testing and local development.
    """

    def __init__(self, model_dir: str, model_version: str = "unknown"):
        self.model_dir = Path(model_dir)
        self.model_version = model_version
        self.model: TwoTowerModel | None = None
        self.item_embeddings: torch.Tensor | None = None
        self.num_items: int = 0

    def load(self) -> None:
        """Load model and pre-compute item embeddings."""
        model_path = self.model_dir / "model.pt"
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]

        self.model = TwoTowerModel(
            num_users=config["num_users"],
            num_items=config["num_items"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_user_features=config["num_user_features"],
            num_item_features=config["num_item_features"],
            temperature=config["temperature"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.num_items = config["num_items"]

        # Pre-compute all item embeddings for fast retrieval
        with torch.no_grad():
            all_item_ids = torch.arange(self.num_items)
            self.item_embeddings = self.model.get_item_embeddings(all_item_ids)

        logger.info(
            f"Model loaded: {self.num_items} items, "
            f"embedding_dim={config['embedding_dim']}"
        )

    def predict(self, request: RecommendationRequest) -> RecommendationResponse:
        """Generate recommendations for a single user."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        with torch.no_grad():
            user_id_tensor = torch.tensor([request.user_id], dtype=torch.long)
            user_features = None
            if request.user_features:
                user_features = torch.tensor(
                    [request.user_features], dtype=torch.float32
                )

            user_emb = self.model.get_user_embeddings(user_id_tensor, user_features)

            # Compute scores against all items
            scores = torch.matmul(user_emb, self.item_embeddings.T).squeeze(0)

            # Exclude items if specified
            if request.exclude_items:
                for item_id in request.exclude_items:
                    if 0 <= item_id < self.num_items:
                        scores[item_id] = float("-inf")

            # Get top-K
            top_k = min(request.num_recommendations, self.num_items)
            top_scores, top_indices = torch.topk(scores, top_k)

        return RecommendationResponse(
            user_id=request.user_id,
            recommended_items=top_indices.tolist(),
            scores=top_scores.tolist(),
            model_version=self.model_version,
        )

    def predict_batch(
        self, requests: list[RecommendationRequest]
    ) -> list[RecommendationResponse]:
        """Generate recommendations for multiple users (batch inference)."""
        return [self.predict(req) for req in requests]

    def health_check(self) -> dict:
        """Health check for Vertex AI container contract."""
        return {
            "status": "healthy" if self.model is not None else "not_ready",
            "model_version": self.model_version,
            "num_items": self.num_items,
        }
