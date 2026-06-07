"""
PyTorch Client for the local Two-Tower Recommendation Engine.
Handles model training, loading, dynamic quantization tracking, and CPU inference.
"""
import logging
from pathlib import Path
import torch

from demo_recommendation_engine.pytorch_model.training.train import run_demo
from demo_recommendation_engine.pytorch_model.serving.predictor import (
    RecommendationPredictor,
    RecommendationRequest,
)

logger = logging.getLogger(__name__)


class LocalTwoTowerClient:
    """Client for local Two-Tower recommendation inference and training."""

    def __init__(self):
        self.output_dir = Path(__file__).resolve().parent / "pytorch_model" / "output"
        self.predictor = None
        self.is_ready = False
        self._load_predictor()

    def _load_predictor(self):
        """Loads the saved PyTorch model checkpoint and instantiates the predictor."""
        try:
            model_path = self.output_dir / "model.pt"
            if not model_path.exists():
                logger.warning(
                    f"⚠️ Two-Tower recommendation model not found at {model_path}. "
                    "Train the model locally to initialize it."
                )
                self.is_ready = False
                return

            self.predictor = RecommendationPredictor(model_dir=str(self.output_dir))
            self.predictor.load()
            self.is_ready = True
            logger.info("⚡ Local Two-Tower recommendation model loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load local PyTorch recommendation model: {e}")
            self.is_ready = False

    def predict(
        self, user_id: int, user_features: list[float] = None, num_recommendations: int = 5
    ) -> dict:
        """Run CPU inference on user inputs and return top recommended item IDs and scores."""
        if not self.is_ready or self.predictor is None:
            return {
                "user_id": user_id,
                "recommended_items": [],
                "scores": [],
                "error": "Model not loaded. Please train the model first.",
            }

        req = RecommendationRequest(
            user_id=user_id,
            user_features=user_features or [],
            num_recommendations=num_recommendations,
        )
        try:
            resp = self.predictor.predict(req)
            return {
                "user_id": resp.user_id,
                "recommended_items": resp.recommended_items,
                "scores": resp.scores,
                "model_version": resp.model_version,
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "user_id": user_id,
                "recommended_items": [],
                "scores": [],
                "error": str(e),
            }

    def train(self, epochs: int = 5) -> dict:
        """Runs the training loop on synthetic e-commerce data and reloads the model."""
        logger.info(f"Starting local training for {epochs} epochs...")
        try:
            # run_demo trains model, runs quantization, saves outputs, and returns metrics
            metrics = run_demo(output_dir=str(self.output_dir), epochs=epochs)

            # Reload predictor with newly trained weights
            self._load_predictor()
            return {"success": True, "metrics": metrics}
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"success": False, "error": str(e)}
