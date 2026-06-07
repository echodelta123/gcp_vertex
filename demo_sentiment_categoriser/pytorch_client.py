import json
import logging
import torch
from pathlib import Path
from typing import Any

from demo_sentiment_categoriser.pytorch_model.model import SentimentAspectClassifier, encode_batch

logger = logging.getLogger(__name__)


class LocalPyTorchClient:
    """Client for local sentiment and aspect classification using a quantized PyTorch model."""

    def __init__(self):
        self.is_ready = False
        self.vocab = {}
        self.labels = []
        self.model = None

        self.output_dir = Path(__file__).resolve().parent / "pytorch_model" / "output"
        self.model_path = self.output_dir / "sentiment_model_quantized.pt"
        self.vocab_path = self.output_dir / "vocab.json"
        self.labels_path = self.output_dir / "labels.json"

        self._load_model()

    def _load_model(self):
        try:
            if not (self.model_path.exists() and self.vocab_path.exists() and self.labels_path.exists()):
                logger.warning(
                    f"⚠️ PyTorch model files not found at {self.output_dir}. "
                    "Run 'python -m pytorch_model.model' to train and generate the quantized weights."
                )
                return

            with open(self.vocab_path, "r") as f:
                self.vocab = json.load(f)
            with open(self.labels_path, "r") as f:
                self.labels = json.load(f)

            vocab_size = len(self.vocab)
            num_labels = len(self.labels)

            # Reconstruct float model
            float_model = SentimentAspectClassifier(vocab_size=vocab_size, num_labels=num_labels)
            
            # Apply dynamic quantization to match saved weights
            self.model = torch.ao.quantization.quantize_dynamic(
                float_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
            )
            
            # Load weights
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device("cpu")))
            self.model.eval()
            self.is_ready = True
            logger.info("⚡ Local PyTorch quantized model loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load local PyTorch model: {e}")

    @torch.inference_mode()
    def predict(self, text: str, threshold: float = 0.35) -> dict[str, Any]:
        """Run inference on the input text and return structured predictions."""
        if not self.is_ready:
            logger.warning("PyTorch model is not loaded. Returning fallback prediction.")
            return self._fallback_prediction(text)

        # 1. Encode text
        indices, offsets = encode_batch([text], self.vocab)
        if indices.numel() == 0:
            # Empty input or all words OOV
            return self._fallback_prediction(text)

        # 2. Run model
        logits = self.model(indices, offsets)
        probs = torch.sigmoid(logits)[0]

        # 3. Map probabilities to labels
        predictions = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}

        # 4. Extract overall sentiment
        sentiment_keys = ["sentiment:positive", "sentiment:negative", "sentiment:mixed"]
        sent_probs = {k: predictions.get(k, 0.0) for k in sentiment_keys}
        
        # Determine dominant sentiment
        best_sent_key = max(sent_probs, key=sent_probs.get)
        best_prob = sent_probs[best_sent_key]

        if best_prob < threshold:
            sentiment = "NEUTRAL"
            confidence = float(1.0 - best_prob)
        else:
            sentiment = best_sent_key.split(":")[-1].upper()
            confidence = float(best_prob)

        # 5. Extract aspects
        aspects = []
        for label, prob in predictions.items():
            if label.startswith("aspect:") and prob >= threshold:
                aspect_name = label.split(":")[-1].replace("_", " ")
                aspects.append({
                    "aspect": aspect_name,
                    "sentiment": sentiment,  # Match overall sentiment since model classifies overall
                    "score": round(prob, 2),
                })

        # Default aspect if none detected
        if not aspects:
            aspects.append({
                "aspect": "overall experience",
                "sentiment": sentiment,
                "score": round(confidence, 2),
            })

        # 6. Urgency heuristic based on negative sentiment prediction
        neg_prob = predictions.get("sentiment:negative", 0.0)
        if sentiment == "NEGATIVE":
            if neg_prob > 0.75:
                urgency = "CRITICAL"
            elif neg_prob > 0.55:
                urgency = "HIGH"
            else:
                urgency = "MEDIUM"
        else:
            urgency = "LOW"

        # 7. Mock entities/key phrases since classification doesn't extract them
        entities = []
        for product in ["iPhone", "MacBook", "AirPods", "Galaxy", "Pixel", "Surface"]:
            if product.lower() in text.lower():
                entities.append({"name": product, "type": "PRODUCT"})

        words = text.split()
        key_phrases = []
        if len(words) > 3:
            for i in range(0, min(len(words) - 2, 6), 2):
                phrase = " ".join(words[i:i+3]).strip(".,!?;:")
                if len(phrase) > 5:
                    key_phrases.append(phrase)

        summary = (
            f"[PyTorch] {sentiment.lower().title()} sentiment detected with "
            f"{confidence:.0%} confidence using local EmbeddingBag model."
        )

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "aspects": aspects,
            "entities": entities,
            "key_phrases": key_phrases[:5],
            "urgency": urgency,
            "summary": summary,
        }

    def _fallback_prediction(self, text: str) -> dict[str, Any]:
        """Simple fallback if model is uninitialized or input is empty."""
        return {
            "sentiment": "NEUTRAL",
            "confidence": 0.5,
            "aspects": [{"aspect": "overall experience", "sentiment": "NEUTRAL", "score": 0.5}],
            "entities": [],
            "key_phrases": [],
            "urgency": "LOW",
            "summary": "[PyTorch Fallback] Model uninitialized or input out-of-vocabulary.",
        }
