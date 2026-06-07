"""
Sentiment analysis service layer.
Handles business logic, AI backend orchestration (Gemini or Ollama), and mock fallback.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Any

from shared.config import settings
from shared.gemini_client import GeminiClient
from shared.ollama_client import OllamaClient
from demo_sentiment_categoriser.pytorch_client import LocalPyTorchClient
from demo_sentiment_categoriser.models import (
    SentimentLabel,
    UrgencyLevel,
    SentimentResult,
    AspectSentiment,
    Entity,
    IABAspectSentiment,
)
from demo_sentiment_categoriser.prompts import (
    build_analysis_prompt,
    SENTIMENT_SYSTEM_INSTRUCTION,
)

logger = logging.getLogger(__name__)


class SentimentService:
    """Core sentiment analysis service with support for multiple AI backends."""

    def __init__(self, client: GeminiClient | None = None):
        self.gemini_client = client or GeminiClient()
        self._ollama_client = None
        self.pytorch_client = LocalPyTorchClient()
        self._history: list[SentimentResult] = []

    @property
    def ollama_client(self):
        if self._ollama_client is None:
            self._ollama_client = OllamaClient()
        return self._ollama_client

    async def analyze(self, text: str, context: str | None = None, backend: str | None = None) -> SentimentResult:
        """Analyze a single piece of text.

        Delegates to the selected backend.
        """
        active_backend = (backend or settings.MODEL_BACKEND).upper()

        if active_backend == "LOCAL_PYTORCH":
            result = self._analyze_pytorch(text)
        elif active_backend == "HEURISTIC":
            result = self._analyze_mock(text, context)
        elif active_backend == "OLLAMA":
            result = await self._analyze_live(text, context, backend="OLLAMA")
        elif active_backend == "GEMINI":
            if self.gemini_client.is_live:
                result = await self._analyze_live(text, context, backend="GEMINI")
            else:
                result = self._analyze_mock(text, context)
        else:
            # Fallback based on config or client status
            if self.gemini_client.is_live:
                result = await self._analyze_live(text, context, backend="GEMINI")
            else:
                result = self._analyze_mock(text, context)

        result.source_text = text[:200]
        self._history.append(result)
        return result

    async def analyze_batch(self, items: list[dict], backend: str | None = None) -> list[SentimentResult]:
        """Analyze a batch of texts.

        ``items`` is a list of ``{"text": ..., "context": ...}`` dictionaries.
        Returns a list of :class:`SentimentResult` objects.
        """
        results = []
        for item in items:
            result = await self.analyze(item["text"], item.get("context"), backend=backend)
            results.append(result)
        return results

    def get_history(self) -> list[SentimentResult]:
        """Return the most recent 50 analysis results (most recent first)."""
        return list(reversed(self._history[-50:]))

    def get_aggregate_stats(self, results: list[SentimentResult]) -> dict:
        """Calculate aggregate statistics for a batch of results.

        Returns a dictionary with totals, distributions, average confidence and
        per‑aspect summary statistics.
        """
        if not results:
            return {}
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0, "MIXED": 0}
        urgency_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        total_confidence = 0.0
        all_aspects: dict[str, dict[str, list[Any]]] = {}
        for r in results:
            sentiment_counts[r.sentiment.value] += 1
            urgency_counts[r.urgency.value] += 1
            total_confidence += r.confidence
            for a in r.aspects:
                key = a.aspect.lower()
                if key not in all_aspects:
                    all_aspects[key] = {"scores": [], "sentiments": []}
                all_aspects[key]["scores"].append(a.score)
                all_aspects[key]["sentiments"].append(a.sentiment.value)
        aspect_summary = {}
        for aspect, data in all_aspects.items():
            avg_score = sum(data["scores"]) / len(data["scores"])
            dominant = max(set(data["sentiments"]), key=data["sentiments"].count)
            aspect_summary[aspect] = {
                "avg_score": round(avg_score, 2),
                "dominant_sentiment": dominant,
                "mentions": len(data["scores"]),
            }
        return {
            "total": len(results),
            "sentiment_distribution": sentiment_counts,
            "urgency_distribution": urgency_counts,
            "avg_confidence": round(total_confidence / len(results), 3),
            "aspect_summary": aspect_summary,
        }

    # --- Live backend analysis -------------------------------------------------
    async def _analyze_live(self, text: str, context: str | None, backend: str = "GEMINI") -> SentimentResult:
        """Call the configured AI backend for a real analysis.

        The prompt is built with :func:`build_analysis_prompt` and the backend
        response is expected to be a JSON object matching the schema used by the
        mock implementation.
        """
        prompt = build_analysis_prompt(text, context)
        client = self.ollama_client if backend == "OLLAMA" else self.gemini_client
        data = client.generate_json(
            prompt, system_instruction=SENTIMENT_SYSTEM_INSTRUCTION
        )
        if "error" in data:
            logger.warning(f"Backend parse error, falling back to mock: {data['error']}")
            return self._analyze_mock(text, context)
        try:
            return SentimentResult(
                sentiment=SentimentLabel(data.get("sentiment", "NEUTRAL")),
                confidence=float(data.get("confidence", 0.5)),
                aspects=[AspectSentiment(**a) for a in data.get("aspects", [])],
                entities=[Entity(**e) for e in data.get("entities", [])],
                key_phrases=data.get("key_phrases", []),
                urgency=UrgencyLevel(data.get("urgency", "LOW")),
                summary=data.get("summary", ""),
                iab_aspects=[
                    IABAspectSentiment(
                        aspect=a["aspect"],
                        sentiment=a["sentiment"],
                        score=a["score"],
                        confidence=a["score"],
                    )
                    for a in data.get("aspects", [])
                ],
            )
        except Exception as e:
            logger.error(f"Failed to parse backend response: {e}")
            return self._analyze_mock(text, context)

    def _analyze_pytorch(self, text: str) -> SentimentResult:
        """Analyze text using local PyTorch model."""
        res = self.pytorch_client.predict(text)
        return SentimentResult(
            sentiment=SentimentLabel(res["sentiment"]),
            confidence=res["confidence"],
            aspects=[AspectSentiment(**a) for a in res["aspects"]],
            entities=[Entity(**e) for e in res["entities"]],
            key_phrases=res["key_phrases"],
            urgency=UrgencyLevel(res["urgency"]),
            summary=res["summary"],
            iab_aspects=[
                IABAspectSentiment(
                    aspect=a["aspect"],
                    sentiment=a["sentiment"],
                    score=a["score"],
                    confidence=a["score"],
                )
                for a in res["aspects"]
            ],
        )

    # --- Mock analysis --------------------------------------------------------
    def _analyze_mock(self, text: str, context: str | None) -> SentimentResult:
        """Deterministic mock analysis used when no live backend is available.

        The heuristics are deliberately simple but produce realistic‑looking
        output for demo purposes.
        """
        text_lower = text.lower()
        pos_words = {
            "love",
            "amazing",
            "excellent",
            "great",
            "fantastic",
            "wonderful",
            "best",
            "perfect",
            "exceeded",
            "recommend",
        }
        neg_words = {
            "terrible",
            "awful",
            "worst",
            "horrible",
            "disappointed",
            "broken",
            "refund",
            "frustrated",
            "slow",
            "damaged",
        }
        pos_count = sum(1 for w in pos_words if w in text_lower)
        neg_count = sum(1 for w in neg_words if w in text_lower)
        if pos_count > neg_count:
            sentiment = SentimentLabel.POSITIVE
            confidence = min(0.95, 0.65 + pos_count * 0.1)
        elif neg_count > pos_count:
            sentiment = SentimentLabel.NEGATIVE
            confidence = min(0.95, 0.65 + neg_count * 0.1)
        elif pos_count > 0 and neg_count > 0:
            sentiment = SentimentLabel.MIXED
            confidence = 0.72
        else:
            sentiment = SentimentLabel.NEUTRAL
            confidence = 0.60
        aspects = []
        aspect_keywords = {
            "product quality": (
                ["quality", "build", "material", "durable", "solid", "premium"],
                0.85,
            ),
            "shipping speed": (
                ["shipping", "delivery", "arrived", "transit", "slow", "fast"],
                0.78,
            ),
            "customer service": (
                ["service", "support", "agent", "response", "help", "call"],
                0.80,
            ),
            "value for money": (
                ["price", "value", "expensive", "cheap", "overpriced", "worth"],
                0.75,
            ),
            "user experience": (
                ["easy", "intuitive", "confusing", "app", "interface", "feature"],
                0.72,
            ),
            "packaging": (
                ["packaging", "box", "wrapped", "damaged", "unboxing"],
                0.70,
            ),
        }
        for aspect_name, (keywords, base_score) in aspect_keywords.items():
            if any(kw in text_lower for kw in keywords):
                aspect_pos = any(pw in text_lower for pw in pos_words)
                aspect_neg = any(nw in text_lower for nw in neg_words)
                if aspect_pos and not aspect_neg:
                    a_sentiment = SentimentLabel.POSITIVE
                    a_score = min(0.95, base_score + random.uniform(0, 0.1))
                elif aspect_neg and not aspect_pos:
                    a_sentiment = SentimentLabel.NEGATIVE
                    a_score = max(0.15, base_score - random.uniform(0.2, 0.4))
                else:
                    a_sentiment = SentimentLabel.MIXED
                    a_score = base_score
                aspects.append(
                    AspectSentiment(
                        aspect=aspect_name,
                        sentiment=a_sentiment,
                        score=round(a_score, 2),
                    )
                )
        if not aspects:
            aspects.append(
                AspectSentiment(
                    aspect="overall experience",
                    sentiment=sentiment,
                    score=round(confidence, 2),
                )
            )
        iab_aspects = [
            IABAspectSentiment(
                aspect=a.aspect,
                sentiment=a.sentiment,
                score=a.score,
                confidence=a.score,
            )
            for a in aspects
        ]
        entities = []
        for product in ["iPhone", "MacBook", "AirPods", "Galaxy", "Pixel", "Surface"]:
            if product.lower() in text_lower:
                entities.append(Entity(name=product, type="PRODUCT"))
        urgency = UrgencyLevel.LOW
        if neg_count >= 3 or "refund" in text_lower or "legal" in text_lower:
            urgency = UrgencyLevel.CRITICAL
        elif neg_count >= 2 or "complaint" in text_lower:
            urgency = UrgencyLevel.HIGH
        elif neg_count >= 1:
            urgency = UrgencyLevel.MEDIUM
        words = text.split()
        key_phrases = []
        if len(words) > 3:
            for i in range(0, min(len(words) - 2, 6), 2):
                phrase = " ".join(words[i:i+3]).strip(".,!?;:")
                if len(phrase) > 5:
                    key_phrases.append(phrase)
        summary = (
            f"{sentiment.value.lower().title()} sentiment detected with "
            f"{confidence:.0%} confidence across {len(aspects)} aspect(s)."
        )
        return SentimentResult(
            sentiment=sentiment,
            confidence=round(confidence, 3),
            aspects=aspects,
            iab_aspects=iab_aspects,
            entities=entities,
            key_phrases=key_phrases[:5],
            urgency=urgency,
            summary=summary,
        )
