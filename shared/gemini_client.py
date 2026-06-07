"""
Unified Gemini client with automatic fallback to rich mock responses.

Supports two modes:
  1. LIVE MODE  – Calls Google Gemini API via `google-generativeai` SDK
  2. DEMO MODE  – Returns realistic mock responses (no API key required)

Usage:
    from shared.gemini_client import GeminiClient
    client = GeminiClient()
    result = client.generate(prompt, system_instruction="You are...")
    structured = client.generate_json(prompt, system_instruction="...")
"""
import json
import logging
import hashlib
from typing import Any

from shared.config import settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """Wrapper around Gemini or Vertex AI text model with transparent mock fallback."""

    def __init__(self, model: str | None = None):
        self._model = None
        self._is_live = False
        self._model_override = model
        self._setup()

    def _setup(self):
        if settings.effective_demo_mode:
            logger.info("🔶 GeminiClient running in DEMO MODE (mock responses)")
            return

        try:
            import google.generativeai as genai

            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._model = genai.GenerativeModel(settings.GEMINI_MODEL)
            self._is_live = True
            logger.info(
                f"✅ GeminiClient connected to {settings.GEMINI_MODEL} (LIVE)"
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize Gemini, falling back to mock: {e}")

    @property
    def is_live(self) -> bool:
        return self._is_live

    def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.3,
    ) -> str:
        """Generate text response from Gemini or mock."""
        if self._is_live:
            return self._call_live(prompt, system_instruction, temperature)
        return self._mock_text(prompt)

    def generate_json(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Generate structured JSON response from Gemini or mock."""
        if self._is_live:
            raw = self._call_live(prompt, system_instruction, temperature)
            return self._parse_json(raw)
        return self._mock_json(prompt)

    def get_embedding(self, text: str, model: str = "models/text-embedding-004") -> list[float]:
        """Generate embedding vector using Gemini/Vertex API or a deterministic mock fallback."""
        if self._is_live:
            try:
                import google.generativeai as genai
                result = genai.embed_content(
                    model=model,
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
            except Exception as e:
                logger.warning(f"Failed to generate live embedding: {e}. Falling back to mock.")

        return self._generate_mock_embedding(text)

    def _generate_mock_embedding(self, text: str, dimensions: int = 768) -> list[float]:
        """Generate a deterministic mock embedding vector based on text hash."""
        import numpy as np
        h = hashlib.sha256(text.encode('utf-8')).digest()
        seed = int.from_bytes(h[:4], byteorder='big')
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dimensions)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


    def _call_live(
        self, prompt: str, system_instruction: str, temperature: float
    ) -> str:
        """Make actual Gemini API call."""
        import google.generativeai as genai

        model = genai.GenerativeModel(
            settings.GEMINI_MODEL,
            system_instruction=system_instruction or None,
            generation_config=genai.GenerationConfig(temperature=temperature),
        )
        response = model.generate_content(prompt)
        return response.text

    def _parse_json(self, raw: str) -> dict[str, Any]:
        """Extract JSON from Gemini response (handles markdown fences)."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from Gemini: {text[:200]}")
            return {"error": "Failed to parse response", "raw": text[:500]}

    # --- Mock response generators (deterministic based on prompt hash) ---

    def _mock_text(self, prompt: str) -> str:
        return f"[DEMO MODE] Analysis complete for input of length {len(prompt)}."

    def _mock_json(self, prompt: str) -> dict[str, Any]:
        """Return a generic mock JSON. Override in service-specific code."""
        return {"demo_mode": True, "prompt_length": len(prompt)}


# Module-level singleton
gemini = GeminiClient()
