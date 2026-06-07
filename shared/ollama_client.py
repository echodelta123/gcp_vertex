import json
import logging
import requests
from typing import Any

from shared.config import settings

logger = logging.getLogger(__name__)

class OllamaClient:
    """Simple Ollama client wrapper.

    Provides `generate_json` similar to GeminiClient, using Ollama's /api/generate endpoint.
    Supports optional model override via constructor.
    """

    def __init__(self, model: str | None = None):
        self._model = model or settings.OLLAMA_MODEL
        self._endpoint = settings.OLLAMA_ENDPOINT
        self._is_live = True  # Ollama assumed available locally
        logger.info(f"🔶 OllamaClient initialized with model {self._model} at {self._endpoint}")

    def generate_json(self, prompt: str, system_instruction: str = "", temperature: float = 0.1) -> dict[str, Any]:
        """Generate JSON response via Ollama.

        Sends the prompt (and optional system instruction) to Ollama and expects a JSON string in the response.
        """
        payload = {
            "model": self._model,
            "prompt": f"{system_instruction}\n{prompt}" if system_instruction else prompt,
            "temperature": temperature,
            "stream": False,
        }
        try:
            response = requests.post(self._endpoint, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            # Ollama returns a 'response' field with the generated text.
            raw = data.get("response", "")
            return self._parse_json(raw)
        except Exception as e:
            logger.warning(f"⚠️ Ollama request failed: {e}. Returning mock.")
            return self._mock_json(prompt)

    def _parse_json(self, raw: str) -> dict[str, Any]:
        """Parse JSON from Ollama output, handling possible markdown fences."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from Ollama response: {text[:200]}")
            return {"error": "Failed to parse response", "raw": text[:500]}

    def _mock_json(self, prompt: str) -> dict[str, Any]:
        """Fallback mock JSON when Ollama is unavailable."""
        return {"demo_mode": True, "prompt_length": len(prompt)}
