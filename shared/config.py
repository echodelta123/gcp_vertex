"""
Centralized configuration management for the demo.
Reads from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings:
    """Application settings loaded from environment."""

    # --- AI / Gemini & alternatives ---
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "true").lower() == "true"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    # New backend selector: GEMINI (default), BISON (Vertex AI text-bison), OLLAMA (local), LOCAL_PYTORCH, HEURISTIC
    MODEL_BACKEND: str = os.getenv("MODEL_BACKEND", "GEMINI").upper()
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_K_M")
    OLLAMA_ENDPOINT: str = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "8"))
    GCP_PROJECT: str = os.getenv("GCP_PROJECT", "")
    GCP_LOCATION: str = os.getenv("GCP_LOCATION", "us-central1")
    # New flag for Vertex AI Vector Search
    USE_VERTEX_VECTOR_SEARCH: bool = os.getenv("USE_VERTEX_VECTOR_SEARCH", "false").lower() == "true"

    # --- Neo4j (Demo 4) ---
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")

    # --- Weaviate ---
    WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    WEAVIATE_API_KEY: str = os.getenv("WEAVIATE_API_KEY", "")

    # --- Server ---
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    CORS_ORIGINS: list[str] = os.getenv(
        "CORS_ORIGINS", "http://localhost:8501,http://localhost:3000"
    ).split(",")

    # --- Paths ---
    PROJECT_ROOT: Path = _PROJECT_ROOT

    @property
    def has_gemini_credentials(self) -> bool:
        return bool(self.GEMINI_API_KEY) or bool(self.GCP_PROJECT)

    @property
    def effective_demo_mode(self) -> bool:
        """True if we should use mock responses (no credentials or explicit demo mode)."""
        return self.DEMO_MODE or not self.has_gemini_credentials


settings = Settings()
