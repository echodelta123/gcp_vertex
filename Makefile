.PHONY: help install dev test lint format clean \
       demo-1-api demo-1-ui demo-2-api demo-2-ui \
       demo-3-api demo-3-ui demo-4-api demo-4-ui \
       demo-5-dagster demo-6-dagster \
       docker-build docker-up docker-down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (lean – demo mode)
	pip install -r requirements.txt

install-full: ## Install all dependencies (including Vertex AI SDK)
	pip install -r requirements-full.txt

# ---------------------------------------------------------------------------
# Run individual demos – API backends (Demos 1–4: FastAPI + Streamlit)
# ---------------------------------------------------------------------------

demo-1-api: ## Run Sentiment API (port 8001)
	uvicorn demo_sentiment_categoriser.backend:app --reload --port 8001

demo-2-api: ## Run Semantic Search API (port 8002)
	uvicorn demo_recommendation_engine.backend:app --reload --port 8002

demo-3-api: ## Run Customer 360 API (port 8003)
	uvicorn demo_customer_support_360.backend:app --reload --port 8003

demo-4-api: ## Run Graph Explorer API (port 8004)
	uvicorn demo_instacart_knowledge_graph.backend:app --reload --port 8004

# ---------------------------------------------------------------------------
# Run individual demos – Streamlit frontends
# ---------------------------------------------------------------------------

demo-1-ui: ## Run Sentiment UI (port 8501)
	streamlit run demo_sentiment_categoriser/frontend.py --server.port 8501

demo-2-ui: ## Run Semantic Search UI (port 8502)
	streamlit run demo_recommendation_engine/frontend.py --server.port 8502

demo-3-ui: ## Run Customer 360 UI (port 8503)
	streamlit run demo_customer_support_360/frontend.py --server.port 8503

demo-4-ui: ## Run Graph Explorer UI (port 8504)
	streamlit run demo_instacart_knowledge_graph/frontend.py --server.port 8504

# ---------------------------------------------------------------------------
# Run Dagster pipelines (Demos 5–6: orchestration pipelines)
# ---------------------------------------------------------------------------

demo-5-dagster: ## Run Data Ingestion Dagster UI (port 3000)
	DEMO_MODE=true dagster dev -m demo_data_ingestion.definitions

demo-6-dagster: ## Run Traditional ML Dagster UI (port 3001)
	DEMO_MODE=true dagster dev -m demo_traditional_ml.definitions -p 3001

# ---------------------------------------------------------------------------
# Testing & Quality
# ---------------------------------------------------------------------------

test: ## Run all tests
	python -m pytest tests/ -v --tb=short

lint: ## Lint with Ruff
	python -m ruff check .

format: ## Auto-format code
	python -m ruff format .

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build: ## Build all Docker images
	docker compose build

docker-up: ## Start all services via Docker Compose
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean: ## Remove caches and temp artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .chromadb/ htmlcov/ .coverage
