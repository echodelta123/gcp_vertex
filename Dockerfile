FROM python:3.11-slim

WORKDIR /app

# Install deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY shared/ shared/
COPY demo_sentiment_categoriser/ demo_sentiment_categoriser/
COPY demo_recommendation_engine/ demo_recommendation_engine/
COPY demo_customer_support_360/ demo_customer_support_360/
COPY demo_instacart_knowledge_graph/ demo_instacart_knowledge_graph/
COPY .env.example .env.example

# Health check (override CMD per service in docker-compose)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/api/v1/health')" || exit 1

# Default: run sentiment API (override in docker-compose)
CMD ["uvicorn", "demo_sentiment_categoriser.backend:app", "--host", "0.0.0.0", "--port", "8001"]
