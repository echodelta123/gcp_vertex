#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Deploy all 4 demo APIs to Google Cloud Run
# Usage: ./scripts/deploy_cloud_run.sh <GCP_PROJECT_ID> [REGION]
# ──────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ID="${1:?Usage: $0 <GCP_PROJECT_ID> [REGION]}"
REGION="${2:-us-central1}"
IMAGE_TAG="gcr.io/${PROJECT_ID}/vertex-ai-customer-demo"

echo "══════════════════════════════════════════════════"
echo "  Deploying Vertex AI Customer Demo to Cloud Run"
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "══════════════════════════════════════════════════"

# Enable required APIs
echo "→ Enabling required GCP APIs..."
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  --project="${PROJECT_ID}" --quiet

# Build the container image
echo "→ Building container image..."
gcloud builds submit \
  --tag "${IMAGE_TAG}:latest" \
  --project="${PROJECT_ID}" \
  --quiet

# Deploy each demo as a separate Cloud Run service
declare -A DEMOS=(
  ["sentiment-api"]="demo_sentiment_categoriser.backend:app --port 8001"
  ["recommender-api"]="demo_recommendation_engine.backend:app --port 8002"
  ["customer360-api"]="demo_customer_support_360.backend:app --port 8003"
  ["graph-explorer-api"]="demo_instacart_knowledge_graph.backend:app --port 8004"
)

declare -A PORTS=(
  ["sentiment-api"]="8001"
  ["recommender-api"]="8002"
  ["customer360-api"]="8003"
  ["graph-explorer-api"]="8004"
)

for SERVICE in "${!DEMOS[@]}"; do
  PORT="${PORTS[$SERVICE]}"
  echo ""
  echo "→ Deploying ${SERVICE} on port ${PORT}..."
  gcloud run deploy "${SERVICE}" \
    --image "${IMAGE_TAG}:latest" \
    --platform managed \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --port "${PORT}" \
    --command "uvicorn" \
    --args "${DEMOS[$SERVICE]//--port/--host,0.0.0.0,--port}" \
    --set-env-vars "DEMO_MODE=true" \
    --memory "512Mi" \
    --cpu "1" \
    --min-instances 0 \
    --max-instances 3 \
    --allow-unauthenticated \
    --quiet

  URL=$(gcloud run services describe "${SERVICE}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --format 'value(status.url)')
  echo "  ✅ ${SERVICE}: ${URL}"
done

echo ""
echo "══════════════════════════════════════════════════"
echo "  Deployment complete! All services are live."
echo "  Each scales to zero when idle (cost: \$0)."
echo "══════════════════════════════════════════════════"
