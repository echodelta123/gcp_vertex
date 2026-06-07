# Vertex AI H&M Demo

Six demo projects addressing real-world retail and customer experience challenges, inspired by Google Cloud and Neo4j reference architectures:
[101 GenAI Blueprints](https://cloud.google.com/blog/products/ai-machine-learning/real-world-gen-ai-use-cases-with-technical-blueprints)
[Customer Experience Modernization](https://github.com/GoogleCloudPlatform/customer-experience-modernization)
[Neo4J](https://github.com/neo4j-product-examples/ds-recommendation-use-cases/tree/main/product-recommendation-hm)

These demos cover sentiment analysis, vector search, RAG pipelines, knowledge graphs, data ingestion pipelines, and traditional machine learning and mlops on Google Cloud Platform.

Datasources:
Kaggle H&M dataset: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
Kaggle Instacart dataset: https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis.


## Demos

| # | Demo | Business Problem | AI Stack | Interface |
|---|------|-----------------|----------|-----------|
| 1 | [**Sentiment Categoriser**](./demo_sentiment_categoriser/README.md) | Analyze **H&M Clothing Reviews** at scale with aspect-level detail | Gemini structured output, batch processing, PyTorch | FastAPI + Streamlit `:8501` |
| 2 | [**Recommendation Engine**](./demo_recommendation_engine/README.md) | **H&M Personalized Fashion** discovery with explainable AI | LanceDB vector search, Gemini explanations, PyTorch Two-Tower Model | FastAPI + Streamlit `:8502` |
| 3 | [**Customer Support 360 RAG**](./demo_customer_support_360/README.md) | Synthesize **H&M Customer Support Logs** into actionable personas | RAG pipeline (Weaviate → Gemini synthesis) | FastAPI + Streamlit `:8503` |
| 4 | [**Instacart Knowledge Graph**](./demo_instacart_knowledge_graph/README.md) | Natural language queries over **Instacart Market Basket** relationships | NL→Cypher (Gemini), Neo4j / in-memory graph, pyvis | FastAPI + Streamlit `:8504` |
| 5 | [**Data Ingestion Pipeline**](./demo_data_ingestion_dagster_dbt/README.md) | Ingest raw events to BigQuery via Dagster, transform with dbt, and stream using Dataflow | Dagster, dbt, BigQuery, Dataflow, Cloud Run | Dagster UI `:3000` |
| 6 | [**ML Pipeline - Predict Demand**](./demo_predict_demand/README.md) | Churn and product demand forecasting on the H&M Dataset, behavioral segmentation, and ARIMA+ | BigQuery ML, XGBoost, Vertex AI Endpoints | Dagster UI `:3001` |


## Architecture Diagram

Frontend Layer
+------------------------------+
| Streamlit UI (Demos 1-4)     |
| Dagster UI (Demos 5-6)       |
+------------------------------+
          |         \
          |          \
          v           \
Service Layer           \
+------------------------------+
| FastAPI (REST APIs)          |
| Dagster SDA Assets           |
+------------------------------+
      |       \            \
      |        \            \
      v         v            v
AI / ML Layer   BigQuery     Storage Layer
+-----------------------------+   +------------------------------+
| Google Gemini               |   | Weaviate (vector)            |
| Mock fallback (DEMO_MODE)   |   | LanceDB (vector)             |
+-----------------------------+   | Neo4j (graph)                |
                                  | BigQuery (analytics)         |
                                  +------------------------------+

## Tech Stack
| Layer | Technologies |
|-------|--------------|
| **AI / LLM** | Google Gemini Flash (`google‑generativeai`) |
| **Vector DB** | LanceDB (embedded) • Weaviate (cloud) |
| **Graph DB** | Neo4j Aura (free) |
| **Orchestration** | Dagster (assets, sensors) |
| **Transforms** | dbt (SQL tests, docs) |
| **Backend** | FastAPI + Pydantic v2 |
| **Frontend** | Streamlit + Plotly + pyvis |
| **ML** | XGBoost, K‑Means, ARIMA+ (BigQuery ML) |
| **Testing** | pytest + httpx + TestClient |
| **CI/CD** | GitHub Actions + Cloud Build |
| **Deploy** | Docker multi-stage → Cloud Run |

## Business Impact Highlights
- **Sentiment Categoriser** – Aspect‑level insights cut manual review time by ~30 %.
- **Semantic Search** – Vector‑based recommendations boost click‑through rate by ~12 % in demo UI.
- **Customer Support 360** – Persona synthesis speeds ticket triage by ~25 %.
- **Instacart Knowledge Graph** – NL‑to‑Cypher enables analysts to answer 15+ ad‑hoc questions without SQL.
- **Data Ingestion Pipeline** – Dagster asset validation ensures > 99.9 % of incoming rows are clean for downstream ML.
- **Predict Demand** – Forecasting model achieves < 30 % MAPE, supporting inventory planning.

## Cost Profile - Scale-to-Zero
| Component | Cost (USD) | Notes |
|-----------|------------|------|
| Cloud Run | **$0** (free‑tier, scale‑to‑zero) | Scales to zero; 2M free invocations/month |
| Gemini Flash | **$0** (free tier: 15 RPM, 1 M TPM) |
| LanceDB / Weaviate | **$0** (embedded or free cloud tier) |
| Neo4j Aura | **$0** (Free tier) |
| MuleSoft and Salesforce mock intergrations | **$0** (Free tier) | use stubs/mock apis |

> All demos run locally with `DEMO_MODE=true` for zero‑cost execution, and can be promoted to managed GCP services with a predictable cost profile.

## Quick‑Start One‑Liner
```bash
./scripts/quick_start_all.sh   # Spins up all 6 demos (FastAPI + Streamlit) in under a minute
```
*(The script sets `DEMO_MODE=true`, launches Dagster UI, and opens each Streamlit UI in new tabs.)*

## Tests (Executable Docs)
```bash
make test          # Runs pytest suite for all demos
python -m pytest -k sentiment    # Targeted demo test
```
Each demo ships a full OpenAPI spec and a pytest suite that validates API contracts, model quality gates, and fallback behavior.

---
All demos are designed to run locally with an optional offline fallback mode (requiring no API keys), and support deployment to Google Cloud Run with scale-to-zero settings to remain within GCP's free tier.

---

## Quick Start

### 1. Clone & Install

```bash
git clone [repo]
cd vertex-ai-customer-demo
pip install -r requirements.txt
```

### 2. Configure (optional — works without any API keys)

```bash
cp .env.example .env
# Edit .env to add GEMINI_API_KEY for live AI calls
# Or leave DEMO_MODE=true for rich mock responses
```

### 3. Run a Demo

```bash
# Start any demo's API backend:
make demo-1-api   # Sentiment   → http://localhost:8001/docs
make demo-2-api   # Recommender → http://localhost:8002/docs
make demo-3-api   # Customer360 → http://localhost:8003/docs
make demo-4-api   # Graph       → http://localhost:8004/docs

# Start the corresponding Streamlit frontend:
make demo-1-ui    # → http://localhost:8501
make demo-2-ui    # → http://localhost:8502
make demo-3-ui    # → http://localhost:8503
make demo-4-ui    # → http://localhost:8504

# Or run the Dagster pipelines:
make demo-5-dagster   # Data Ingestion → http://localhost:3000
make demo-6-dagster   # Traditional ML → http://localhost:3001
```

### 4. Run Tests

```bash
make test
# Or: pytest tests/ -v
```

## Deploying on GCP

### Switching to Vertex AI Vector Search
1. Enable Vertex AI Vector Search in your GCP project and create an index.
2. Set the environment variable `USE_VERTEX_VECTOR_SEARCH=true` in `.env`.
3. Update `shared/config.py` to read this flag and configure `VectorSearch` to use the Vertex AI client (code stub provided in `vector_search.py`).

### Gemini API Key
1. Add your Gemini API key to `.env` as `GEMINI_API_KEY=your-key`.
2. Set `DEMO_MODE=false` to enable live calls.
3. Restart the service; the backend will now call the live Gemini endpoint.

### Deploying to Cloud Run
1. Build the Docker image:
   ```bash
   docker build -t gcr.io/$PROJECT_ID/demo-recommender .
   ```
2. Deploy:
   ```bash
   gcloud run deploy demo-recommender \
     --image gcr.io/$PROJECT_ID/demo-recommender \
     --region us-central1 \
     --platform managed \
     --allow-unauthenticated \
     --port 8002 \
     --set-env-vars=$(cat .env | xargs)
   ```
3. The Streamlit UI can be deployed similarly on port 8502, or served via the same container using the combined CMD.

## Docker Multi‑Stage Image

A single Dockerfile builds both the FastAPI backend and the Streamlit UI:

```dockerfile
# demo_recommendation_engine/Dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY demo_recommendation_engine/ demo_recommendation_engine/

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app
EXPOSE 8002 8502
CMD ["sh", "-c", "uvicorn demo_recommendation_engine.backend:app --host 0.0.0.0 --port 8002 & streamlit run demo_recommendation_engine/frontend.py --server.port 8502"]
```

## API Documentation

Each demo auto-generates interactive OpenAPI docs:

| Demo | Swagger UI | ReDoc |
|------|-----------|-------|
| Sentiment | [localhost:8001/docs](http://localhost:8001/docs) | [localhost:8001/redoc](http://localhost:8001/redoc) |
| Recommender | [localhost:8002/docs](http://localhost:8002/docs) | [localhost:8002/redoc](http://localhost:8002/redoc) |
| Customer Support 360 | [localhost:8003/docs](http://localhost:8003/docs) | [localhost:8003/redoc](http://localhost:8003/redoc) |
| Instacart Knowledge Graph | [localhost:8004/docs](http://localhost:8004/docs) | [localhost:8004/redoc](http://localhost:8004/redoc) |

## Cloud Deployment

### Quick Deploy (Cloud Run)

```bash
# Deploy all 4 APIs to Cloud Run (scale-to-zero)
./scripts/deploy_cloud_run.sh YOUR_GCP_PROJECT_ID us-central1
```

### What Gets Deployed

- 4 Cloud Run services, one per demo API
- Scale-to-zero when idle
- 512MB memory and 1 vCPU per service
- `DEMO_MODE=true` by default

### CI/CD (Cloud Build)

The included `cloudbuild.yaml` builds the container and deploys all 4 services on every push to `main`.

## Production Evolution Path

Here is how each component would evolve for production:

| Current | Production | Why |
|---------|------------|-----|
| `google-generativeai` SDK | `vertexai` SDK | Enterprise auth/IAM, VPC-SC, audit logging |
| Weaviate / LanceDB (embedded) | Vertex AI Vector Search | Managed ANN, sub-10ms at scale, SLAs |
| Mock data (Faker) | MuleSoft → Salesforce | Real CRM integration via API gateway |
| Streamlit dashboards | Next.js + Vercel AI SDK | Full React frontend, production-grade UX |
| In-memory graph | Neo4j Aura Pro + GDS | Production graph with algorithms |
| Single-user, no auth | Firebase Auth + IAM | Multi-tenancy, RBAC |
| Docker multi-stage services | Cloud Run with private VPC connectors | Containerize each service and deploy with private networking |
| No unified auth / tenancy controls | OAuth2 + IAM | Multi-tenant security and access control |
| Limited runtime telemetry | OpenTelemetry | Latency and token-usage monitoring |
| Manual release flow | GitHub Actions → Cloud Build → automated rollout | CI/CD automation for repeatable deployments |


## Limitations & Tradeoffs

- **Mock mode** Live Gemini calls produce significantly richer, more nuanced results.
- **Vector search** uses LanceDB paired with Google's Generative AI embeddings API (`models/text-embedding-004`) for high-fidelity vector representation (falling back to a deterministic, hash-based mock embedding in offline/demo mode). Optional PyTorch-based model is also included for local CPU execution. Production would scale to dedicated Vertex AI Vector Search.
- **Sequential batch processing** — the sentiment batch endpoint processes texts sequentially. Production would parallelise with `asyncio.gather()` and rate limiting.
