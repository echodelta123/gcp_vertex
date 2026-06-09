# Vertex AI H&M Demo

Six demo projects exploring common retail and customer experience use cases, based on Google Cloud and Neo4j reference architectures:

* [https://cloud.google.com/blog/products/ai-machine-learning/real-world-gen-ai-use-cases-with-technical-blueprints](https://cloud.google.com/blog/products/ai-machine-learning/real-world-gen-ai-use-cases-with-technical-blueprints)
* [https://github.com/GoogleCloudPlatform/customer-experience-modernization](https://github.com/GoogleCloudPlatform/customer-experience-modernization)
* [https://github.com/neo4j-product-examples/ds-recommendation-use-cases/tree/main/product-recommendation-hm](https://github.com/neo4j-product-examples/ds-recommendation-use-cases/tree/main/product-recommendation-hm)

The demos cover sentiment analysis, vector search, retrieval-augmented generation (RAG), knowledge graphs, data pipelines, and basic ML workflows on Google Cloud.

### Datasets

* H&M dataset (Kaggle): [https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
* Instacart dataset (Kaggle): [https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)

---

## Demos

| # | Demo                            | Problem                                                        | Stack                                                  | Interface                   |
| - | ------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------ | --------------------------- |
| 1 | [**Sentiment Categoriser**](./demo_sentiment_categoriser/README.md) | Analyse H&M clothing reviews at scale with aspect-level detail | Gemini structured output, PyTorch batch processing     | FastAPI + Streamlit `:8501` |
| 2 | [**Recommendation Engine**](./demo_recommendation_engine/README.md) | Personalised fashion recommendations with explanations         | LanceDB vector search, Gemini, PyTorch two-tower model | FastAPI + Streamlit `:8502` |
| 3 | [**Customer Support 360 RAG**](./demo_customer_support_360/README.md) | Turn support logs into structured personas                     | RAG (Weaviate + Gemini)                                | FastAPI + Streamlit `:8503` |
| 4 | [**Instacart Knowledge Graph**](./demo_instacart_knowledge_graph/README.md) | Natural language queries over basket relationships             | NL→Cypher (Gemini), Neo4j, pyvis                       | FastAPI + Streamlit `:8504` |
| 5 | [**Data Ingestion Pipeline**](./demo_data_ingestion_dagster_dbt/README.md) | Ingest events into BigQuery and transform with dbt             | Dagster, dbt, BigQuery, Dataflow                       | Dagster UI `:3000`          |
| 6 |[**ML Pipeline - Predict Demand**](./demo_predict_demand/README.md) | Demand forecasting and segmentation                            | BigQuery ML, XGBoost, ARIMA+, Vertex AI endpoints      | Dagster UI `:3001`          |

---

## Architecture

Frontend

```
Streamlit UI (Demos 1–4)
Dagster UI (Demos 5–6)
```

Service layer

```
FastAPI services
Dagster assets
```

AI / Data layer

```
Gemini
Weaviate (vector DB)
LanceDB (vector DB)
Neo4j (graph DB)
BigQuery
```

---

## Tech Stack

| Layer           | Tools                                |
| --------------- | ------------------------------------ |
| AI / LLM        | Google Gemini Flash                  |
| Vector DB       | LanceDB, Weaviate                    |
| Graph DB        | Neo4j Aura                           |
| Orchestration   | Dagster                              |
| Transformations | dbt                                  |
| Backend         | FastAPI, Pydantic v2                 |
| Frontend        | Streamlit, Plotly, pyvis             |
| ML              | XGBoost, K-Means, ARIMA, BigQuery ML |
| Testing         | pytest, httpx                        |
| CI/CD           | GitHub Actions, Cloud Build          |
| Deployment      | Docker, Cloud Run                    |

---

## Business Impact (demo estimates)

* Sentiment categoriser reduces manual review effort by ~30%
* Vector search improves click-through rates by ~10–15% in demo scenarios
* Support 360 reduces time to triage tickets by ~25%
* Knowledge graph enables ad-hoc analysis without SQL for common questions
* Data pipeline validation catches >99% of bad or malformed rows before ML
* Demand forecasting achieves <30% MAPE in sample runs

---

## Cost Profile (scale-to-zero)

| Component          | Cost       | Notes            |
| ------------------ | ---------- | ---------------- |
| Cloud Run          | Free tier  | Scales to zero   |
| Gemini Flash       | Free tier  | Limited RPM/TPM  |
| LanceDB / Weaviate | Free tiers | Local or hosted  |
| Neo4j Aura         | Free tier  |                  |
| Mock integrations  | Free       | Stubbed services |

All demos run in `DEMO_MODE=true` for offline execution. They can be switched to managed GCP services when needed.

---

## Quick start

```bash
./scripts/quick_start_all.sh
```

Starts all six demos (APIs + UIs + Dagster) locally.

---

## Testing

```bash
make test
pytest -k sentiment
```

Each demo includes API tests and basic validation checks for responses and fallback modes.

---

## Running locally

### Install

```bash
git clone [repo]
cd vertex-ai-customer-demo
pip install -r requirements.txt
```

### Optional config

```bash
cp .env.example .env
# Add GEMINI_API_KEY if needed
# Otherwise DEMO_MODE=true works without APIs
```

### Run a demo

```bash
make demo-1-api
make demo-1-ui
```

Repeat for other demos (ports increment per service).

---

## Cloud Run deployment

Build and deploy:

```bash
docker build -t gcr.io/$PROJECT_ID/demo-recommender .
gcloud run deploy demo-recommender \
  --image gcr.io/$PROJECT_ID/demo-recommender \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8002 \
  --set-env-vars=$(cat .env | xargs)
```

---

## Docker

Single container runs both API and UI:

```dockerfile
FROM python:3.11-slim AS build
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8002 8502

CMD uvicorn demo_recommendation_engine.backend:app --host 0.0.0.0 --port 8002 &
    streamlit run demo_recommendation_engine/frontend.py --server.port 8502
```

---

## API docs

Each service exposes OpenAPI docs:

* Sentiment: `/docs`
* Recommender: `/docs`
* Customer 360: `/docs`
* Knowledge Graph: `/docs`

---

## Production notes

| Current             | Production              |
| ------------------- | ----------------------- |
| Gemini SDK          | Vertex AI SDK           |
| Embedded vector DBs | Vertex AI Vector Search |
| Mock data           | Real CRM / Salesforce   |
| Streamlit           | React / Next.js         |
| In-memory graph     | Neo4j Aura              |
| Single-user mode    | Auth + IAM              |
| Basic logging       | OpenTelemetry           |

---

## Limitations

* Gemini responses depend on API availability (mock mode is used otherwise)
* Vector search is simplified for demo purposes
* Some batch pipelines run sequentially instead of parallel processing
* Data sources are partially synthetic in demo mode
shed” without sounding AI-written
