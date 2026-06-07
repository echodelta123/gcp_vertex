# Demo 5: H&M Fashion Data Ingestion Pipeline

**Features**  
| Feature | Description |
|---|---|
| GCS sensor | Auto‑triggers on new raw files |
| Demo mode | Runs locally with synthetic data |
| dbt integration | Loads dbt models as Dagster assets with tests |
| Nightly schedule | Automated nightly runs at 02:00 UTC |

> A production-pattern data engineering pipeline orchestrating GCS → BigQuery ingestion and dbt SQL transformation, managed entirely through **Dagster software-defined assets**.

## The Business Problem

Millions of transaction records, customer reviews, and product metadata events may be generated daily. Before any ML model can learn from this data, it must be reliably ingested, validated, and transformed into analytical mart tables with schema validation, lineage tracking, ml model re-training runs on stale or incorrect data

## Data Source

This demo is based on the Kaggle **H&M Personalized Fashion Recommendations** competition dataset: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations. The pipeline mirrors the three core raw inputs from that source - customer reviews, transactions, and article metadata - and in `DEMO_MODE` generates synthetic records shaped to match the same general structure so the full Dagster flow can run locally without external dependencies.

## Architecture

```text
GCS Bucket (gs://hm-fashion-raw-data/)
  ├── reviews/*.json          ── raw_fashion_reviews     ─┐
  ├── transactions/*.csv      ── raw_fashion_transactions ┼──► dbt_fashion_marts ──► BigQuery analytics.*
  └── articles/*.csv          ── raw_fashion_articles    ─┘
                                      ▲
                              GCS New-File Sensor
                              (polls every 60s)
```

### Dagster Asset Dependency Graph

```
raw_fashion_reviews ────────────────────────────────────────────────────────┐
                                                                            ▼
raw_fashion_transactions ──────────────────────────────────────► dbt_fashion_marts
                                                                            ▲
raw_fashion_articles ───────────────────────────────────────────────────────┘
```

**dbt mart tables produced:**
- `analytics.mart_customer_behaviour` — RFM aggregations per customer
- `analytics.mart_aspect_sentiments` — NLP-derived sentiment scores per article
- `analytics.mart_demand_forecast_features` — Daily sales time-series for ARIMA+

## Project Structure

```
demo_data_ingestion_dagster_dbt/
├── definitions.py          # ← Dagster entrypoint (Definitions object)
├── resources.py            # BigQuery, GCS, and dbt resource configs
├── sensors.py              # GCS new-file sensor (auto-triggers pipeline)
└── assets/
    ├── raw_ingestion.py    # GCS → BigQuery raw landing assets
    └── dbt_transforms.py   # dbt mart compilation asset
```

## Running Locally (DEMO_MODE)

No GCP credentials required. `DEMO_MODE=true` generates synthetic H&M data and simulates the full pipeline execution locally.

```bash
# Install dependencies
pip install dagster dagster-webserver pandas numpy

# Start Dagster UI (Dagit) — opens on http://localhost:3000
DEMO_MODE=true dagster dev -m demo_data_ingestion_dagster_dbt.definitions

# Or materialise all assets in one shot (no daemon required)
DEMO_MODE=true dagster asset materialize \
  -m demo_data_ingestion_dagster_dbt.definitions \
  --select '*'
```

## Running Against GCP (Production Mode)

```bash
# Set GCP credentials via Application Default Credentials
gcloud auth application-default login

# Configure environment
export GCP_PROJECT_ID="your-gcp-project"
export GCS_BUCKET_NAME="your-hm-raw-data-bucket"
export DBT_PROJECT_DIR="./dbt"
export DBT_TARGET="prod"

# Run the ingestion job
DEMO_MODE=false dagster job execute \
  -m demo_data_ingestion_dagster_dbt.definitions \
  -j hm_fashion_ingestion_job
```

## Dagster Jobs & Schedules

| Job | Trigger | Description |
|-----|---------|-------------|
| `hm_fashion_ingestion_job` | Nightly @ 02:00 UTC or GCS sensor | Full pipeline: raw ingestion + dbt transforms |
| `hm_fashion_raw_only_job` | Manual / ad-hoc | Raw ingestion only (skip dbt) |

## Technical Design Decisions

### Why Dagster over Airflow?
Dagster's **software-defined asset** model surfaces data quality (row counts, schema) as first-class metadata in the UI, whereas Airflow operators treat data as opaque side-effects. This makes debugging data quality issues dramatically faster in production.

### Why dbt for SQL transforms?
dbt compiles, tests, and documents SQL models in a single tool. The schema tests (`not_null`, `unique`, `accepted_range`) run automatically as part of every pipeline execution, catching data quality regressions before they reach downstream ML models.

### Demo Mode Design
The `DEMO_MODE` flag allows the full pipeline DAG to execute with synthetic data — ensuring the project can be run locally via the Dagster UI to observe the asset graph, lineage, and metadata without requiring GCP credentials or incurring costs.

## Downstream Consumers

The `analytics.*` mart tables produced by this pipeline are consumed directly by:
- **[Demo 6: Predict Demand Pipeline](../demo_predict_demand/README.md)** — Feature extraction for XGBoost, K-Means, and ARIMA+ models
- **[Demo 2: Recommendation Engine](../demo_recommendation_engine/README.md)** — Product metadata enrichment for vector embedding generation
- **[Demo 3: Sentiment Analysis](../demo_sentiment_categoriser/README.md)** — Customer review corpus for aspect-level NLP
