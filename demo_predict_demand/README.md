# Demo 6: H&M Fashion Traditional ML Pipeline

> An end-to-end Dagster ML pipeline training XGBoost, K-Means, and ARIMA+ models against H&M fashion retail data — with automated quality gate evaluation and Vertex AI Model Registry integration.

## The Business Problem

Modern retail relies on predictive ML — but deploying and maintaining classical ML models in production requires more than a Jupyter notebook. This pipeline demonstrates how to:

1. **Predict customer churn** before it happens (giving the CRM team time to intervene)
2. **Segment customers** into behavioural personas for targeted marketing
3. **Forecast product demand** 30 days ahead for inventory and supply chain planning

These are bread-and-butter ML use cases in e-commerce. This pipeline shows them implemented end-to-end with proper feature engineering, evaluation metrics, and MLOps quality gates.

## Pipeline Architecture

```text
BigQuery analytics.*  (produced by demo_data_ingestion_dagster_dbt dbt pipeline)
  ├── mart_customer_behaviour      ──► churn_feature_table        ──► churn_model_xgboost
  │                                └─► segmentation_feature_table ──► customer_segments_kmeans
  └── mart_demand_forecast_features──► demand_forecast_feature_table ──► demand_forecast_arima
                                                                              │
                                                                              ▼
                                                                   ml_evaluation_report
                                                               (Quality gates + Vertex AI promotion)
```

## Models

### 1. Churn Propensity — XGBoost Classifier

Predicts the probability that a customer will stop purchasing within 90 days.

**Features:**
| Feature | Description | Signal Direction |
|---------|-------------|-----------------|
| `days_since_last_purchase` | Recency signal | ↑ days = ↑ churn risk |
| `support_tickets` | Dissatisfaction signal | ↑ tickets = ↑ churn risk |
| `avg_review_rating` | Satisfaction signal | ↓ rating = ↑ churn risk |
| `order_count` | Engagement depth | ↓ orders = ↑ churn risk |
| `discount_code_usage_ratio` | Price sensitivity | ↑ ratio = ↑ churn risk |
| `session_count_30d` | Recency engagement | ↓ sessions = ↑ churn risk |

**Quality Gate:** AUC-ROC ≥ 0.70, F1-Score ≥ 0.55

**Production path:** XGBoost booster registered with Vertex AI Model Registry → served via Vertex AI Endpoint → integrated with Customer Support 360 demo for real-time at-risk flagging.

---

### 2. Customer Segmentation — K-Means (k=5)

Assigns each customer to one of five behavioural segments based on RFM (Recency, Frequency, Monetary) features.

**Segments:**
| Segment | Profile | Recommended Action |
|---------|---------|-------------------|
| High-Value Collector | High spend, high frequency | Exclusive previews, VIP events |
| Brand Loyalist | Frequent buyer across categories | Cross-sell new season arrivals |
| Value Seeker | Price-sensitive, regular visitor | Seasonal sale alerts, multi-buy offers |
| Gifting Shopper | Narrow categories, mid spend | Holiday gift guides, greeting cards |
| Casual Browser | Low frequency, low spend | Onboarding nurture, welcome coupons |

**Quality Gate:** Silhouette Score ≥ 0.30

**Production path:** BigQuery ML `CREATE OR REPLACE MODEL ... OPTIONS(model_type='kmeans')` trained on the mart table. Segment assignments written back to BigQuery and surfaced in the Recommender and Customer Support 360 demos.

---

### 3. Demand Forecasting — ARIMA+ (Seasonal)

Generates 30-day forward demand forecasts per product type with 95% prediction intervals.

**Training data:** Daily units sold per article over the trailing 90 days.
**Seasonality:** Weekly (Friday/Saturday weekend uplift for fashion categories).
**Production path:** BigQuery ML `ARIMA_PLUS` model with `holiday_region='GB'` for public holiday adjustments. Forecasts consumed by Supply Chain and Inventory teams.

**Quality Gate:** Average MAPE ≤ 30%

## Project Structure

```
demo_traditional_ml/
├── definitions.py          # ← Dagster entrypoint
├── resources.py            # BigQuery + Vertex AI resource configs
└── assets/
    ├── feature_engineering.py  # BigQuery mart → feature DataFrames
    ├── model_training.py        # XGBoost, K-Means, ARIMA+ training
    └── model_evaluation.py      # Quality gates + evaluation report
```

## Running Locally (DEMO_MODE)

```bash
# Install dependencies
pip install dagster dagster-webserver pandas numpy scikit-learn statsmodels

# Optional but recommended: install xgboost for the churn model
pip install xgboost

# Start Dagster UI — opens on http://localhost:3000
DEMO_MODE=true dagster dev -m demo_traditional_ml.definitions

# Or run the full pipeline directly
DEMO_MODE=true dagster asset materialize \
  -m demo_traditional_ml.definitions \
  --select '*'
```

## Running Against GCP (Production Mode)

Requires the upstream `demo_data_ingestion_dagster_dbt` pipeline to have populated `analytics.*` in BigQuery.

```bash
export GCP_PROJECT_ID="your-gcp-project"
export VERTEX_LOCATION="europe-west1"

DEMO_MODE=false dagster job execute \
  -m demo_traditional_ml.definitions \
  -j hm_ml_full_pipeline_job
```

## Dagster Jobs & Schedules

| Job | Trigger | Description |
|-----|---------|-------------|
| `hm_ml_full_pipeline_job` | Nightly @ 03:00 UTC (or ingestion sensor) | Feature engineering + all model training + evaluation |
| `hm_ml_feature_engineering_job` | Manual | BigQuery feature extraction only |
| `hm_ml_training_evaluation_job` | Manual / hyperparameter iteration | Training + evaluation (re-uses cached features) |

## MLOps Quality Gates

All three models must pass automated quality checks before promotion to Vertex AI Model Registry:

```
Churn XGBoost    AUC-ROC ≥ 0.70  ✅ / ❌
                 F1-Score ≥ 0.55  ✅ / ❌

K-Means          Silhouette ≥ 0.30  ✅ / ❌

ARIMA+           MAPE ≤ 30%  ✅ / ❌

── If ALL pass → PROMOTE to Vertex AI Model Registry
── If ANY fail → BLOCK promotion, alert ML Engineering team
```

## Upstream Dependency

This pipeline reads from the `analytics.*` dataset produced by **[Demo 5: Data Ingestion Pipeline](../demo_data_ingestion_dagster_dbt/README.md)**. The `ingestion_completion_sensor` automatically triggers ML training when a successful ingestion run is detected.

## Connections to Other Demos

| Demo | Integration |
|------|------------|
| [Customer Support 360](../demo_customer_support_360/README.md) | Churn propensity scores surfaced in agent context |
| [AI-Powered RAG](../demo_recommendation_engine/README.md) | K-Means segment drives personalised recommendation weights |
| [Instacart Knowledge Graph](../demo_instacart_knowledge_graph/README.md) | Graph embeddings as supplementary features for segmentation |
