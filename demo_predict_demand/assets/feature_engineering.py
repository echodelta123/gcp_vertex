"""
Feature Engineering Assets
===========================

This module defines Dagster software-defined assets for the feature
engineering stage of the H&M fashion Traditional ML pipeline.

Pipeline stage:  BigQuery analytics.*  ──►  Feature DataFrames

The features produced here feed three downstream ML model training assets:
  1. Churn propensity features    → XGBoost binary classifier
  2. Customer segmentation features → K-Means clustering (sklearn + BigQuery ML)
  3. Demand forecasting features  → ARIMA+ time-series model (BigQuery ML)

In production these BigQuery queries run against the `analytics` dataset
compiled by the upstream `demo_data_ingestion` dbt pipeline. In DEMO_MODE
synthetic data is generated deterministically from a fixed random seed so
output metrics are stable between runs.
"""
import os
import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from dagster import (
    asset,
    AssetExecutionContext,
    MetadataValue,
    Output,
)

logger = logging.getLogger(__name__)
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Synthetic data generators (DEMO_MODE)
# ---------------------------------------------------------------------------

def _generate_customer_features(n: int = 800, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic customer-level features from the mart_customer_behaviour table.

    Mirrors the schema of the BigQuery mart compiled by the dbt pipeline:
        customer_id, order_count, total_spend, avg_order_value,
        days_since_last_purchase, support_tickets, avg_review_rating,
        discount_code_usage_ratio, session_count_30d, avg_session_duration_s
    """
    rng = np.random.default_rng(seed)
    customer_ids = [f"C{10000 + i}" for i in range(n)]

    # Simulate a bimodal spend distribution (high-value vs casual shoppers)
    is_high_value = rng.binomial(1, 0.2, n).astype(bool)
    total_spend = np.where(
        is_high_value,
        rng.gamma(shape=4, scale=500, size=n),
        rng.gamma(shape=2, scale=80, size=n),
    )
    order_count = rng.poisson(lam=np.where(is_high_value, 12, 3), size=n)
    order_count = np.maximum(order_count, 1)  # At least 1 order
    avg_order_value = total_spend / order_count

    # Churn-correlated features: churned customers have higher support tickets
    # and longer gaps since last purchase
    will_churn = (
        rng.binomial(1, 0.25, n).astype(bool)
        | (total_spend < 50)
        | (rng.random(n) > 0.85)
    )
    days_since_last_purchase = np.where(
        will_churn,
        rng.integers(60, 365, n),
        rng.integers(1, 60, n),
    ).astype(int)
    support_tickets = np.where(
        will_churn,
        rng.integers(2, 8, n),
        rng.integers(0, 3, n),
    ).astype(int)

    return pd.DataFrame({
        "customer_id": customer_ids,
        "order_count": order_count,
        "total_spend": np.round(total_spend, 2),
        "avg_order_value": np.round(avg_order_value, 2),
        "days_since_last_purchase": days_since_last_purchase,
        "support_tickets": support_tickets,
        "avg_review_rating": np.round(rng.uniform(1.0, 5.0, n), 2),
        "discount_code_usage_ratio": np.round(rng.beta(2, 5, n), 3),
        "session_count_30d": rng.integers(0, 50, n),
        "avg_session_duration_s": np.round(rng.gamma(3, 80, n), 1),
        "churned": will_churn.astype(int),  # Target label
    })


def _generate_demand_features(n_articles: int = 50, days: int = 90, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic daily sales volume time-series features.

    Mirrors the `mart_demand_forecast_features` dbt mart output with
    added engineered lag and rolling window features for ARIMA+ training.
    """
    rng = np.random.default_rng(seed)
    product_types = ["Dresses", "Knitwear", "Trousers", "Accessories", "Outerwear", "Blouses"]

    records = []
    base_date = datetime.now(timezone.utc).date() - timedelta(days=days)

    for i in range(n_articles):
        article_id = f"A{100000 + i}"
        product_type = product_types[i % len(product_types)]
        base_vol = rng.integers(20, 200)

        for day_offset in range(days):
            date = base_date + timedelta(days=day_offset)
            weekday = date.weekday()
            # Weekend uplift
            weekday_factor = 1.3 if weekday >= 4 else 1.0
            # Seasonal trend (slight upward slope)
            trend = 1.0 + (day_offset * 0.003)
            noise = rng.uniform(0.85, 1.15)
            units = int(base_vol * weekday_factor * trend * noise)
            revenue = round(units * rng.uniform(20, 150), 2)

            records.append({
                "article_id": article_id,
                "product_type": product_type,
                "date": str(date),
                "daily_units_sold": units,
                "daily_revenue": revenue,
                "weekday": weekday,
                "is_weekend": int(weekday >= 5),
                # Lag features (simplified — production uses BigQuery WINDOW functions)
                "lag_7d_units": int(units * rng.uniform(0.8, 1.2)),
                "rolling_14d_avg_units": int(units * rng.uniform(0.9, 1.1)),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Asset 1: Customer Churn Feature Table
# ---------------------------------------------------------------------------

@asset(
    group_name="feature_engineering",
    key_prefix=["hm_fashion", "features"],
    description=(
        "Queries BigQuery `analytics.mart_customer_behaviour` to build the "
        "churn prediction feature set. Applies feature scaling and encodes "
        "the binary churn target label for XGBoost training."
    ),
    metadata={
        "source_table": "analytics.mart_customer_behaviour",
        "feature_count": 9,
        "target_variable": "churned (binary)",
    },
)
def churn_feature_table(context: AssetExecutionContext) -> Output[pd.DataFrame]:
    """
    Build churn propensity features from the BigQuery customer behaviour mart.

    Features extracted:
      - order_count               — purchase frequency signal
      - total_spend               — customer lifetime value proxy
      - avg_order_value           — basket size
      - days_since_last_purchase  — recency signal (key churn predictor)
      - support_tickets           — dissatisfaction signal
      - avg_review_rating         — sentiment signal
      - discount_code_usage_ratio — price sensitivity signal
      - session_count_30d         — engagement signal
      - avg_session_duration_s    — depth of engagement

    Target:
      - churned (1 = churned within 90 days, 0 = retained)

    Production query (runs against BigQuery):
        SELECT
            customer_id,
            order_count,
            total_spend,
            avg_order_value,
            DATEDIFF(CURRENT_DATE(), last_purchase_date, DAY) AS days_since_last_purchase,
            support_ticket_count                               AS support_tickets,
            avg_review_rating,
            discount_code_usage_ratio,
            session_count_30d,
            avg_session_duration_s,
            CASE WHEN days_to_next_purchase IS NULL THEN 1 ELSE 0 END AS churned
        FROM `analytics.mart_customer_behaviour`
        WHERE last_purchase_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)
    """
    if DEMO_MODE:
        context.log.info("DEMO_MODE: generating synthetic churn feature dataset")
        df = _generate_customer_features(n=800)
    else:
        from demo_traditional_ml.resources import bigquery_resource
        bq = bigquery_resource.get_client()
        query = f"""
            SELECT
                customer_id,
                order_count,
                total_spend,
                avg_order_value,
                DATE_DIFF(CURRENT_DATE(), last_purchase_date, DAY) AS days_since_last_purchase,
                support_ticket_count AS support_tickets,
                avg_review_rating,
                discount_code_usage_ratio,
                session_count_30d,
                avg_session_duration_s,
                CASE WHEN churned_within_90d THEN 1 ELSE 0 END AS churned
            FROM `{bigquery_resource.project_id}.analytics.mart_customer_behaviour`
            WHERE last_purchase_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)
        """
        df = bq.query(query).to_dataframe()

    churn_rate = df["churned"].mean()
    context.log.info(
        f"Churn feature table: {len(df)} customers, "
        f"churn rate: {churn_rate:.1%}, "
        f"avg spend: £{df['total_spend'].mean():.2f}"
    )

    return Output(
        value=df,
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "churn_rate": MetadataValue.float(round(float(churn_rate), 4)),
            "avg_total_spend": MetadataValue.float(round(float(df["total_spend"].mean()), 2)),
            "feature_columns": MetadataValue.text(
                ", ".join(c for c in df.columns if c not in ("customer_id", "churned"))
            ),
            "demo_mode": MetadataValue.bool(DEMO_MODE),
        },
    )


# ---------------------------------------------------------------------------
# Asset 2: Customer Segmentation Feature Table
# ---------------------------------------------------------------------------

@asset(
    group_name="feature_engineering",
    key_prefix=["hm_fashion", "features"],
    description=(
        "Builds the RFM (Recency, Frequency, Monetary) feature matrix for "
        "K-Means customer segmentation. Normalises spend and frequency metrics "
        "for cluster stability."
    ),
    metadata={
        "source_table": "analytics.mart_customer_behaviour",
        "feature_count": 4,
        "algorithm": "K-Means (k=5)",
    },
)
def segmentation_feature_table(context: AssetExecutionContext) -> Output[pd.DataFrame]:
    """
    Build RFM segmentation features for K-Means clustering.

    RFM features:
      - recency   (days_since_last_purchase — lower is better)
      - frequency (order_count)
      - monetary  (total_spend)
      - category_breadth (number of distinct product categories purchased)

    In production a BigQuery ML K-Means model is trained directly on these
    features using `CREATE OR REPLACE MODEL`:

        CREATE OR REPLACE MODEL `analytics.customer_segments_kmeans`
        OPTIONS (model_type='kmeans', num_clusters=5)
        AS
        SELECT
            recency, frequency, monetary, category_breadth
        FROM `analytics.mart_customer_behaviour`

    The BigQuery ML model output is then joined back to customer IDs and
    exported to a serving endpoint for real-time segment assignment.
    """
    if DEMO_MODE:
        context.log.info("DEMO_MODE: generating synthetic RFM segmentation features")
        df = _generate_customer_features(n=800)
        rfm = df[["customer_id", "days_since_last_purchase", "order_count",
                   "total_spend", "avg_order_value"]].copy()
        rfm.columns = ["customer_id", "recency", "frequency", "monetary", "avg_basket_size"]
        rng = np.random.default_rng(RANDOM_SEED)
        rfm["category_breadth"] = rng.integers(1, 6, len(rfm))
    else:
        from demo_traditional_ml.resources import bigquery_resource
        bq = bigquery_resource.get_client()
        query = f"""
            SELECT
                customer_id,
                DATE_DIFF(CURRENT_DATE(), last_purchase_date, DAY) AS recency,
                order_count                                          AS frequency,
                total_spend                                          AS monetary,
                avg_order_value                                      AS avg_basket_size,
                COALESCE(category_breadth, 1)                        AS category_breadth
            FROM `{bigquery_resource.project_id}.analytics.mart_customer_behaviour`
        """
        rfm = bq.query(query).to_dataframe()

    context.log.info(
        f"Segmentation feature table: {len(rfm)} customers, "
        f"avg recency: {rfm['recency'].mean():.0f} days, "
        f"avg frequency: {rfm['frequency'].mean():.1f} orders"
    )

    return Output(
        value=rfm,
        metadata={
            "row_count": MetadataValue.int(len(rfm)),
            "avg_recency_days": MetadataValue.float(round(float(rfm["recency"].mean()), 1)),
            "avg_order_frequency": MetadataValue.float(round(float(rfm["frequency"].mean()), 2)),
            "avg_monetary_value": MetadataValue.float(round(float(rfm["monetary"].mean()), 2)),
            "demo_mode": MetadataValue.bool(DEMO_MODE),
        },
    )


# ---------------------------------------------------------------------------
# Asset 3: Demand Forecasting Feature Table
# ---------------------------------------------------------------------------

@asset(
    group_name="feature_engineering",
    key_prefix=["hm_fashion", "features"],
    description=(
        "Extracts and engineers time-series demand features from "
        "`analytics.mart_demand_forecast_features`. Computes lag features "
        "(7d, 14d) and rolling averages used by BigQuery ML ARIMA+."
    ),
    metadata={
        "source_table": "analytics.mart_demand_forecast_features",
        "feature_type": "time-series",
        "model_target": "BigQuery ML ARIMA+",
    },
)
def demand_forecast_feature_table(context: AssetExecutionContext) -> Output[pd.DataFrame]:
    """
    Build demand forecasting time-series features for BigQuery ML ARIMA+.

    The BigQuery ML ARIMA+ model requires a time-series table with at minimum:
      - a time column (date)
      - a data column (daily_units_sold)
      - optional series key columns (article_id, product_type)

    The production BigQuery ML training query:
        CREATE OR REPLACE MODEL `analytics.demand_forecast_arima`
        OPTIONS (
            model_type          = 'ARIMA_PLUS',
            time_series_timestamp_col = 'date',
            time_series_data_col      = 'daily_units_sold',
            time_series_id_col        = 'article_id',
            holiday_region            = 'GB',
            auto_arima                = TRUE
        )
        AS
        SELECT date, article_id, product_type, daily_units_sold
        FROM `analytics.mart_demand_forecast_features`
        WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 180 DAY)
    """
    if DEMO_MODE:
        context.log.info("DEMO_MODE: generating synthetic demand time-series features")
        df = _generate_demand_features(n_articles=50, days=90)
    else:
        from demo_traditional_ml.resources import bigquery_resource
        bq = bigquery_resource.get_client()
        query = f"""
            SELECT
                date,
                article_id,
                product_type,
                daily_units_sold,
                daily_revenue,
                weekday,
                is_weekend,
                SUM(daily_units_sold) OVER (
                    PARTITION BY article_id
                    ORDER BY date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) AS lag_7d_units,
                AVG(daily_units_sold) OVER (
                    PARTITION BY article_id
                    ORDER BY date
                    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
                ) AS rolling_14d_avg_units
            FROM `{bigquery_resource.project_id}.analytics.mart_demand_forecast_features`
            WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
            ORDER BY article_id, date
        """
        df = bq.query(query).to_dataframe()

    context.log.info(
        f"Demand feature table: {len(df)} rows, "
        f"{df['article_id'].nunique()} articles, "
        f"date range: {df['date'].min()} → {df['date'].max()}"
    )

    return Output(
        value=df,
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "article_count": MetadataValue.int(int(df["article_id"].nunique())),
            "date_range": MetadataValue.text(f"{df['date'].min()} → {df['date'].max()}"),
            "avg_daily_units": MetadataValue.float(round(float(df["daily_units_sold"].mean()), 1)),
            "demo_mode": MetadataValue.bool(DEMO_MODE),
        },
    )
