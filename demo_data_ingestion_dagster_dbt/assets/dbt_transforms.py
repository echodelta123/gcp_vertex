"""
Software-Defined Assets: dbt Transformation Layer
==================================================

This module defines the Dagster asset that triggers dbt model compilation
and testing on top of the raw BigQuery tables populated in the previous
pipeline stage.

Pipeline stage:  BigQuery raw_data.*  ──►  dbt  ──►  BigQuery analytics.*

dbt models compiled by this asset:
  - marts.fashion.mart_customer_behaviour  — enriched customer-level
      aggregations (total spend, order frequency, avg basket size).
  - marts.fashion.mart_aspect_sentiments   — NLP-derived aspect sentiment
      scores aggregated per article and product category.
  - marts.fashion.mart_demand_forecast_features — time-series demand
      feature table used by the Traditional ML pipeline downstream.

The asset takes an upstream dependency on all three raw ingestion assets
to enforce ordering: dbt cannot run until the raw data is present.
"""
import os
import subprocess
import logging
from datetime import datetime, timezone

import pandas as pd
from dagster import (
    asset,
    AssetExecutionContext,
    AssetIn,
    MetadataValue,
    Output,
)

from demo_data_ingestion.resources import dbt_resource, BigQueryResource
from dagster_dbt import DbtCliResource

logger = logging.getLogger(__name__)
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"


# ---------------------------------------------------------------------------
# dbt mart models (for documentation purposes)
# ---------------------------------------------------------------------------

DBT_MODELS = [
    "marts.fashion.mart_customer_behaviour",
    "marts.fashion.mart_aspect_sentiments",
    "marts.fashion.mart_demand_forecast_features",
]

DBT_TESTS = [
    "not_null:mart_customer_behaviour.customer_id",
    "unique:mart_customer_behaviour.customer_id",
    "accepted_range:mart_aspect_sentiments.sentiment_score [-1, 1]",
    "not_null:mart_demand_forecast_features.article_id",
]


def _run_dbt_command(args: list[str], project_dir: str, profiles_dir: str) -> str:
    """
    Execute a dbt CLI command as a subprocess.

    Returns the combined stdout/stderr output for logging.
    Raises RuntimeError if the command exits with a non-zero code.
    """
    cmd = ["dbt"] + args + [
        "--project-dir", project_dir,
        "--profiles-dir", profiles_dir,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"dbt command failed (exit {result.returncode}):\n{output}"
        )
    return output


# ---------------------------------------------------------------------------
# Asset: dbt Transformation Run
# ---------------------------------------------------------------------------

@asset(
    group_name="dbt_transforms",
    key_prefix=["hm_fashion", "analytics"],
    ins={
        "raw_reviews": AssetIn(key=["hm_fashion", "raw", "raw_fashion_reviews"]),
        "raw_transactions": AssetIn(key=["hm_fashion", "raw", "raw_fashion_transactions"]),
        "raw_articles": AssetIn(key=["hm_fashion", "raw", "raw_fashion_articles"]),
    },
    description=(
        "Executes dbt `marts.fashion` model suite to compile analytical mart tables "
        "from raw BigQuery landing tables. Runs schema tests and emits model-level "
        "row counts and test pass rates as Dagster metadata."
    ),
    metadata={
        "dbt_models": str(DBT_MODELS),
        "dbt_tests": str(DBT_TESTS),
        "target_dataset": "analytics",
    },
)
def dbt_fashion_marts(
    context: AssetExecutionContext,
    dbt: DbtCliResource,
    bigquery: BigQueryResource,
    raw_reviews: pd.DataFrame,
    raw_transactions: pd.DataFrame,
    raw_articles: pd.DataFrame,
) -> Output[dict]:
    """
    Run dbt transformations to compile the H&M analytics mart layer.

    Execution steps:
      1. `dbt deps`   — Install/refresh dbt package dependencies.
      2. `dbt run`    — Compile and materialise mart models in BigQuery.
      3. `dbt test`   — Run schema constraints and custom data tests.
      4. Emit model row counts and test results as Dagster metadata.

    Upstream dependencies:
      - raw_fashion_reviews      (provides the review dimension data)
      - raw_fashion_transactions (provides the sales fact data)
      - raw_fashion_articles     (provides the product dimension data)

    The downstream `demo_traditional_ml` pipeline reads from these mart
    tables to train and evaluate BigQuery ML models.
    """
    context.log.info(
        f"Upstream row counts — reviews: {len(raw_reviews)}, "
        f"transactions: {len(raw_transactions)}, articles: {len(raw_articles)}"
    )
    context.log.info(f"dbt project dir: {dbt.project_dir}")
    context.log.info(f"dbt target: {dbt.target}\n",
        f"dbt models: {', '.join(DBT_MODELS)}\n")

    if DEMO_MODE:
        context.log.info("Running in DEMO_MODE — simulating dbt execution")

        # Simulate dbt mart compilation from in-memory DataFrames
        customer_behaviour = _compute_customer_behaviour(raw_transactions, raw_reviews)
        aspect_sentiments = _compute_aspect_sentiments(raw_reviews, raw_articles)
        demand_features = _compute_demand_features(raw_transactions, raw_articles)

        results = {
            "mart_customer_behaviour": customer_behaviour,
            "mart_aspect_sentiments": aspect_sentiments,
            "mart_demand_forecast_features": demand_features,
        }

        context.log.info(
            f"Compiled {len(customer_behaviour)} customer behaviour rows, "
            f"{len(aspect_sentiments)} sentiment rows, "
            f"{len(demand_features)} demand feature rows"
        )

    else:
        # --- Production path ---
        context.log.info("Installing dbt package dependencies...")
        deps_output = _run_dbt_command(["deps"], dbt.project_dir, dbt.profiles_dir)
        context.log.info(deps_output)

        context.log.info(f"Running dbt models: {dbt.select}")
        run_output = _run_dbt_command(
            ["run", "--select", dbt.select, "--target", dbt.target],
            dbt.project_dir, dbt.profiles_dir,
        )
        context.log.info(run_output)

        context.log.info("Running dbt schema tests...")
        test_output = _run_dbt_command(
            ["test", "--select", dbt.select, "--target", dbt.target],
            dbt.project_dir, dbt.profiles_dir,
        )
        context.log.info(test_output)

        # Query BigQuery for actual row counts
        bq_client = bigquery.get_client()
        results = {}
        for model in ["mart_customer_behaviour", "mart_aspect_sentiments", "mart_demand_forecast_features"]:
            table_id = bigquery.full_table_id(bigquery.dataset_marts, model)
            count_query = f"SELECT COUNT(*) as row_count FROM `{table_id}`"
            row = list(bq_client.query(count_query).result())[0]
            results[model] = row["row_count"]

    return Output(
        value={k: (v if not isinstance(v, pd.DataFrame) else len(v)) for k, v in results.items()},
        metadata={
            "mart_customer_behaviour_rows": MetadataValue.int(
                len(results["mart_customer_behaviour"]) if isinstance(results["mart_customer_behaviour"], pd.DataFrame) else results["mart_customer_behaviour"]
            ),
            "mart_aspect_sentiments_rows": MetadataValue.int(
                len(results["mart_aspect_sentiments"]) if isinstance(results["mart_aspect_sentiments"], pd.DataFrame) else results["mart_aspect_sentiments"]
            ),
            "mart_demand_features_rows": MetadataValue.int(
                len(results["mart_demand_forecast_features"]) if isinstance(results["mart_demand_forecast_features"], pd.DataFrame) else results["mart_demand_forecast_features"]
            ),
            "dbt_models_run": MetadataValue.int(len(DBT_MODELS)),
            "dbt_tests_run": MetadataValue.int(len(DBT_TESTS)),
            "demo_mode": MetadataValue.bool(DEMO_MODE),
            "completed_at": MetadataValue.text(datetime.now(timezone.utc).isoformat()),
        },
    )


# ---------------------------------------------------------------------------
# In-memory dbt mart emulation (DEMO_MODE only)
# ---------------------------------------------------------------------------

def _compute_customer_behaviour(
    transactions: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulate `mart_customer_behaviour` dbt model logic.

    Equivalent SQL (simplified):
        SELECT
            customer_id,
            COUNT(DISTINCT transaction_id)  AS order_count,
            SUM(price)                      AS total_spend,
            AVG(price)                      AS avg_order_value,
            MAX(t_dat)                      AS last_purchase_date,
            AVG(r.rating)                   AS avg_review_rating
        FROM raw_data.fashion_transactions t
        LEFT JOIN raw_data.fashion_reviews r USING (customer_id)
        GROUP BY 1
    """
    tx_agg = (
        transactions.groupby("customer_id")
        .agg(
            order_count=("transaction_id", "count"),
            total_spend=("price", "sum"),
            avg_order_value=("price", "mean"),
            last_purchase_date=("t_dat", "max"),
        )
        .reset_index()
    )

    if "customer_id" in reviews.columns and "rating" in reviews.columns:
        rev_agg = reviews.groupby("customer_id")["rating"].mean().reset_index()
        rev_agg.columns = ["customer_id", "avg_review_rating"]
        tx_agg = tx_agg.merge(rev_agg, on="customer_id", how="left")

    tx_agg["total_spend"] = tx_agg["total_spend"].round(2)
    tx_agg["avg_order_value"] = tx_agg["avg_order_value"].round(2)
    return tx_agg


def _compute_aspect_sentiments(
    reviews: pd.DataFrame,
    articles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulate `mart_aspect_sentiments` dbt model logic.

    Uses a simple keyword heuristic to assign sentiment scores — in
    production this table is populated by a Cloud Run NLP pipeline
    that runs VADER or a fine-tuned BERT model against review text.
    """
    import re

    positive_words = {"love", "great", "excellent", "perfect", "recommend", "amazing"}
    negative_words = {"disappointed", "poor", "broken", "terrible", "bad", "worst"}

    def sentiment_score(text: str) -> float:
        words = set(re.findall(r"\b\w+\b", str(text).lower()))
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        total = pos + neg
        return round((pos - neg) / total, 3) if total > 0 else 0.0

    reviews = reviews.copy()
    reviews["sentiment_score"] = reviews["body"].apply(sentiment_score)

    if "article_id" in reviews.columns:
        agg = (
            reviews.groupby("article_id")
            .agg(
                avg_sentiment=("sentiment_score", "mean"),
                review_count=("review_id", "count"),
                avg_rating=("rating", "mean"),
            )
            .reset_index()
        )
    else:
        agg = pd.DataFrame(columns=["article_id", "avg_sentiment", "review_count", "avg_rating"])

    if "article_id" in articles.columns:
        agg = agg.merge(
            articles[["article_id", "product_type", "category" if "category" in articles.columns else "product_type"]],
            on="article_id",
            how="left",
        )

    return agg


def _compute_demand_features(
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulate `mart_demand_forecast_features` dbt model logic.

    Aggregates daily sales volumes per article — this is the feature table
    consumed by the Traditional ML ARIMA+ demand forecasting pipeline.
    """
    tx = transactions.copy()
    tx["t_dat"] = pd.to_datetime(tx["t_dat"]).dt.date

    if "article_id" in tx.columns:
        agg = (
            tx.groupby(["article_id", "t_dat"])
            .agg(daily_units_sold=("transaction_id", "count"), daily_revenue=("price", "sum"))
            .reset_index()
        )
        if "article_id" in articles.columns:
            agg = agg.merge(
                articles[["article_id", "product_type"]],
                on="article_id",
                how="left",
            )
    else:
        agg = pd.DataFrame()

    return agg
