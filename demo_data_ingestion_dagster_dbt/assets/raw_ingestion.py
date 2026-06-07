"""
Software-Defined Assets: Raw Data Ingestion Layer
==================================================

This module defines Dagster assets for the first stage of the H&M fashion
data pipeline: landing raw files from GCS into BigQuery's raw data layer.

Pipeline stage:  GCS (raw files)  ──►  BigQuery raw_data.*
Data sources:
  - Customer reviews   (JSON)   → raw_data.fashion_reviews
  - Transactions       (CSV)    → raw_data.fashion_transactions
  - Article metadata   (CSV)    → raw_data.fashion_articles

Each asset is independently materialisable, meaning Dagster can re-run
only the reviews asset without re-loading transactions — useful for partial
backfills after a source schema change.

The DEMO_MODE flag (env var) bypasses live GCP calls and emits realistic
metadata, allowing portfolio reviewers to run the Dagster UI locally.
"""
import os
import json
import logging
from datetime import datetime, timezone
from typing import Generator

import pandas as pd
from dagster import (
    asset,
    AssetExecutionContext,
    MetadataValue,
    Output,
    AssetKey,
)

from demo_data_ingestion.resources import BigQueryResource, GCSResource

logger = logging.getLogger(__name__)
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"


# ---------------------------------------------------------------------------
# Synthetic demo data generators
# ---------------------------------------------------------------------------

def _synthetic_reviews(n: int = 500) -> list[dict]:
    """Generate synthetic H&M customer review records for demo execution."""
    import random, uuid
    categories = ["Dresses", "Knitwear", "Trousers", "Accessories", "Outerwear"]
    adjectives = ["Absolutely love", "Great quality", "Disappointed with", "Pleasantly surprised by", "Average"]
    nouns = ["the fit", "the fabric", "the colour", "the sizing", "the stitching"]
    records = []
    for _ in range(n):
        cat = random.choice(categories)
        rating = random.choices([1, 2, 3, 4, 5], weights=[5, 8, 15, 35, 37])[0]
        records.append({
            "review_id": str(uuid.uuid4()),
            "customer_id": f"C{random.randint(10000, 99999)}",
            "article_id": f"A{random.randint(100000, 999999)}",
            "category": cat,
            "rating": rating,
            "body": f"{random.choice(adjectives)} {random.choice(nouns)}. "
                    f"Would {'recommend' if rating >= 4 else 'not recommend'}.",
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        })
    return records


def _synthetic_transactions(n: int = 2000) -> list[dict]:
    """Generate synthetic H&M transaction records for demo execution."""
    import random, uuid
    channels = ["online", "in-store", "mobile_app"]
    records = []
    for _ in range(n):
        price = round(random.uniform(4.99, 299.99), 2)
        records.append({
            "transaction_id": str(uuid.uuid4()),
            "customer_id": f"C{random.randint(10000, 99999)}",
            "article_id": f"A{random.randint(100000, 999999)}",
            "price": price,
            "sales_channel": random.choice(channels),
            "t_dat": (datetime.now(timezone.utc)).strftime("%Y-%m-%d"),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        })
    return records


def _synthetic_articles(n: int = 300) -> list[dict]:
    """Generate synthetic H&M article metadata records for demo execution."""
    import random
    product_types = ["Dress", "Blouse", "Trousers", "Jacket", "Skirt", "Knitwear", "Accessories"]
    colours = ["Black", "White", "Navy", "Beige", "Grey", "Red", "Green", "Blue", "Burgundy"]
    departments = ["Womens", "Mens", "Kids", "Sport", "Home"]
    records = []
    seen_ids = set()
    while len(records) < n:
        article_id = f"A{100000 + len(records)}"
        if article_id in seen_ids:
            continue
        seen_ids.add(article_id)
        records.append({
            "article_id": article_id,
            "product_name": f"H&M {random.choice(colours)} {random.choice(product_types)}",
            "product_type": random.choice(product_types),
            "colour": random.choice(colours),
            "department": random.choice(departments),
            "detail_desc": f"Premium quality {random.choice(product_types).lower()} in {random.choice(colours).lower()}.",
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        })
    return records


# ---------------------------------------------------------------------------
# Asset 1: Raw Customer Reviews
# ---------------------------------------------------------------------------

@asset(
    group_name="raw_ingestion",
    key_prefix=["hm_fashion", "raw"],
    description=(
        "Ingests H&M customer review JSON files from GCS into BigQuery "
        "`raw_data.fashion_reviews`. Validates schema and logs row counts."
    ),
    metadata={"source": "gs://hm-fashion-raw-data/reviews/", "target_table": "raw_data.fashion_reviews"},
)
def raw_fashion_reviews(
    context: AssetExecutionContext,
    bigquery: BigQueryResource,
    gcs: GCSResource,
) -> Output[pd.DataFrame]:
    """
    Load raw customer review records from GCS into BigQuery.

    Steps:
      1. List objects under `reviews/` prefix in the configured GCS bucket.
      2. Download and parse JSON review files into a DataFrame.
      3. Validate required columns are present.
      4. Write to BigQuery `raw_data.fashion_reviews` via load_table_from_dataframe.
      5. Emit row count and schema as Dagster materialisation metadata.

    In DEMO_MODE the GCS/BQ calls are skipped and synthetic data is returned.
    """
    context.log.info(f"DEMO_MODE={DEMO_MODE}")
    context.log.info(f"Source: {gcs.gcs_uri(gcs.reviews_prefix)}")
    context.log.info(f"Target: {bigquery.full_table_id(bigquery.dataset_raw, 'fashion_reviews')}")

    if DEMO_MODE:
        context.log.info("Running in DEMO_MODE — generating synthetic review data")
        records = _synthetic_reviews(500)
        df = pd.DataFrame(records)
        context.log.info(f"Generated {len(df)} synthetic review records")
    else:
        # --- Production path ---
        gcs_client = gcs.get_client()
        bucket = gcs_client.bucket(gcs.bucket_name)
        blobs = list(bucket.list_blobs(prefix=gcs.reviews_prefix))
        context.log.info(f"Found {len(blobs)} review files in GCS")

        records = []
        for blob in blobs:
            raw = blob.download_as_text()
            records.extend(json.loads(raw))

        df = pd.DataFrame(records)

        # Schema validation
        required = {"review_id", "customer_id", "article_id", "rating", "body"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Schema validation failed — missing columns: {missing}")

        # Write to BigQuery
        bq_client = bigquery.get_client()
        table_id = bigquery.full_table_id(bigquery.dataset_raw, "fashion_reviews")
        job = bq_client.load_table_from_dataframe(df, table_id)
        job.result()  # Block until complete
        context.log.info(f"Loaded {len(df)} rows into {table_id}")

    return Output(
        value=df,
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "columns": MetadataValue.text(", ".join(df.columns.tolist())),
            "target_table": MetadataValue.text(
                bigquery.full_table_id(bigquery.dataset_raw, "fashion_reviews")
            ),
            "demo_mode": MetadataValue.bool(DEMO_MODE),
            "ingested_at": MetadataValue.text(datetime.now(timezone.utc).isoformat()),
        },
    )


# ---------------------------------------------------------------------------
# Asset 2: Raw Transactions
# ---------------------------------------------------------------------------

@asset(
    group_name="raw_ingestion",
    key_prefix=["hm_fashion", "raw"],
    description=(
        "Ingests H&M transaction CSV files from GCS into BigQuery "
        "`raw_data.fashion_transactions`. One row per purchase event."
    ),
    metadata={"source": "gs://hm-fashion-raw-data/transactions/", "target_table": "raw_data.fashion_transactions"},
)
def raw_fashion_transactions(
    context: AssetExecutionContext,
    bigquery: BigQueryResource,
    gcs: GCSResource,
) -> Output[pd.DataFrame]:
    """
    Load raw transaction records from GCS into BigQuery.

    Steps:
      1. List objects under `transactions/` prefix.
      2. Download and concatenate CSV files into a DataFrame.
      3. Validate required columns (transaction_id, customer_id, article_id, price, t_dat).
      4. Cast `price` to FLOAT64, `t_dat` to DATE.
      5. Write to BigQuery `raw_data.fashion_transactions`.
    """
    context.log.info(f"Source: {gcs.gcs_uri(gcs.transactions_prefix)}")
    context.log.info(f"Target: {bigquery.full_table_id(bigquery.dataset_raw, 'fashion_transactions')}")

    if DEMO_MODE:
        context.log.info("Running in DEMO_MODE — generating synthetic transaction data")
        records = _synthetic_transactions(2000)
        df = pd.DataFrame(records)
        context.log.info(f"Generated {len(df)} synthetic transaction records")
    else:
        gcs_client = gcs.get_client()
        bucket = gcs_client.bucket(gcs.bucket_name)
        blobs = list(bucket.list_blobs(prefix=gcs.transactions_prefix))
        context.log.info(f"Found {len(blobs)} transaction CSV files in GCS")

        frames = []
        for blob in blobs:
            from io import StringIO
            raw = blob.download_as_text()
            frames.append(pd.read_csv(StringIO(raw)))
        df = pd.concat(frames, ignore_index=True)

        required = {"transaction_id", "customer_id", "article_id", "price", "t_dat"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Schema validation failed — missing columns: {missing}")

        df["price"] = df["price"].astype(float)
        df["t_dat"] = pd.to_datetime(df["t_dat"]).dt.date

        bq_client = bigquery.get_client()
        table_id = bigquery.full_table_id(bigquery.dataset_raw, "fashion_transactions")
        job = bq_client.load_table_from_dataframe(df, table_id)
        job.result()
        context.log.info(f"Loaded {len(df)} rows into {table_id}")

    return Output(
        value=df,
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "total_revenue": MetadataValue.float(round(df["price"].sum(), 2)),
            "date_range": MetadataValue.text(
                f"{df['t_dat'].min()} → {df['t_dat'].max()}"
            ),
            "target_table": MetadataValue.text(
                bigquery.full_table_id(bigquery.dataset_raw, "fashion_transactions")
            ),
            "demo_mode": MetadataValue.bool(DEMO_MODE),
            "ingested_at": MetadataValue.text(datetime.now(timezone.utc).isoformat()),
        },
    )


# ---------------------------------------------------------------------------
# Asset 3: Raw Article Metadata
# ---------------------------------------------------------------------------

@asset(
    group_name="raw_ingestion",
    key_prefix=["hm_fashion", "raw"],
    description=(
        "Ingests H&M article (product) metadata from GCS into BigQuery "
        "`raw_data.fashion_articles`. Provides product catalogue dimension data."
    ),
    metadata={"source": "gs://hm-fashion-raw-data/articles/", "target_table": "raw_data.fashion_articles"},
)
def raw_fashion_articles(
    context: AssetExecutionContext,
    bigquery: BigQueryResource,
    gcs: GCSResource,
) -> Output[pd.DataFrame]:
    """
    Load H&M article/product metadata from GCS into BigQuery.

    The articles table acts as the product dimension, joined downstream by
    the dbt models to enrich transaction and review fact tables.
    """
    context.log.info(f"Source: {gcs.gcs_uri(gcs.articles_prefix)}")
    context.log.info(f"Target: {bigquery.full_table_id(bigquery.dataset_raw, 'fashion_articles')}")

    if DEMO_MODE:
        context.log.info("Running in DEMO_MODE — generating synthetic article data")
        records = _synthetic_articles(300)
        df = pd.DataFrame(records)
        context.log.info(f"Generated {len(df)} synthetic article records")
    else:
        gcs_client = gcs.get_client()
        bucket = gcs_client.bucket(gcs.bucket_name)
        blobs = list(bucket.list_blobs(prefix=gcs.articles_prefix))

        from io import StringIO
        frames = [pd.read_csv(StringIO(b.download_as_text())) for b in blobs]
        df = pd.concat(frames, ignore_index=True)

        bq_client = bigquery.get_client()
        table_id = bigquery.full_table_id(bigquery.dataset_raw, "fashion_articles")
        job = bq_client.load_table_from_dataframe(df, table_id)
        job.result()
        context.log.info(f"Loaded {len(df)} rows into {table_id}")

    return Output(
        value=df,
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "product_types": MetadataValue.text(
                ", ".join(df["product_type"].unique().tolist()) if "product_type" in df.columns else "N/A"
            ),
            "target_table": MetadataValue.text(
                bigquery.full_table_id(bigquery.dataset_raw, "fashion_articles")
            ),
            "demo_mode": MetadataValue.bool(DEMO_MODE),
        },
    )
