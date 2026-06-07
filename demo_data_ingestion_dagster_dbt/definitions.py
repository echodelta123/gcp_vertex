"""
Dagster Definitions: H&M Fashion Data Ingestion Pipeline
=========================================================

This is the top-level entry point for the Dagster project.

`Definitions` is the single object Dagster reads to discover all assets,
jobs, sensors, schedules, and resources. Point `dagster dev` at this
module and everything below is wired up automatically.

Usage:
    # Start the Dagster UI (Dagit) locally
    DEMO_MODE=true dagster dev -m demo_data_ingestion.definitions

    # Materialise all assets in one-shot (no daemon required)
    DEMO_MODE=true dagster asset materialize -m demo_data_ingestion.definitions --select '*'

    # Run the ingestion job directly
    DEMO_MODE=true dagster job execute -m demo_data_ingestion.definitions -j hm_fashion_ingestion_job
"""
from dagster import (
    Definitions,
    define_asset_job,
    AssetSelection,
    ScheduleDefinition,
)

from demo_data_ingestion.assets import (
    raw_fashion_reviews,
    raw_fashion_transactions,
    raw_fashion_articles,
    dbt_fashion_marts,
)
from demo_data_ingestion.sensors import gcs_new_file_sensor
from demo_data_ingestion.resources import bigquery_resource, gcs_resource, dbt_resource


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

hm_fashion_ingestion_job = define_asset_job(
    name="hm_fashion_ingestion_job",
    selection=AssetSelection.groups("raw_ingestion", "dbt_transforms"),
    description=(
        "Materialises all H&M fashion data ingestion assets in dependency order: "
        "raw GCS ingestion (reviews, transactions, articles) followed by dbt mart "
        "compilation. Triggered by the GCS sensor or on a nightly schedule."
    ),
    tags={"team": "data-engineering", "domain": "fashion-retail"},
)

# Raw-only job — useful for incremental backfills when dbt models are unchanged
raw_only_job = define_asset_job(
    name="hm_fashion_raw_only_job",
    selection=AssetSelection.groups("raw_ingestion"),
    description="Materialises only the raw GCS ingestion assets (skips dbt transforms).",
)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

nightly_ingestion_schedule = ScheduleDefinition(
    job=hm_fashion_ingestion_job,
    cron_schedule="0 2 * * *",  # 02:00 UTC daily
    name="nightly_hm_ingestion_schedule",
    description=(
        "Runs the full H&M data ingestion and dbt transformation pipeline nightly "
        "at 02:00 UTC, ensuring the analytics mart tables are fresh for the "
        "traditional ML training jobs that run at 03:00 UTC."
    ),
)


# ---------------------------------------------------------------------------
# Definitions object — the Dagster project entrypoint
# ---------------------------------------------------------------------------

defs = Definitions(
    assets=[
        raw_fashion_reviews,
        raw_fashion_transactions,
        raw_fashion_articles,
        dbt_fashion_marts,
    ],
    jobs=[
        hm_fashion_ingestion_job,
        raw_only_job,
    ],
    schedules=[
        nightly_ingestion_schedule,
    ],
    sensors=[
        gcs_new_file_sensor,
    ],
    resources={
        # Resource keys map to the parameter names in @asset functions
        "bigquery": bigquery_resource,
        "gcs": gcs_resource,
        "dbt": dbt_resource,
    },
)
