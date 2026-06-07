"""
Dagster Definitions: H&M Traditional ML Pipeline
=================================================

Top-level Dagster `Definitions` object for the traditional ML pipeline.

The pipeline is structured in three stages:
  1. Feature Engineering  — query BigQuery analytics marts, build feature DataFrames
  2. Model Training       — train XGBoost, K-Means, and ARIMA+ models
  3. Model Evaluation     — quality gates + structured evaluation report

The pipeline is designed to run after `demo_data_ingestion` has materialised
the `analytics.*` mart tables. In production this is enforced via a Dagster
asset dependency that crosses job boundaries (using cross-repo asset references).

Usage:
    # Start the Dagster UI locally
    DEMO_MODE=true dagster dev -m demo_traditional_ml.definitions

    # Materialise the full ML pipeline
    DEMO_MODE=true dagster asset materialize -m demo_traditional_ml.definitions --select '*'

    # Run only feature engineering
    DEMO_MODE=true dagster asset materialize -m demo_traditional_ml.definitions \\
        --select 'hm_fashion/features/*'

    # Run training + evaluation (assumes features already materialised)
    DEMO_MODE=true dagster asset materialize -m demo_traditional_ml.definitions \\
        --select 'hm_fashion/models/* hm_fashion/evaluation/*'
"""
from dagster import (
    Definitions,
    define_asset_job,
    AssetSelection,
    ScheduleDefinition,
    sensor,
    RunRequest,
    SkipReason,
    SensorEvaluationContext,
    DefaultSensorStatus,
)

from demo_traditional_ml.assets import (
    churn_feature_table,
    segmentation_feature_table,
    demand_forecast_feature_table,
    churn_model_xgboost,
    customer_segments_kmeans,
    demand_forecast_arima,
    ml_evaluation_report,
)
from demo_traditional_ml.resources import bigquery_resource, vertex_resource


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

hm_ml_full_pipeline_job = define_asset_job(
    name="hm_ml_full_pipeline_job",
    selection=AssetSelection.groups(
        "feature_engineering", "model_training", "model_evaluation"
    ),
    description=(
        "Runs the complete H&M Traditional ML pipeline: feature engineering "
        "from BigQuery marts → XGBoost churn training → K-Means segmentation → "
        "ARIMA+ demand forecasting → consolidated evaluation report with quality gates."
    ),
    tags={"team": "ml-engineering", "domain": "fashion-retail"},
)

feature_only_job = define_asset_job(
    name="hm_ml_feature_engineering_job",
    selection=AssetSelection.groups("feature_engineering"),
    description="Materialises only the feature engineering assets (BigQuery mart extraction).",
)

training_evaluation_job = define_asset_job(
    name="hm_ml_training_evaluation_job",
    selection=AssetSelection.groups("model_training", "model_evaluation"),
    description=(
        "Runs model training and evaluation on pre-built feature tables. "
        "Designed for hyperparameter iteration without re-running BigQuery queries."
    ),
)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

nightly_ml_training_schedule = ScheduleDefinition(
    job=hm_ml_full_pipeline_job,
    cron_schedule="0 3 * * *",  # 03:00 UTC — runs after the 02:00 ingestion job
    name="nightly_hm_ml_training_schedule",
    description=(
        "Runs the H&M Traditional ML pipeline nightly at 03:00 UTC, one hour "
        "after the data ingestion job completes and refreshes the analytics mart tables."
    ),
)


# ---------------------------------------------------------------------------
# Sensor: React to upstream data ingestion completing
# ---------------------------------------------------------------------------

import os
_DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"


@sensor(
    job=hm_ml_full_pipeline_job,
    minimum_interval_seconds=120,
    default_status=DefaultSensorStatus.RUNNING,
    name="ingestion_completion_sensor",
    description=(
        "Triggers the ML training pipeline when the upstream data ingestion job "
        "completes successfully. Uses a simple cursor-based check against the "
        "Dagster run history for `hm_fashion_ingestion_job`."
    ),
)
def ingestion_completion_sensor(context: SensorEvaluationContext):
    """
    Trigger ML training when upstream ingestion pipeline completes.

    In DEMO_MODE, emits one RunRequest on first evaluation then stops.
    In production, polls Dagster run history for completed ingestion runs
    newer than the cursor timestamp.
    """
    if _DEMO_MODE:
        if context.cursor:
            yield SkipReason("DEMO_MODE — ML pipeline already triggered.")
        else:
            context.update_cursor("demo_triggered")
            yield RunRequest(
                run_key="demo_ml_initial_run",
                tags={"trigger": "ingestion_completion_sensor", "mode": "demo"},
            )
        return

    # Production: Check if ingestion job completed since last cursor
    from dagster import DagsterRunStatus
    runs = context.instance.get_runs(
        filters=context.instance.RunsFilter(
            job_name="hm_fashion_ingestion_job",
            statuses=[DagsterRunStatus.SUCCESS],
        ),
        limit=1,
    )
    if not runs:
        yield SkipReason("No successful ingestion runs found.")
        return

    latest_run = runs[0]
    last_end_time = str(latest_run.end_time)

    if context.cursor == last_end_time:
        yield SkipReason(f"ML pipeline already triggered for ingestion run {latest_run.run_id[:8]}.")
        return

    context.update_cursor(last_end_time)
    yield RunRequest(
        run_key=f"ml_triggered_by_{latest_run.run_id[:8]}",
        tags={
            "trigger": "ingestion_completion",
            "upstream_run_id": latest_run.run_id[:8],
        },
    )


# ---------------------------------------------------------------------------
# Definitions object
# ---------------------------------------------------------------------------

defs = Definitions(
    assets=[
        churn_feature_table,
        segmentation_feature_table,
        demand_forecast_feature_table,
        churn_model_xgboost,
        customer_segments_kmeans,
        demand_forecast_arima,
        ml_evaluation_report,
    ],
    jobs=[
        hm_ml_full_pipeline_job,
        feature_only_job,
        training_evaluation_job,
    ],
    schedules=[
        nightly_ml_training_schedule,
    ],
    sensors=[
        ingestion_completion_sensor,
    ],
    resources={
        "bigquery": bigquery_resource,
        "vertex_ai": vertex_resource,
    },
)
