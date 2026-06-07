"""
Dagster Sensor: GCS New-File Trigger
=====================================

Sensors in Dagster are long-running processes that poll an external system
and emit RunRequests when conditions are met.

This sensor monitors the H&M raw data GCS bucket for new files under the
`reviews/` or `transactions/` prefixes. When a new object is detected
(based on a persisted cursor of the last-seen blob name), the sensor emits
a RunRequest that triggers a full pipeline materialisation run.

Deployment context:
  In production this sensor runs inside the Dagster Daemon on Cloud Run,
  polling every 60 seconds. It emits run requests scoped to the
  `hm_fashion_ingestion_job` job, which targets all assets in the
  `raw_ingestion` and `dbt_transforms` asset groups.

DEMO_MODE behaviour:
  With DEMO_MODE=true the sensor always emits a RunRequest immediately
  (bypassing GCS polling), so reviewers can trigger the full pipeline
  from the Dagster UI without GCP credentials.
"""
import os
from dagster import (
    sensor,
    RunRequest,
    SensorEvaluationContext,
    DefaultSensorStatus,
    SkipReason,
)

DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "hm-fashion-raw-data")
WATCHED_PREFIXES = ["reviews/", "transactions/", "articles/"]


@sensor(
    job_name="hm_fashion_ingestion_job",
    minimum_interval_seconds=60,
    default_status=DefaultSensorStatus.RUNNING,
    description=(
        "Polls the H&M raw data GCS bucket for new files. Emits a RunRequest "
        "when objects newer than the cursor are detected under the reviews/, "
        "transactions/, or articles/ prefixes."
    ),
)
def gcs_new_file_sensor(context: SensorEvaluationContext):
    """
    Trigger the ingestion pipeline whenever new source files arrive in GCS.

    Cursor format: ISO-8601 timestamp of the most recently seen blob
    update time. On each evaluation:
      1. List all blobs under WATCHED_PREFIXES.
      2. Filter to those updated after the cursor.
      3. If any new blobs found → emit RunRequest and advance cursor.
      4. If none → emit SkipReason.

    In DEMO_MODE the sensor emits exactly one RunRequest per Dagster
    daemon startup to kick off the demo pipeline, then stops emitting
    to avoid infinite loops.
    """
    if DEMO_MODE:
        # In demo mode, emit one run if we haven't already
        if context.cursor:
            yield SkipReason(
                f"DEMO_MODE — pipeline already triggered (cursor: {context.cursor}). "
                "Reset cursor in the Dagster UI to trigger again."
            )
        else:
            context.log.info("DEMO_MODE: emitting initial RunRequest to trigger demo pipeline")
            context.update_cursor("demo_triggered")
            yield RunRequest(
                run_key="demo_initial_run",
                run_config={
                    "ops": {
                        "hm_fashion__raw__raw_fashion_reviews": {"config": {}},
                        "hm_fashion__raw__raw_fashion_transactions": {"config": {}},
                        "hm_fashion__raw__raw_fashion_articles": {"config": {}},
                    }
                },
                tags={"trigger": "sensor", "mode": "demo"},
            )
        return

    # --- Production path ---
    try:
        from google.cloud import storage
        gcs_client = storage.Client()
        bucket = gcs_client.bucket(GCS_BUCKET)
    except Exception as e:
        yield SkipReason(f"GCS client initialisation failed: {e}")
        return

    last_seen_time = context.cursor or "1970-01-01T00:00:00Z"

    from datetime import datetime, timezone
    cursor_dt = datetime.fromisoformat(last_seen_time.replace("Z", "+00:00"))

    new_blobs = []
    for prefix in WATCHED_PREFIXES:
        blobs = list(bucket.list_blobs(prefix=prefix))
        for blob in blobs:
            if blob.updated and blob.updated > cursor_dt:
                new_blobs.append(blob)

    if not new_blobs:
        yield SkipReason(
            f"No new files detected in gs://{GCS_BUCKET} since {last_seen_time}"
        )
        return

    new_cursor = max(b.updated for b in new_blobs).isoformat()
    context.log.info(
        f"Detected {len(new_blobs)} new file(s) in GCS. "
        f"Advancing cursor to {new_cursor}"
    )
    context.update_cursor(new_cursor)

    blob_names = [b.name for b in new_blobs[:5]]  # log up to 5 for brevity
    yield RunRequest(
        run_key=new_cursor,
        tags={
            "trigger": "gcs_sensor",
            "new_file_count": str(len(new_blobs)),
            "sample_files": str(blob_names),
        },
    )
