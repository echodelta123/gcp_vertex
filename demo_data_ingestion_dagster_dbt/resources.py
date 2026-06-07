"""
Dagster resource definitions for the H&M Fashion data ingestion pipeline.

Resources encapsulate external service connections (BigQuery, GCS) and are
injected at runtime, making assets testable without live infrastructure.
In production, these are configured via environment variables or Dagster's
Secrets Manager integration.
"""
import os
from dagster import ConfigurableResource, EnvVar
from dagster_dbt import DbtCliResource
from pydantic import Field


class BigQueryResource(ConfigurableResource):
    """
    Wraps Google BigQuery client configuration.

    In production this resolves credentials from the Workload Identity
    attached to the Cloud Run job or GKE service account — no key file
    is ever written to disk.

    Attributes:
        project_id: GCP project that owns the BigQuery datasets.
        dataset_raw: Landing zone for raw GCS-sourced data.
        dataset_marts: Downstream analytical mart tables (written by dbt).
        location: Multi-region or regional BigQuery location.
    """
    project_id: str = Field(
        default_factory=lambda: os.getenv("GCP_PROJECT_ID", "your-gcp-project"),
        description="GCP project ID",
    )
    dataset_raw: str = Field(
        default="raw_data",
        description="BigQuery dataset for raw ingestion landing tables",
    )
    dataset_marts: str = Field(
        default="analytics",
        description="BigQuery dataset for dbt-compiled analytical mart tables",
    )
    location: str = Field(
        default="EU",
        description="BigQuery dataset location (EU | US | us-central1 etc.)",
    )

    def get_client(self):
        """
        Return an authenticated BigQuery client.

        Uses Application Default Credentials (ADC) — no service-account key
        required when running on GCP infrastructure.
        """
        try:
            from google.cloud import bigquery
            return bigquery.Client(project=self.project_id, location=self.location)
        except ImportError:
            raise ImportError(
                "google-cloud-bigquery is required. "
                "Install with: pip install google-cloud-bigquery"
            )

    def full_table_id(self, dataset: str, table: str) -> str:
        """Return a fully-qualified BigQuery table reference."""
        return f"{self.project_id}.{dataset}.{table}"


class GCSResource(ConfigurableResource):
    """
    Wraps Google Cloud Storage client configuration.

    The pipeline polls this bucket for new H&M fashion data files dropped
    by upstream systems (e.g. Dataflow export jobs, nightly ETL processes).

    Attributes:
        bucket_name: GCS bucket containing raw H&M data files.
        reviews_prefix: Object prefix for customer review JSON files.
        transactions_prefix: Object prefix for transaction CSV files.
        articles_prefix: Object prefix for article/product metadata.
    """
    bucket_name: str = Field(
        default_factory=lambda: os.getenv("GCS_BUCKET_NAME", "hm-fashion-raw-data"),
        description="GCS bucket name for raw H&M source files",
    )
    reviews_prefix: str = Field(
        default="reviews/",
        description="GCS prefix (folder) for customer review JSON files",
    )
    transactions_prefix: str = Field(
        default="transactions/",
        description="GCS prefix (folder) for transaction CSV files",
    )
    articles_prefix: str = Field(
        default="articles/",
        description="GCS prefix (folder) for H&M article metadata files",
    )

    def get_client(self):
        """Return an authenticated GCS client using ADC."""
        try:
            from google.cloud import storage
            return storage.Client()
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required. "
                "Install with: pip install google-cloud-storage"
            )

    def gcs_uri(self, prefix: str) -> str:
        """Return a gs:// URI for a given prefix."""
        return f"gs://{self.bucket_name}/{prefix}"


dbt_resource = DbtCliResource(
    project_dir=os.getenv("DBT_PROJECT_DIR", "./dbt"),
    profiles_dir=os.getenv("DBT_PROFILES_DIR", "~/.dbt"),
    target=os.getenv("DBT_TARGET", "dev"),
)


# ---------------------------------------------------------------------------
# Pre-configured resource instances for local / DEMO_MODE execution
# ---------------------------------------------------------------------------

# These instances use environment variables with safe fallback values,
# enabling the pipeline to run in demo mode without live GCP credentials.

bigquery_resource = BigQueryResource(
    project_id=os.getenv("GCP_PROJECT_ID", "demo-project-hm"),
    dataset_raw="raw_data",
    dataset_marts="analytics",
    location="EU",
)

gcs_resource = GCSResource(
    bucket_name=os.getenv("GCS_BUCKET_NAME", "hm-fashion-raw-data"),
    reviews_prefix="reviews/",
    transactions_prefix="transactions/",
    articles_prefix="articles/",
)


