"""
Dagster resource definitions for the H&M Traditional ML pipeline.

These resources are separate from the data ingestion pipeline to allow
each Dagster project to be deployed independently on different Cloud Run
services or Vertex AI Training jobs.
"""
import os
from dagster import ConfigurableResource
from pydantic import Field


class BigQueryResource(ConfigurableResource):
    """
    Wraps Google BigQuery client for the ML pipeline.

    Reads from the `analytics` dataset populated by the upstream
    demo_data_ingestion dbt pipeline, and writes trained model artefacts
    and evaluation results back to a `ml_results` dataset.
    """
    project_id: str = Field(
        default_factory=lambda: os.getenv("GCP_PROJECT_ID", "demo-project-hm"),
    )
    dataset_analytics: str = Field(default="analytics")
    dataset_ml_results: str = Field(default="ml_results")
    location: str = Field(default="EU")

    def get_client(self):
        from google.cloud import bigquery
        return bigquery.Client(project=self.project_id, location=self.location)

    def full_table_id(self, dataset: str, table: str) -> str:
        return f"{self.project_id}.{dataset}.{table}"


class VertexAIResource(ConfigurableResource):
    """
    Wraps Vertex AI SDK initialisation for model registration and deployment.

    In production, trained model artefacts (pickled sklearn pipelines or
    XGBoost booster files) are uploaded to GCS and registered with Vertex AI
    Model Registry. The resource manages SDK initialisation.
    """
    project_id: str = Field(
        default_factory=lambda: os.getenv("GCP_PROJECT_ID", "demo-project-hm"),
    )
    location: str = Field(
        default_factory=lambda: os.getenv("VERTEX_LOCATION", "europe-west1"),
    )
    staging_bucket: str = Field(
        default_factory=lambda: os.getenv("VERTEX_STAGING_BUCKET", "gs://hm-vertex-staging"),
    )

    def init(self):
        """Initialise Vertex AI SDK with project and location."""
        try:
            from google.cloud import aiplatform
            aiplatform.init(
                project=self.project_id,
                location=self.location,
                staging_bucket=self.staging_bucket,
            )
            return aiplatform
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform is required. "
                "Install with: pip install google-cloud-aiplatform"
            )


# Pre-configured instances
bigquery_resource = BigQueryResource(
    project_id=os.getenv("GCP_PROJECT_ID", "demo-project-hm"),
)

vertex_resource = VertexAIResource(
    project_id=os.getenv("GCP_PROJECT_ID", "demo-project-hm"),
    location=os.getenv("VERTEX_LOCATION", "europe-west1"),
)
