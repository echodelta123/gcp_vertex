"""
Model Evaluation & Reporting Assets
=====================================

This module defines the final stage of the H&M Traditional ML pipeline:
consolidated model evaluation and a structured JSON report asset.

The evaluation asset consumes outputs from all three training assets and:
  - Aggregates evaluation metrics across models
  - Flags models that fall below minimum quality thresholds
  - Emits a structured evaluation report as a Dagster asset
  - (Production) Triggers a Vertex AI Model Registry promotion workflow
    when all models pass quality gates

This pattern mirrors MLOps best practices where model promotion is gated
on automated quality checks rather than manual approval — a key principle
of the Google Cloud MLOps maturity model.
"""
import os
import json
import logging
from datetime import datetime, timezone

from dagster import (
    asset,
    AssetIn,
    AssetExecutionContext,
    MetadataValue,
    Output,
)

logger = logging.getLogger(__name__)
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

# Quality gates — models must exceed these thresholds to be promoted
QUALITY_GATES = {
    "churn_auc_roc_min": 0.70,
    "churn_f1_min": 0.55,
    "segmentation_silhouette_min": 0.30,
    "demand_mape_max": 30.0,  # MAPE must be BELOW this threshold
}


@asset(
    group_name="model_evaluation",
    key_prefix=["hm_fashion", "evaluation"],
    ins={
        "churn_model": AssetIn(key=["hm_fashion", "models", "churn_model_xgboost"]),
        "segmentation_model": AssetIn(key=["hm_fashion", "models", "customer_segments_kmeans"]),
        "demand_model": AssetIn(key=["hm_fashion", "models", "demand_forecast_arima"]),
    },
    description=(
        "Aggregates evaluation metrics from all three H&M ML models. "
        "Applies quality gate checks and produces a structured JSON report "
        "suitable for MLOps dashboards and Vertex AI Model Registry promotion decisions."
    ),
    metadata={
        "quality_gates": str(QUALITY_GATES),
        "report_format": "JSON",
        "production_action": "Vertex AI Model Registry promotion",
    },
)
def ml_evaluation_report(
    context: AssetExecutionContext,
    churn_model: dict,
    segmentation_model: dict,
    demand_model: dict,
) -> Output[dict]:
    """
    Produce a consolidated ML model evaluation report with quality gate checks.

    Quality Gate Logic:
      - Churn AUC-ROC >= 0.70   (minimum discrimination ability)
      - Churn F1-Score >= 0.55   (balanced precision/recall for imbalanced classes)
      - Segmentation Silhouette >= 0.30  (meaningful cluster separation)
      - Demand MAPE <= 30%       (acceptable forecast accuracy for fashion demand)

    Production Vertex AI Model Registry integration:
        if all(gate_results.values()):
            model = aiplatform.Model.upload(
                display_name=f"hm-churn-xgboost-v{version}",
                artifact_uri=f"gs://hm-models/churn/v{version}/",
                serving_container_image_uri=SERVING_IMAGE,
            )
            endpoint.deploy(model, traffic_split={"0": 10})  # Canary: 10% traffic
    """
    churn_metrics = churn_model.get("metrics", {})
    segmentation_metrics = segmentation_model.get("metrics", {})
    demand_results = demand_model.get("results", {})

    avg_mape = None
    if demand_results:
        mapes = [v["mape_pct"] for v in demand_results.values() if v.get("mape_pct") is not None]
        avg_mape = sum(mapes) / len(mapes) if mapes else None

    # Quality gate evaluation
    gate_results = {
        "churn_auc_roc": churn_metrics.get("auc_roc", 0.0) >= QUALITY_GATES["churn_auc_roc_min"],
        "churn_f1": churn_metrics.get("f1_score", 0.0) >= QUALITY_GATES["churn_f1_min"],
        "segmentation_silhouette": (
            segmentation_metrics.get("silhouette_score", 0.0) >= QUALITY_GATES["segmentation_silhouette_min"]
        ),
        "demand_mape": avg_mape is not None and avg_mape <= QUALITY_GATES["demand_mape_max"],
    }
    all_gates_passed = all(gate_results.values())

    # Log gate results
    context.log.info("=== Quality Gate Results ===")
    for gate, passed in gate_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        context.log.info(f"  {gate}: {status}")
    context.log.info(f"=== Overall: {'✅ ALL PASS — Promote to Registry' if all_gates_passed else '❌ QUALITY GATES FAILED — Block Promotion'} ===")

    report = {
        "pipeline": "hm_fashion_traditional_ml",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "demo_mode": DEMO_MODE,
        "models": {
            "churn_propensity_xgboost": {
                "status": "PASS" if gate_results["churn_auc_roc"] and gate_results["churn_f1"] else "FAIL",
                "metrics": churn_metrics,
                "train_samples": churn_model.get("train_rows"),
                "test_samples": churn_model.get("test_rows"),
                "top_features": list(churn_model.get("feature_importances", {}).keys())[:5],
            },
            "customer_segmentation_kmeans": {
                "status": "PASS" if gate_results["segmentation_silhouette"] else "FAIL",
                "metrics": segmentation_metrics,
                "k": segmentation_model.get("k"),
                "segments": list(segmentation_model.get("segment_map", {}).values()),
            },
            "demand_forecasting_arima": {
                "status": "PASS" if gate_results["demand_mape"] else "FAIL",
                "avg_mape_pct": round(avg_mape, 2) if avg_mape else None,
                "product_types_modelled": demand_model.get("product_types_modelled"),
                "forecast_horizon_days": demand_model.get("forecast_horizon_days"),
            },
        },
        "quality_gates": {
            "thresholds": QUALITY_GATES,
            "results": gate_results,
            "all_passed": all_gates_passed,
        },
        "recommendation": (
            "PROMOTE: All models meet quality thresholds. "
            "Trigger Vertex AI Model Registry registration workflow."
            if all_gates_passed
            else "HOLD: One or more models failed quality gates. "
                 "Review metrics and retrain with adjusted hyperparameters."
        ),
    }

    context.log.info(f"Evaluation report:\n{json.dumps(report, indent=2, default=str)}")

    return Output(
        value=report,
        metadata={
            "all_gates_passed": MetadataValue.bool(all_gates_passed),
            "churn_auc_roc": MetadataValue.float(churn_metrics.get("auc_roc", 0.0)),
            "churn_f1": MetadataValue.float(churn_metrics.get("f1_score", 0.0)),
            "segmentation_silhouette": MetadataValue.float(
                segmentation_metrics.get("silhouette_score", 0.0)
            ),
            "demand_avg_mape_pct": MetadataValue.float(round(avg_mape, 2) if avg_mape else 0.0),
            "recommendation": MetadataValue.text(report["recommendation"]),
            "demo_mode": MetadataValue.bool(DEMO_MODE),
        },
    )
