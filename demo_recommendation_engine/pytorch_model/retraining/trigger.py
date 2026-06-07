"""Retraining trigger logic for automated model refresh.

Implements the decision logic for when to retrain the recommendation model.
Integrates with:
- Drift detection outputs
- Business metric monitoring
- Data freshness signals
- Vertex AI Pipelines (triggers a new pipeline run)

In production, this runs as:
- A Cloud Function triggered by Cloud Scheduler (periodic)
- Or as a Cloud Function triggered by Pub/Sub (event-driven, e.g., drift alert)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..monitoring.drift_detector import DriftReport

logger = logging.getLogger(__name__)


@dataclass
class RetrainingDecision:
    """Encapsulates the decision and metadata for a retraining trigger."""

    should_retrain: bool
    priority: str  # "critical", "high", "normal", "skip"
    trigger_reasons: list[str]
    recommended_config: dict
    timestamp: str


def evaluate_retraining_need(
    drift_report: DriftReport,
    days_since_last_training: int,
    new_interactions_count: int,
    min_interactions_for_retrain: int = 10000,
    max_days_without_retrain: int = 14,
    critical_metric_decay: float = 0.2,
) -> RetrainingDecision:
    """Evaluate whether retraining should be triggered.

    Decision matrix:
    - CRITICAL: Business metric decay > 20% → retrain immediately
    - HIGH: Drift detected + > 7 days since last train → retrain soon
    - NORMAL: Scheduled (> 14 days) + sufficient new data → retrain
    - SKIP: No signals, recent training, insufficient new data

    Args:
        drift_report: Latest drift detection report.
        days_since_last_training: Days since last successful training run.
        new_interactions_count: Number of new interactions since last training.
        min_interactions_for_retrain: Minimum new data to justify retraining.
        max_days_without_retrain: Force retrain after this many days.
        critical_metric_decay: Business metric decay threshold for critical priority.
    """
    reasons = []
    priority = "skip"

    # Check for critical business metric decay
    if drift_report.business_metric_baseline > 0:
        decay = (
            (drift_report.business_metric_baseline - drift_report.business_metric_current)
            / drift_report.business_metric_baseline
        )
        if decay > critical_metric_decay:
            reasons.append(
                f"Critical business metric decay: {decay*100:.1f}% "
                f"(threshold: {critical_metric_decay*100:.0f}%)"
            )
            priority = "critical"

    # Check drift signals
    if drift_report.needs_retraining:
        reasons.extend(drift_report.reasons)
        if priority != "critical":
            priority = "high" if days_since_last_training > 7 else "normal"

    # Check staleness
    if days_since_last_training >= max_days_without_retrain:
        reasons.append(
            f"Model stale: {days_since_last_training} days since last training "
            f"(max: {max_days_without_retrain})"
        )
        if priority == "skip":
            priority = "normal"

    # Check data sufficiency
    has_enough_data = new_interactions_count >= min_interactions_for_retrain
    if not has_enough_data and priority != "critical":
        reasons.append(
            f"Insufficient new data: {new_interactions_count}/{min_interactions_for_retrain} "
            f"interactions (waiting for more data)"
        )
        if priority in ("normal", "skip"):
            priority = "skip"

    should_retrain = priority in ("critical", "high", "normal") and has_enough_data

    # Determine training configuration based on priority
    recommended_config = _get_recommended_config(priority, new_interactions_count)

    decision = RetrainingDecision(
        should_retrain=should_retrain,
        priority=priority,
        trigger_reasons=reasons,
        recommended_config=recommended_config,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        f"Retraining decision: should_retrain={should_retrain}, "
        f"priority={priority}, reasons={len(reasons)}"
    )

    return decision


def _get_recommended_config(priority: str, new_interactions: int) -> dict:
    """Determine training hyperparameters based on retraining priority.

    Critical retraining uses more conservative settings (lower LR, more epochs)
    to ensure quality. Normal scheduled retraining can be faster.
    """
    if priority == "critical":
        return {
            "epochs": 15,
            "learning_rate": 5e-4,
            "batch_size": 512,
            "warmup_steps": 500,
            "use_full_dataset": True,
            "run_extended_eval": True,
        }
    elif priority == "high":
        return {
            "epochs": 10,
            "learning_rate": 1e-3,
            "batch_size": 512,
            "warmup_steps": 200,
            "use_full_dataset": True,
            "run_extended_eval": True,
        }
    else:
        return {
            "epochs": 5,
            "learning_rate": 1e-3,
            "batch_size": 256,
            "warmup_steps": 100,
            "use_full_dataset": False,
            "run_extended_eval": False,
        }


def save_retraining_decision(decision: RetrainingDecision, path: Path) -> None:
    """Persist decision for audit trail and pipeline triggering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "should_retrain": decision.should_retrain,
                "priority": decision.priority,
                "trigger_reasons": decision.trigger_reasons,
                "recommended_config": decision.recommended_config,
                "timestamp": decision.timestamp,
            },
            f,
            indent=2,
        )
