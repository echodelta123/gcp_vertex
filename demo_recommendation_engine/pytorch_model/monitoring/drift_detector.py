"""Model monitoring and drift detection for the recommendation engine.

Implements three types of monitoring critical for production recommendation systems:

1. **Prediction Drift**: Distribution of recommendation scores shifts over time
   (indicates model staleness or population shift).

2. **Feature Drift**: Input feature distributions diverge from training baseline
   (indicates upstream data pipeline changes or real-world distribution shift).

3. **Business Metric Decay**: Online metrics (CTR, conversion) degrade below threshold
   (the strongest signal that retraining is needed).

In production on Vertex AI, this integrates with:
- Vertex AI Model Monitoring (automatic feature drift detection)
- Cloud Monitoring custom metrics (business KPIs)
- Cloud Functions / Cloud Scheduler for periodic checks
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Summary of a drift detection run."""

    feature_drift_score: float  # KL divergence or PSI
    prediction_drift_score: float  # KS statistic on score distributions
    business_metric_current: float  # Current online metric (e.g., CTR)
    business_metric_baseline: float  # Baseline when model was deployed
    needs_retraining: bool
    reasons: list[str]
    timestamp: str


def compute_psi(
    baseline: list[float], current: list[float], num_bins: int = 10
) -> float:
    """Population Stability Index (PSI) for feature drift detection.

    PSI > 0.1: Some drift, monitor closely
    PSI > 0.2: Significant drift, consider retraining
    PSI > 0.25: Major drift, retrain immediately

    This is the industry-standard metric used in financial services and
    recommendation systems for detecting distributional shift.
    """
    if not baseline or not current:
        return 0.0

    # Compute histogram bins from baseline
    min_val = min(min(baseline), min(current))
    max_val = max(max(baseline), max(current))

    if max_val == min_val:
        return 0.0

    bin_width = (max_val - min_val) / num_bins
    eps = 1e-6  # Avoid log(0)

    baseline_counts = [0] * num_bins
    current_counts = [0] * num_bins

    for val in baseline:
        bin_idx = min(int((val - min_val) / bin_width), num_bins - 1)
        baseline_counts[bin_idx] += 1

    for val in current:
        bin_idx = min(int((val - min_val) / bin_width), num_bins - 1)
        current_counts[bin_idx] += 1

    # Normalize to proportions
    baseline_total = len(baseline)
    current_total = len(current)

    psi = 0.0
    for i in range(num_bins):
        p_baseline = (baseline_counts[i] / baseline_total) + eps
        p_current = (current_counts[i] / current_total) + eps
        psi += (p_current - p_baseline) * math.log(p_current / p_baseline)

    return psi


def compute_ks_statistic(
    baseline_scores: list[float], current_scores: list[float]
) -> float:
    """Kolmogorov-Smirnov statistic for prediction drift.

    Measures the maximum difference between two empirical CDFs.
    KS > 0.05: Notable prediction distribution shift.
    """
    if not baseline_scores or not current_scores:
        return 0.0

    all_values = sorted(set(baseline_scores + current_scores))
    max_diff = 0.0

    baseline_sorted = sorted(baseline_scores)
    current_sorted = sorted(current_scores)
    n_base = len(baseline_sorted)
    n_curr = len(current_sorted)

    base_idx = 0
    curr_idx = 0

    for val in all_values:
        while base_idx < n_base and baseline_sorted[base_idx] <= val:
            base_idx += 1
        while curr_idx < n_curr and current_sorted[curr_idx] <= val:
            curr_idx += 1

        cdf_base = base_idx / n_base
        cdf_curr = curr_idx / n_curr
        max_diff = max(max_diff, abs(cdf_base - cdf_curr))

    return max_diff


def detect_drift(
    baseline_features: dict[str, list[float]],
    current_features: dict[str, list[float]],
    baseline_scores: list[float],
    current_scores: list[float],
    business_metric_current: float,
    business_metric_baseline: float,
    feature_drift_threshold: float = 0.2,
    prediction_drift_threshold: float = 0.05,
    business_metric_decay_threshold: float = 0.1,
) -> DriftReport:
    """Run comprehensive drift detection.

    Returns a DriftReport with actionable retraining recommendations.

    Args:
        baseline_features: Feature distributions at training time.
        current_features: Current feature distributions from serving.
        baseline_scores: Model prediction scores at deployment.
        current_scores: Current prediction scores from serving.
        business_metric_current: Current business metric (e.g., CTR).
        business_metric_baseline: Business metric at deployment time.
        feature_drift_threshold: PSI threshold for feature drift.
        prediction_drift_threshold: KS threshold for prediction drift.
        business_metric_decay_threshold: Relative decay threshold.
    """
    reasons = []

    # Feature drift (average PSI across all features)
    feature_psi_values = []
    for feature_name in baseline_features:
        if feature_name in current_features:
            psi = compute_psi(baseline_features[feature_name], current_features[feature_name])
            feature_psi_values.append(psi)
            if psi > feature_drift_threshold:
                reasons.append(f"Feature '{feature_name}' PSI={psi:.3f} exceeds threshold")

    avg_feature_psi = sum(feature_psi_values) / max(len(feature_psi_values), 1)

    # Prediction drift
    prediction_ks = compute_ks_statistic(baseline_scores, current_scores)
    if prediction_ks > prediction_drift_threshold:
        reasons.append(f"Prediction drift KS={prediction_ks:.3f} exceeds threshold")

    # Business metric decay
    if business_metric_baseline > 0:
        relative_decay = (
            (business_metric_baseline - business_metric_current) / business_metric_baseline
        )
    else:
        relative_decay = 0.0

    if relative_decay > business_metric_decay_threshold:
        reasons.append(
            f"Business metric decayed {relative_decay*100:.1f}% "
            f"(from {business_metric_baseline:.4f} to {business_metric_current:.4f})"
        )

    needs_retraining = len(reasons) > 0

    from datetime import datetime, timezone

    report = DriftReport(
        feature_drift_score=avg_feature_psi,
        prediction_drift_score=prediction_ks,
        business_metric_current=business_metric_current,
        business_metric_baseline=business_metric_baseline,
        needs_retraining=needs_retraining,
        reasons=reasons,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        f"Drift detection complete: needs_retraining={needs_retraining}, "
        f"feature_psi={avg_feature_psi:.4f}, pred_ks={prediction_ks:.4f}, "
        f"biz_decay={relative_decay:.4f}"
    )

    return report


def save_drift_report(report: DriftReport, path: Path) -> None:
    """Persist drift report for audit trail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "feature_drift_score": report.feature_drift_score,
                "prediction_drift_score": report.prediction_drift_score,
                "business_metric_current": report.business_metric_current,
                "business_metric_baseline": report.business_metric_baseline,
                "needs_retraining": report.needs_retraining,
                "reasons": report.reasons,
                "timestamp": report.timestamp,
            },
            f,
            indent=2,
        )
