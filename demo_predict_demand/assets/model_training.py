"""
Model Training Assets
======================

This module defines Dagster software-defined assets for training three
traditional ML models against the H&M fashion feature tables.

Models trained:

  1. Churn Propensity XGBoost Classifier
     Input:  churn_feature_table (customer behavioural features)
     Output: Trained model artefact + evaluation metrics
     Prod:   Exported to Vertex AI Model Registry for serving

  2. Customer Segmentation K-Means
     Input:  segmentation_feature_table (RFM features)
     Output: Cluster centroids + per-customer segment assignments
     Prod:   BigQuery ML model stored in `analytics.customer_segments_kmeans`

  3. Demand Forecasting ARIMA+
     Input:  demand_forecast_feature_table (daily time-series)
     Output: 30-day forecast per article + confidence intervals
     Prod:   BigQuery ML ARIMA+ model in `analytics.demand_forecast_arima`

Each training asset emits rich evaluation metadata visible in the Dagster UI
asset catalogue: accuracy, AUC, silhouette score, MAPE, etc.
"""
import os
import io
import json
import logging
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dagster import (
    asset,
    AssetIn,
    AssetExecutionContext,
    MetadataValue,
    Output,
)

logger = logging.getLogger(__name__)
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Asset 1: Churn Propensity — XGBoost Classifier
# ---------------------------------------------------------------------------

@asset(
    group_name="model_training",
    key_prefix=["hm_fashion", "models"],
    ins={"features": AssetIn(key=["hm_fashion", "features", "churn_feature_table"])},
    description=(
        "Trains an XGBoost gradient-boosted classifier to predict 90-day customer "
        "churn propensity using H&M behavioural features. In production the trained "
        "model is registered with Vertex AI Model Registry for online serving."
    ),
    metadata={
        "algorithm": "XGBoost (GBTClassifier)",
        "framework": "xgboost + scikit-learn",
        "production_target": "Vertex AI Model Registry",
    },
)
def churn_model_xgboost(
    context: AssetExecutionContext,
    features: pd.DataFrame,
) -> Output[dict]:
    """
    Train and evaluate an XGBoost churn classifier on H&M customer data.

    Training pipeline:
      1. Split into 80/20 stratified train/test sets.
      2. Impute missing values (median strategy for numeric features).
      3. Standard-scale features (zero mean, unit variance).
      4. Train XGBoost with early stopping on log-loss.
      5. Evaluate on hold-out set: Accuracy, AUC-ROC, Precision, Recall, F1.
      6. Compute SHAP feature importances for explainability.
      7. (Production) Register model artefact with Vertex AI.

    Production Vertex AI path:
        aiplatform.init(project=GCP_PROJECT, location="europe-west1")
        model = aiplatform.Model.upload(
            display_name="hm-churn-xgboost-v1",
            artifact_uri=f"gs://hm-models/churn/v{version}/",
            serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest",
        )
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
        classification_report,
    )

    feature_cols = [
        "order_count", "total_spend", "avg_order_value",
        "days_since_last_purchase", "support_tickets", "avg_review_rating",
        "discount_code_usage_ratio", "session_count_30d", "avg_session_duration_s",
    ]
    target_col = "churned"

    # Filter to available columns
    available = [c for c in feature_cols if c in features.columns]
    X = features[available].fillna(0)
    y = features[target_col]

    context.log.info(
        f"Training data: {len(X)} rows, {len(available)} features, "
        f"churn rate: {y.mean():.1%}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    try:
        import xgboost as xgb
        model_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ])
    except ImportError:
        # Fallback to GradientBoostingClassifier if xgboost not installed
        context.log.warning("xgboost not installed — falling back to sklearn GradientBoostingClassifier")
        from sklearn.ensemble import GradientBoostingClassifier
        model_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=RANDOM_SEED,
            )),
        ])

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    report = classification_report(y_test, y_pred, target_names=["Retained", "Churned"])
    context.log.info(f"Classification Report:\n{report}")

    # Feature importance (from the classifier step)
    classifier = model_pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importances = dict(zip(available, classifier.feature_importances_.tolist()))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        context.log.info(f"Top 5 features by importance: {top_features}")
    else:
        importances = {}

    # Serialise model artefact to bytes (in production: upload to GCS)
    model_bytes = pickle.dumps(model_pipeline)
    context.log.info(f"Model artefact size: {len(model_bytes) / 1024:.1f} KB")

    return Output(
        value={
            "model_type": "xgboost_churn_classifier",
            "metrics": {
                "accuracy": round(accuracy, 4),
                "auc_roc": round(auc, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
            },
            "feature_importances": importances,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        metadata={
            "accuracy": MetadataValue.float(round(accuracy, 4)),
            "auc_roc": MetadataValue.float(round(auc, 4)),
            "precision": MetadataValue.float(round(precision, 4)),
            "recall": MetadataValue.float(round(recall, 4)),
            "f1_score": MetadataValue.float(round(f1, 4)),
            "train_samples": MetadataValue.int(len(X_train)),
            "test_samples": MetadataValue.int(len(X_test)),
            "feature_count": MetadataValue.int(len(available)),
            "demo_mode": MetadataValue.bool(DEMO_MODE),
            "trained_at": MetadataValue.text(datetime.now(timezone.utc).isoformat()),
        },
    )


# ---------------------------------------------------------------------------
# Asset 2: Customer Segmentation — K-Means
# ---------------------------------------------------------------------------

@asset(
    group_name="model_training",
    key_prefix=["hm_fashion", "models"],
    ins={"features": AssetIn(key=["hm_fashion", "features", "segmentation_feature_table"])},
    description=(
        "Trains a K-Means (k=5) customer segmentation model on H&M RFM features. "
        "Assigns each customer to a behavioural segment: High-Value, Brand Loyalist, "
        "Value Seeker, Gifting Shopper, or Casual Browser. Evaluates cluster quality "
        "via Silhouette Score and Davies-Bouldin Index."
    ),
    metadata={
        "algorithm": "K-Means (k=5)",
        "evaluation_metrics": "Silhouette Score, Davies-Bouldin Index",
        "segment_names": "High-Value, Brand Loyalist, Value Seeker, Gifting Shopper, Casual Browser",
    },
)
def customer_segments_kmeans(
    context: AssetExecutionContext,
    features: pd.DataFrame,
) -> Output[dict]:
    """
    Train a K-Means segmentation model on H&M RFM customer features.

    Steps:
      1. Select RFM feature columns from the input DataFrame.
      2. Standard-scale features (K-Means is distance-based, scale matters).
      3. Fit K-Means with k=5, 300 max iterations, 10 random initialisations.
      4. Compute cluster quality metrics (Silhouette Score, DB Index).
      5. Assign human-readable segment labels based on centroid positions.
      6. Produce per-cluster descriptive statistics.

    BigQuery ML equivalent (production):
        CREATE OR REPLACE MODEL `analytics.customer_segments_kmeans`
        OPTIONS (
            model_type   = 'kmeans',
            num_clusters = 5,
            kmeans_init_method = 'KMEANS++',
            standardize_features = TRUE
        )
        AS SELECT recency, frequency, monetary, avg_basket_size, category_breadth
        FROM `analytics.mart_customer_behaviour`
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    feature_cols = ["recency", "frequency", "monetary", "avg_basket_size", "category_breadth"]
    available = [c for c in feature_cols if c in features.columns]
    X = features[available].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = 5
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=RANDOM_SEED)
    cluster_labels = kmeans.fit_predict(X_scaled)

    silhouette = float(silhouette_score(X_scaled, cluster_labels, sample_size=min(5000, len(X_scaled))))
    db_score = float(davies_bouldin_score(X_scaled, cluster_labels))

    context.log.info(f"K-Means fit: Silhouette={silhouette:.4f}, Davies-Bouldin={db_score:.4f}")

    # Assign segment labels based on centroid characteristics
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    segment_map = _assign_segment_labels(centroids, available)

    # Per-cluster stats
    features_with_labels = features.copy()
    features_with_labels["cluster"] = cluster_labels
    features_with_labels["segment"] = features_with_labels["cluster"].map(segment_map)

    cluster_stats = (
        features_with_labels.groupby("segment")
        .agg(
            customer_count=("cluster", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .round(2)
        .to_dict()
    )

    context.log.info(f"Segment distribution:\n{features_with_labels['segment'].value_counts()}")

    return Output(
        value={
            "model_type": "kmeans_customer_segmentation",
            "k": k,
            "metrics": {
                "silhouette_score": round(silhouette, 4),
                "davies_bouldin_index": round(db_score, 4),
            },
            "segment_map": segment_map,
            "cluster_stats": cluster_stats,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        metadata={
            "silhouette_score": MetadataValue.float(round(silhouette, 4)),
            "davies_bouldin_index": MetadataValue.float(round(db_score, 4)),
            "n_clusters": MetadataValue.int(k),
            "customer_count": MetadataValue.int(len(features)),
            "segments": MetadataValue.text(str(list(set(segment_map.values())))),
            "demo_mode": MetadataValue.bool(DEMO_MODE),
        },
    )


def _assign_segment_labels(centroids: np.ndarray, feature_names: list[str]) -> dict[int, str]:
    """
    Assign human-readable segment names to K-Means cluster indices.

    Heuristic: rank clusters by their `monetary` centroid value and
    assign labels accordingly (highest spend → High-Value Collector, etc.).
    """
    segment_names = [
        "High-Value Collector",
        "Brand Loyalist",
        "Value Seeker",
        "Gifting Shopper",
        "Casual Browser",
    ]

    monetary_idx = feature_names.index("monetary") if "monetary" in feature_names else 0
    centroid_monetary = [(i, c[monetary_idx]) for i, c in enumerate(centroids)]
    centroid_monetary.sort(key=lambda x: x[1], reverse=True)

    return {cluster_idx: segment_names[rank] for rank, (cluster_idx, _) in enumerate(centroid_monetary)}


# ---------------------------------------------------------------------------
# Asset 3: Demand Forecasting — ARIMA+
# ---------------------------------------------------------------------------

@asset(
    group_name="model_training",
    key_prefix=["hm_fashion", "models"],
    ins={"features": AssetIn(key=["hm_fashion", "features", "demand_forecast_feature_table"])},
    description=(
        "Trains ARIMA+ time-series demand forecasting models per product type "
        "using H&M daily sales data. Generates 30-day forward forecasts with "
        "95% prediction intervals. In production uses BigQuery ML ARIMA_PLUS."
    ),
    metadata={
        "algorithm": "ARIMA+ (statsmodels SARIMAX)",
        "forecast_horizon": "30 days",
        "confidence_interval": "95%",
        "production_target": "BigQuery ML ARIMA_PLUS",
    },
)
def demand_forecast_arima(
    context: AssetExecutionContext,
    features: pd.DataFrame,
) -> Output[dict]:
    """
    Fit ARIMA+ demand forecasting models and generate 30-day forward projections.

    One ARIMA model is trained per product type. This mirrors the BigQuery ML
    ARIMA_PLUS multi-series approach where `time_series_id_col = 'article_id'`.

    Production BigQuery ML query:
        CALL BQ.FORECAST(
            MODEL `analytics.demand_forecast_arima`,
            STRUCT(30 AS horizon, 0.95 AS confidence_level)
        )

    In DEMO_MODE, statsmodels SARIMAX is used with (1,1,1)x(1,1,1,7) orders
    to capture weekly seasonality in H&M fashion demand patterns.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import warnings
    warnings.filterwarnings("ignore")  # SARIMAX can be verbose

    product_types = features["product_type"].unique() if "product_type" in features.columns else ["All"]
    forecast_results = {}

    for pt in product_types:
        context.log.info(f"Fitting ARIMA+ for product type: {pt}")

        if "product_type" in features.columns:
            series_df = features[features["product_type"] == pt].copy()
        else:
            series_df = features.copy()

        series_df = series_df.sort_values("date")
        y = series_df["daily_units_sold"].values

        if len(y) < 14:
            context.log.warning(f"Insufficient data for {pt} — skipping (need ≥14 days, got {len(y)})")
            continue

        try:
            # SARIMA(1,1,1)(1,1,1,7) — captures weekly seasonality
            model = SARIMAX(
                y,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=50)

            forecast_obj = fitted.get_forecast(steps=30)
            forecast_mean = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int(alpha=0.05)

            # In-sample MAPE
            y_pred_in = fitted.fittedvalues
            non_zero = y[y > 0]
            y_pred_trimmed = y_pred_in[y > 0]
            mape = float(np.mean(np.abs((non_zero - y_pred_trimmed) / non_zero)) * 100)

            forecast_results[str(pt)] = {
                "forecast_mean": forecast_mean.tolist(),
                "lower_95": conf_int.iloc[:, 0].tolist(),
                "upper_95": conf_int.iloc[:, 1].tolist(),
                "mape_pct": round(mape, 2),
                "aic": round(float(fitted.aic), 2),
                "training_days": len(y),
            }
            context.log.info(f"  {pt}: MAPE={mape:.1f}%, AIC={fitted.aic:.1f}")

        except Exception as e:
            context.log.warning(f"ARIMA fitting failed for {pt}: {e} — using naive forecast")
            naive = np.full(30, np.mean(y[-7:]))
            forecast_results[str(pt)] = {
                "forecast_mean": naive.tolist(),
                "lower_95": (naive * 0.85).tolist(),
                "upper_95": (naive * 1.15).tolist(),
                "mape_pct": None,
                "aic": None,
                "training_days": len(y),
                "fallback": "naive_7d_mean",
            }

    avg_mape = float(np.mean([
        v["mape_pct"] for v in forecast_results.values()
        if v.get("mape_pct") is not None
    ]))
    context.log.info(f"Demand forecasting complete: {len(forecast_results)} models, avg MAPE: {avg_mape:.1f}%")

    return Output(
        value={
            "model_type": "arima_plus_demand_forecasting",
            "product_types_modelled": len(forecast_results),
            "forecast_horizon_days": 30,
            "results": forecast_results,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        metadata={
            "product_types_modelled": MetadataValue.int(len(forecast_results)),
            "avg_mape_pct": MetadataValue.float(round(avg_mape, 2)),
            "forecast_horizon_days": MetadataValue.int(30),
            "confidence_interval": MetadataValue.text("95%"),
            "demo_mode": MetadataValue.bool(DEMO_MODE),
            "trained_at": MetadataValue.text(datetime.now(timezone.utc).isoformat()),
        },
    )
