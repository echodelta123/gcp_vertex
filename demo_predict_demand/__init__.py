# demo_traditional_ml/__init__.py
"""
H&M Fashion Traditional ML Pipeline — Dagster Project

A production-pattern machine learning pipeline that trains and evaluates
three classic ML models against the H&M fashion analytics mart:

  BigQuery analytics.*
    └── churn_feature_table           [Dagster SDA — Feature Engineering]
    └── segmentation_feature_table    [Dagster SDA — Feature Engineering]
    └── demand_forecast_feature_table [Dagster SDA — Feature Engineering]
          ├── churn_model_xgboost     [Dagster SDA — XGBoost Training]
          ├── customer_segments_kmeans [Dagster SDA — K-Means Training]
          ├── demand_forecast_arima   [Dagster SDA — ARIMA+ Training]
          └── ml_evaluation_report    [Dagster SDA — Quality Gates + Reporting]

Entry point: demo_traditional_ml.definitions:defs
"""
