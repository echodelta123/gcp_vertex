# demo_data_ingestion/__init__.py
"""
H&M Fashion Data Ingestion Pipeline — Dagster Project

A production-pattern data engineering pipeline that orchestrates the full
ingestion and transformation lifecycle for H&M fashion retail data:

  GCS (raw files)
    └── raw_fashion_reviews        [Dagster SDA]
    └── raw_fashion_transactions   [Dagster SDA]
    └── raw_fashion_articles       [Dagster SDA]
          └── dbt_fashion_marts    [Dagster SDA + dbt subprocess]
                └── BigQuery analytics.*  (consumed by demo_traditional_ml)

Entry point: demo_data_ingestion.definitions:defs
"""
