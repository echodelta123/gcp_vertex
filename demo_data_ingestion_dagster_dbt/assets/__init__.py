# demo_data_ingestion/assets/__init__.py
from demo_data_ingestion.assets.raw_ingestion import (
    raw_fashion_reviews,
    raw_fashion_transactions,
    raw_fashion_articles,
)
from demo_data_ingestion.assets.dbt_transforms import dbt_fashion_marts

__all__ = [
    "raw_fashion_reviews",
    "raw_fashion_transactions",
    "raw_fashion_articles",
    "dbt_fashion_marts",
]
