#!/usr/bin/env python
"""Simple pytest to verify demo mode assets run without GCP.
The test loads the Dagster Definitions and materialises the raw ingestion assets
in DEMO_MODE (environment variable). It asserts that each output DataFrame
contains rows.
"""
import os

import pytest
from dagster import materialize_to_memory, Definitions

# Ensure demo mode
os.environ.setdefault("DEMO_MODE", "true")

# Import the definitions from the demo package
from demo_data_ingestion.definitions import defs

@pytest.mark.parametrize("asset_key", [
    "hm_fashion.raw.raw_fashion_reviews",
    "hm_fashion.raw.raw_fashion_transactions",
    "hm_fashion.raw.raw_fashion_articles",
])
def test_demo_asset_materialization(asset_key):
    # Materialise only the selected asset
    result = materialize_to_memory(
        defs,
        selection=asset_key,
    )
    assert result.success, f"Asset {asset_key} failed to materialize"
    # Extract the Output value
    asset_result = result.assets[asset_key]
    df = asset_result.output_value
    assert not df.empty, "DataFrame should contain synthetic rows"
