"""Tests for the Recommendation Engine API."""
import pytest
from fastapi.testclient import TestClient
from demo_recommendation_engine.backend import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_reports_catalog_size(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["catalog_size"] > 0


def test_recommend_returns_products_for_valid_query(client):
    response = client.post("/api/v1/recommend", json={
        "query": "lightweight running shoes", "top_k": 3
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) <= 3
    for rec in data["recommendations"]:
        assert "product" in rec
        assert "explanation" in rec
        assert rec["product"]["name"]
        assert rec["product"]["price"] > 0


def test_catalog_endpoint_returns_all_products(client):
    response = client.get("/api/v1/catalog")
    assert response.status_code == 200
    assert len(response.json()) > 0


@pytest.mark.parametrize("query, expected_status", [
    ("I need hiking boots for wet conditions", 200),
    ("ab", 422),   # Too short
])
def test_recommend_validates_query_length(client, query, expected_status):
    response = client.post("/api/v1/recommend", json={"query": query})
    assert response.status_code == expected_status
