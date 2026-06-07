"""
Tests for the Sentiment Intelligence Engine API.

Demonstrates:
  - Executable documentation (test names describe API capabilities)
  - Parameterized validation testing
  - Happy path + edge case coverage

Run: pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient
from demo_sentiment_categoriser.backend import app


@pytest.fixture
def client():
    """Create a test client for the Sentiment API."""
    return TestClient(app)


# --- Health Check ---

def test_health_endpoint_returns_200_with_status(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "demo_mode" in data


# --- Single Analysis ---

def test_analyze_returns_structured_sentiment_for_valid_text(client):
    response = client.post("/api/v1/analyze", json={
        "text": "This product is absolutely amazing! Best purchase ever."
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["result"]["sentiment"] in ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"]
    assert 0 <= data["result"]["confidence"] <= 1
    assert isinstance(data["result"]["aspects"], list)
    assert isinstance(data["result"]["key_phrases"], list)
    assert data["result"]["urgency"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def test_analyze_returns_negative_sentiment_for_complaint(client):
    response = client.post("/api/v1/analyze", json={
        "text": "Terrible quality, completely disappointed. Worst purchase, demanding refund."
    })
    data = response.json()
    assert data["result"]["sentiment"] in ["NEGATIVE", "MIXED"]


def test_analyze_with_context_accepts_valid_context(client):
    response = client.post("/api/v1/analyze", json={
        "text": "Great service, fast delivery, very satisfied.",
        "context": "product review"
    })
    assert response.status_code == 200
    assert response.json()["success"] is True


# --- Validation ---

@pytest.mark.parametrize("input_data, expected_status", [
    ({"text": "Valid review text for analysis"}, 200),
    ({"text": "OK"}, 422),         # Too short (min_length=5)
    ({}, 422),                      # Missing required field
    ({"text": ""}, 422),            # Empty string
])
def test_analyze_validates_input_correctly(client, input_data, expected_status):
    response = client.post("/api/v1/analyze", json=input_data)
    assert response.status_code == expected_status


# --- Batch Analysis ---

def test_batch_analyze_processes_multiple_texts(client):
    items = [
        {"text": "Amazing product, love it!"},
        {"text": "Terrible experience, very frustrated with shipping."},
        {"text": "It's okay, nothing special but works fine."},
    ]
    response = client.post("/api/v1/analyze/batch", json=items)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 3
    assert len(data["results"]) == 3
    assert "sentiment_distribution" in data["aggregate"]


# --- History ---

def test_history_returns_previous_analyses(client):
    # Perform an analysis first
    client.post("/api/v1/analyze", json={"text": "Great product quality!"})
    response = client.get("/api/v1/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_analyze_with_pytorch_backend_returns_valid_prediction(client):
    response = client.post("/api/v1/analyze", json={
        "text": "The dress design is gorgeous and fits perfectly, but the zipper broke.",
        "backend": "LOCAL_PYTORCH"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["model_used"] == "local-pytorch"
    assert data["result"]["sentiment"] in ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"]
    assert len(data["result"]["aspects"]) > 0

