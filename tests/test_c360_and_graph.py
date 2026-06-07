"""Tests for Customer 360 and Graph Explorer APIs."""
import pytest
from fastapi.testclient import TestClient
from demo_customer_support_360.backend import app as c360_app
from demo_instacart_knowledge_graph.backend import app as graph_app


@pytest.fixture
def c360_client():
    return TestClient(c360_app)


@pytest.fixture
def graph_client():
    return TestClient(graph_app)


# --- Customer 360 ---

def test_c360_health(c360_client):
    r = c360_client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["customers_loaded"] > 0


def test_c360_lists_customers(c360_client):
    r = c360_client.get("/api/v1/customers")
    assert r.status_code == 200
    customers = r.json()
    assert len(customers) > 0
    assert customers[0]["name"]


def test_c360_generates_persona(c360_client):
    r = c360_client.post("/api/v1/persona", json={"customer_id": "CUST-1001"})
    assert r.status_code == 200
    data = r.json()
    assert data["success"]
    assert data["persona"]["summary"]
    assert data["persona"]["churn_risk"] in ["LOW", "MEDIUM", "HIGH"]


def test_c360_timeline_returns_interactions(c360_client):
    r = c360_client.get("/api/v1/customers/CUST-1001/timeline")
    assert r.status_code == 200
    assert len(r.json()) > 0


# --- Graph Explorer ---

def test_graph_health(graph_client):
    r = graph_client.get("/api/v1/health")
    assert r.status_code == 200


def test_graph_schema_returns_stats(graph_client):
    r = graph_client.get("/api/v1/schema")
    assert r.status_code == 200
    data = r.json()
    assert data["total_products"] > 0
    assert data["total_relationships"] > 0


def test_graph_query_returns_cypher_and_results(graph_client):
    r = graph_client.post("/api/v1/query", json={
        "query": "What products are frequently bought with denim?"
    })
    assert r.status_code == 200
    data = r.json()
    assert data["generated_cypher"]
    assert data["explanation"]
    assert isinstance(data["nodes"], list)


def test_graph_full_graph_endpoint(graph_client):
    r = graph_client.get("/api/v1/graph")
    assert r.status_code == 200
    data = r.json()
    assert len(data["nodes"]) > 0
    assert len(data["edges"]) > 0
