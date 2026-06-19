import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "app": "ResearchMind AI"}

def test_list_papers():
    # If the database is empty, it returns an empty list, but the endpoint should succeed
    response = client.get("/api/v1/papers")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
