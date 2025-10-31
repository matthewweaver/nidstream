"""
Integration tests for API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


# NOTE: Prediction tests require a trained model to be loaded
# These will be skipped in CI unless model artifacts are available
@pytest.mark.skip(reason="Requires trained model")
def test_predict_endpoint(client):
    """Test prediction endpoint."""
    sample_flow = {
        "features": {
            "Flow Duration": 1000000,
            "Tot Fwd Pkts": 100,
            "Tot Bwd Pkts": 50,
            # ... add all required features
        }
    }
    
    response = client.post("/predict", json=sample_flow)
    assert response.status_code == 200
    data = response.json()
    assert "is_attack" in data
    assert "anomaly_score" in data
