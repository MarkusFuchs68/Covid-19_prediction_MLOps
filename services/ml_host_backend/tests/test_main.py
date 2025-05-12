from app.main import (
    app,  # Replace 'main' with the actual module where your FastAPI app is defined
)
from fastapi.testclient import TestClient

# Create a TestClient instance
client = TestClient(app)


def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"ping": "pong!"}


def test_health_endpoint():
    """Test a health check endpoint (e.g., GET /health)."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}  # Adjust expected response


def test_invalid_endpoint():
    """Test an invalid endpoint."""
    response = client.get("/invalid-endpoint")
    assert response.status_code == 404
