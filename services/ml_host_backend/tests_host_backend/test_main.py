def test_ping(test_host_backend_client):
    """Test ping endpoint (e.g., GET /ping)."""
    response = test_host_backend_client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"ping": "pong!"}


def test_health_endpoint(test_host_backend_client):
    """Test health check endpoint (e.g., GET /health)."""
    response = test_host_backend_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_invalid_endpoint(test_host_backend_client):
    """Test invalid endpoint."""
    response = test_host_backend_client.get("/invalid-endpoint")
    assert response.status_code == 404
