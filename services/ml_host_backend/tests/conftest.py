import pytest
from fastapi.testclient import TestClient
from app.main import app


# fixture for TestClient instance
@pytest.fixture(scope="module")
def test_host_backend_client():
    client = TestClient(app)
    yield client
