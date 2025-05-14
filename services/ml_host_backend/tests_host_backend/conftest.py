import pytest
from app.main import app
from fastapi.testclient import TestClient


# fixture for TestClient instance
@pytest.fixture(scope="module")
def test_host_backend_client():
    client = TestClient(app)
    yield client
