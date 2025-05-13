import pytest
from app.main import app
from fastapi.testclient import TestClient


# fixture for TestClient instance
@pytest.fixture(scope="module")
def test_train_hub_client():
    client = TestClient(app)
    yield client
