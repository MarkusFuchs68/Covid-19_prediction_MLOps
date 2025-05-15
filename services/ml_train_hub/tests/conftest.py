import pytest
from fastapi.testclient import TestClient
from ml_train_hub.app.main import app


# fixture for TestClient instance
@pytest.fixture(scope="module")
def test_train_hub_client():
    client = TestClient(app)
    yield client
