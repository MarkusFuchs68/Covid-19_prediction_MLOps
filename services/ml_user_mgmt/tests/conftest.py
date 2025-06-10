import pytest
from fastapi.testclient import TestClient
from ml_user_mgmt.app.main import app


# fixture for TestClient instance
@pytest.fixture(scope="module")
def test_user_mgmt_client():
    client = TestClient(app)
    yield client
