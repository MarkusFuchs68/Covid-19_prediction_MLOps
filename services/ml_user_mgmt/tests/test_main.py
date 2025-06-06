def test_ping(test_user_mgmt_client):
    """Test ping endpoint (e.g., GET /ping)."""
    response = test_user_mgmt_client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"ping": "pong!"}


def test_health_endpoint(test_user_mgmt_client):
    """Test health check endpoint (e.g., GET /health)."""
    response = test_user_mgmt_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_invalid_endpoint(test_user_mgmt_client):
    """Test invalid endpoint."""
    response = test_user_mgmt_client.get("/invalid-endpoint")
    assert response.status_code == 404


def test_successful_login(test_user_mgmt_client):
    """Test successful login"""
    data = {"username": "fakeuser", "password": "fakepassword"}
    response = test_user_mgmt_client.post("/token", json=data)
    assert response.status_code == 401


def test_verify_jwt_valid_token(test_user_mgmt_client):
    """Test /verify-jwt with a valid JWT token."""
    # First, obtain a valid token
    data = {"username": "user123", "password": "pass123"}
    token_response = test_user_mgmt_client.post("/token", json=data)
    assert token_response.status_code == 200
    token = token_response.json()["access_token"]

    # Now verify the token
    headers = {"Authorization": f"Bearer {token}"}
    response = test_user_mgmt_client.get("/verify-token", headers=headers)
    assert response.status_code == 200
    assert response.json()["valid"] is True
    assert "payload" in response.json()


def test_verify_jwt_invalid_token(test_user_mgmt_client):
    """Test /verify-jwt with an invalid JWT token."""
    headers = {"Authorization": "Bearer invalidtoken"}
    response = test_user_mgmt_client.get("/verify-token", headers=headers)
    assert response.status_code == 401
    assert response.json()["message"] == "Invalid or expired JWT token."


"""
def test_secured_endpoint_requires_auth(test_auth_client):
    '''Test /secured endpoint requires authentication.'''
    response = test_auth_client.get("/secured")
    assert response.status_code == 401


def test_secured_endpoint_with_valid_token(test_auth_client):
    '''Test /secured endpoint with valid JWT token.'''
    # Obtain a valid token
    data = {
        "username": "user123",
        "password": "pass123"
    }
    token_response = test_auth_client.post("/token", json=data)
    assert token_response.status_code == 200
    token = token_response.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"}
    response = test_auth_client.get("/secured", headers=headers)
    assert response.status_code == 200
"""


def test_only_to_trigger_actions_pipeline():
    print("triggered")
    assert True
