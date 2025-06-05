import os

import requests
from fastapi import HTTPException, status


def get_env_variable(name: str, default=None):
    value = os.environ.get(name, default)
    # if value is None:
    #     raise RuntimeError(f"Environment variable '{name}' is not set")
    return value


running_stage = get_env_variable("RUNNING_STAGE", "dev")
if running_stage == "prod":
    USER_MGMT_PORT = 8083  # production port
else:
    USER_MGMT_PORT = 8003  # dev or test port

if os.path.exists(
    "/.dockerenv"
):  # When we are inside the container, we must use the container network name
    USER_MGMT_URL = f"http://ml_user_mgmt_{running_stage}:{USER_MGMT_PORT}"
else:
    USER_MGMT_URL = f"http://localhost:{USER_MGMT_PORT}"

USER_MGMT_TOKEN_ENDPOINT = USER_MGMT_URL + "/token"
USER_MGMT_VERIFY_ENDPOINT = USER_MGMT_URL + "/verify-token"


def issue_jwt_token(username, password):
    # Call the ml_auth endpoint for token generation
    credentials = {"username": username, "password": password}
    resp = requests.post(USER_MGMT_TOKEN_ENDPOINT, json=credentials)
    if resp.status_code != status.HTTP_200_OK:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    return resp.json().get("access_token")


def verify_jwt_with_user_mgmt(token: str):
    # Call the ml_auth endpoint for token verification
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(USER_MGMT_VERIFY_ENDPOINT, headers=headers)
    if resp.status_code != status.HTTP_200_OK:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    return resp.json()


def get_current_user(credentials):
    # This expects a HTTPAuthorizationCredentials = Security(HTTPBearer() object
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    return verify_jwt_with_user_mgmt(token)


"""
# This is a demo, how to create a secured endpoint, which expects a JWT:

from <your_service>.app.security import get_current_user
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security

@app.get("/secured")
async def secure_dummy(
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
):
    user = get_current_user(credentials)  # This authenticates the user via a call to to ml_auth/verify-token
    return {"message": f"Hello, {user['user']}!"}
"""
