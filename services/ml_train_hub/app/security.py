import logging
import os

import requests
from fastapi import HTTPException, status

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
logger.info(f"The ml_user_mgmt service is used at: {USER_MGMT_URL}")


def issue_jwt_token(username, password):
    # Call the ml_auth endpoint for token generation
    credentials = {"username": username, "password": password}
    resp = requests.post(USER_MGMT_TOKEN_ENDPOINT, json=credentials)
    if resp.status_code != status.HTTP_200_OK:
        logger.error("Failed to obtain JWT")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    logger.info(f"Obtained a JWT for user {username}")
    return resp.json().get("access_token")


def verify_jwt_with_user_mgmt(token: str):
    # Call the ml_auth endpoint for token verification
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(USER_MGMT_VERIFY_ENDPOINT, headers=headers)
    if resp.status_code != status.HTTP_200_OK:
        logger.error(f"Failed to verify JWT: {resp.status_code}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    logger.info(f"Successfully verified JWT for user: {resp.json()}")
    return resp.json()


def get_current_user(credentials):
    # This expects a HTTPAuthorizationCredentials = Security(HTTPBearer() object
    token = credentials.credentials
    if not token:
        logger.error("JWT missing")
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
