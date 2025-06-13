import logging
import os
import time

import jwt
import requests
from fastapi import Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from ml_host_backend.app.exceptions.client_exceptions import UnauthroizedException
from ml_host_backend.app.exceptions.service_exceptions import (
    MLUserMgmtConfigurationException,
    MLUserMgmtException,
    MLUserMgmtUnavailableException,
)

JWT_SECRET = "secret"  # this should be specified in a vault in a real application
JWT_ALGORITHM = "HS256"

logger = logging.getLogger(__name__)


def verify_jwt(jwtoken: str):
    try:
        payload = jwt.decode(jwtoken, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.exceptions.InvalidTokenError:
        logger.error("Invalid token.")
        raise UnauthroizedException("Invalid token.")
    except jwt.exceptions.ExpiredSignatureError:
        logger.error("Token is expired.")
        raise UnauthroizedException("Signature has expired.")
    if payload:
        if payload["expires"] >= time.time():
            return True
        raise UnauthroizedException("Token is expired.")
    logger.error("Unable to retrieve token payload")
    raise UnauthroizedException("Unable to retrieve token payload")


class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super(
            JWTBearer, self
        ).__call__(request)
        if credentials:
            if not credentials.scheme == "Bearer":
                raise UnauthroizedException("Invalid authentication scheme.")
            if not verify_jwt(credentials.credentials):
                raise UnauthroizedException("Invalid token or expired token.")
            return credentials.credentials
        else:
            raise UnauthroizedException("Invalid authorization code.")


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer()),
):
    token = credentials.credentials
    if not token:
        logger.error("JWT missing")
        raise UnauthroizedException("Missing token")
    return verify_jwt(token)


def get_ml_user_mgmt_host_and_port():
    ml_user_mgmt_host = os.getenv("ML_USER_MGMT_HOST")
    ml_user_mgmt_port = os.getenv("ML_USER_MGMT_PORT")
    if not ml_user_mgmt_host or not ml_user_mgmt_port:
        logger.error(
            "ML_USER_MGMT_HOST or ML_USER_MGMT_PORT environment variable not set."
        )
        raise MLUserMgmtConfigurationException(
            "ML_USER_MGMT_HOST or MML_USER_MGMT_PORTLFLOW_PORT environment variable not set."
        )
    return ml_user_mgmt_host, ml_user_mgmt_port


def check_service_availability_or_throw():
    ml_user_mgmt_host, ml_user_mgmt_port = get_ml_user_mgmt_host_and_port()
    url = f"http://{ml_user_mgmt_host}:{ml_user_mgmt_port}/health"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        logger.info("ML User Mgmt service is available.")
        return ml_user_mgmt_host, ml_user_mgmt_port
    except requests.exceptions.ConnectionError as e:
        logger.error(
            f"ML User Mgmt service is not available at provided host and port: {e}",
            exc_info=True,
        )
        raise MLUserMgmtUnavailableException(
            "ML User Mgmt service is not available at the provided host and port."
        )


def login_user(username: str, password: str):
    logger.info("Beginning User Login")
    ml_user_mgmt_host, ml_user_mgmt_port = check_service_availability_or_throw()
    url = f"http://{ml_user_mgmt_host}:{ml_user_mgmt_port}/token"
    logger.info(f"Requesting token from ML User Mgmt at: {url}")
    try:
        response = requests.post(
            url, json={"username": username, "password": password}, timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.HTTPError as err:
        logger.error(f"Failed to login user: {err}", exc_info=True)
        raise UnauthroizedException("Incorrect username or password")
    except Exception as e:
        logger.error(f"Failed to login user: {e}", exc_info=True)
        raise MLUserMgmtException(f"Failed to login user: {e}")
