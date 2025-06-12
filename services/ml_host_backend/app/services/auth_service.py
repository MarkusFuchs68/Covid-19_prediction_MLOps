import logging

import jwt
from fastapi import Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from ml_host_backend.app.exceptions.client_exceptions import UnauthroizedException

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
        raise UnauthroizedException("Token is expired.")
    if payload:
        return True
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


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer()),
):
    token = credentials.credentials
    if not token:
        logger.error("JWT missing")
        raise UnauthroizedException("Missing token")
    return verify_jwt(token)
