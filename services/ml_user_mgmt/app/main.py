import logging
import os

print("Runtime root folder:", os.getcwd())
import ml_user_mgmt.app.exceptions.auth_exceptions as ae
from fastapi import Body, FastAPI, Request, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from ml_user_mgmt.app.jwt_handler import decode_jwt, sign_jwt
from ml_user_mgmt.app.logging_config import LOGGING_CONFIG
from ml_user_mgmt.app.user_db import UserDb, UserSchema
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
# init custom logging config
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# our singleton MLFlow API
app = FastAPI()

# setup Prometheus instrumentator
Instrumentator().instrument(app).expose(app)

# our singleton user DB
user_db = UserDb()


@app.exception_handler(ae.FailedAuthentification)
async def handle_failed_authentification_exception(
    request: Request, exception: ae.FailedAuthentification
):
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED, content={"message": exception.message}
    )


@app.exception_handler(ae.InvalidArgumentException)
async def handle_invalid_argument(
    request: Request, exception: ae.InvalidArgumentException
):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, content={"message": exception.message}
    )


@app.get("/ping")
def pong():
    """Ping Pong."""
    return {"ping": "pong!"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/token")
async def create_token(user: UserSchema = Body(...)):
    if user_db.check_user(user):
        logger.info(f"Created JWT for user {user.username}.")
        return sign_jwt(user.username)
    else:
        logger.info(
            f"User {user.username} tried to get JWT, but credentials are wrong."
        )
        raise ae.FailedAuthentification(message="Wrong login credentials!")


def verify_jwt_token(token: str):
    try:
        payload = decode_jwt(token)
        return payload if payload else None
    except Exception as e:
        logger.error(f"JWT verification failed: {e}")
        return None


@app.get("/verify-token")
async def verify_jwt(
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer()),
):
    token = credentials.credentials
    payload = verify_jwt_token(token)
    if payload:
        logger.info(f"Verified JWT: {payload}")
        return {"valid": True, "payload": payload}
    else:
        raise ae.FailedAuthentification(message="Invalid or expired JWT token.")


# For debugging
if __name__ == "__main__":
    user = UserSchema(username="user123", password="pass123")
    print(sign_jwt(user.username))
