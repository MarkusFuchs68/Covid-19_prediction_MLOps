import logging

import ml_auth.app.exceptions.auth_exceptions as ae
from ml_auth.app.user_db import UserSchema
import ml_auth.app.jwt_handler
from fastapi import Request, Body, Depends
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from fastapi import status


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# our singleton MLFlow API
app = FastAPI()


@app.exception_handler(ae.FailedAuthentification)
async def handle_failed_authentification_exception(
    request: Request, exception: ae.FailedAuthentification
):
    return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"message": exception.message})


@app.exception_handler(ae.InvalidArgumentException)
async def handle_invalid_argument(
    request: Request, exception: ae.InvalidArgumentException
):
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": exception.message})


@app.get("/ping")
def pong():
    """Ping Pong."""
    return {"ping": "pong!"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


api = FastAPI()


@api.get("/secured", dependencies=[Depends(JWTBearer())], tags=["root"])
async def read_root_secured():
    return {"message": "Hello World! but secured"}


@api.post("/login", tags=["user"])
async def user_login(user: UserSchema = Body(...)):
    if check_user(user):
        return sign_jwt(user.email)
    return {"error": "Wrong login details!"}


def verify_jwt_token(token: str):
    try:
        payload = ml_auth.app.jwt_handler.decode_jwt(token)
        return payload if payload else None
    except Exception as e:
        logger.error(f"JWT verification failed: {e}")
        return None


@app.post("/verify-jwt", tags=["auth"])
async def verify_jwt_endpoint(token: str = Body(...)):
    payload = verify_jwt_token(token)
    if payload:
        return {"valid": True, "payload": payload}
    else:
        raise ae.FailedAuthentification(
            message="Invalid or expired JWT token."
        )
