import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from ml_host_backend.app.exceptions.client_exceptions import InvalidArgumentException
from ml_host_backend.app.exceptions.service_exceptions import (
    MLFlowConfigurationException,
    MLFlowUnavailableException,
    ModelNotFoundException,
)
from ml_host_backend.app.logging_config import LOGGING_CONFIG
from ml_host_backend.app.routes.models import router as models_router

# init custom logging config
logging.config.dictConfig(LOGGING_CONFIG)

app = FastAPI()
logger = logging.getLogger(__name__)


@app.exception_handler(InvalidArgumentException)
async def handle_invalid_argument_exception(
    request: Request, exception: InvalidArgumentException
):
    return JSONResponse(status_code=400, content={"message": exception.message})


@app.exception_handler(ModelNotFoundException)
async def handle_model_not_found(request: Request, exception: ModelNotFoundException):
    return JSONResponse(status_code=404, content={"message": exception.message})


@app.exception_handler(MLFlowUnavailableException)
async def handle_mlflow_unavailable(
    request: Request, exception: MLFlowUnavailableException
):
    return JSONResponse(
        status_code=503, content={"message": "Service unavailable at the moment."}
    )


@app.exception_handler(MLFlowConfigurationException)
async def handle_mlflow_not_configured_correctly(
    request: Request, exception: MLFlowConfigurationException
):
    return JSONResponse(
        status_code=503, content={"message": "Service unavailable at the moment."}
    )


app.include_router(models_router, prefix="/api/models", tags=["models"])


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}
