import logging

import ml_train_hub.app.exceptions.client_exceptions as ce
import ml_train_hub.app.exceptions.service_exceptions as se
from fastapi import FastAPI, File, UploadFile
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from ml_train_hub.app.mlflow_util import (  # log_mlflow_experiment,
    get_mlflow_model,
    list_mlflow_models,
)

# from ml_train_hub.app.model_util import evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# our singleton MLFlow API
app = FastAPI()


@app.exception_handler(se.RegisterModelException)
async def handle_register_model_exception(
    request: Request, exception: se.RegisterModelException
):
    return JSONResponse(status_code=500, content={"message": exception.message})


@app.exception_handler(se.ModelNotFoundException)
async def handle_model_not_found(
    request: Request, exception: se.ModelNotFoundException
):
    return JSONResponse(status_code=404, content={"message": exception.message})


@app.exception_handler(se.ModelNotFoundInArtifactsException)
async def handle_model_not_found_in_artifacts(
    request: Request, exception: se.ModelNotFoundInArtifactsException
):
    return JSONResponse(status_code=404, content={"message": exception.message})


@app.exception_handler(ce.InvalidArgumentException)
async def handle_invalid_argument(
    request: Request, exception: ce.InvalidArgumentException
):
    return JSONResponse(status_code=400, content={"message": exception.message})


@app.get("/ping")
def pong():
    """Ping Pong."""
    return {"ping": "pong!"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/models/")
async def list_models():
    """
    This function returns a list of all available models in MLFlow
    """
    return {"models": list_mlflow_models()}


@app.get("/models/{model_name}")
async def get_model(model_name: str):
    """
    This function returns the information of the desired model and an MLFLow artifact pathname, from where to load.
    It also returns the human readable class names associated with the prediction index.

    Parameters:
    - model_name: The model to be retrieved
    """
    return get_mlflow_model(model_name)


@app.post("/models/{model_name}/register")
async def register_model(
    model_name: str, class_names: str, file: UploadFile = File(...)
):
    """
    This function evaluates the model via the evaluation set
    and registers it in MLFlow as a new version.

    Params:
    - model_name: Name, under which the model shall be registered
    - class_names: List of human readable class names associated with the prediction index
    - file: The model file to be uploaded
    """
    return {"run_name": "example_run_name"}
