import logging

import ml_train_hub.app.exceptions.client_exceptions as ce
import ml_train_hub.app.exceptions.service_exceptions as se
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from ml_train_hub.app.mlflow_util import (
    get_mlflow_model,
    list_mlflow_models,
    log_mlflow_experiment,
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
    model_filepath: str,
    model_name: str,
    class_names: list[str],
    experiment_name: str = "Covid_Models",
):
    """
    Evaluates the model using the evaluation set and registers it in MLflow as a new version.

    Args:
    - model_filepath (str): Path to the trained model file.
    - model_name (str): Name under which the model will be registered.
    - class_names (list[str]): List of human-readable class names associated with the prediction indices.
    - experiment_name (str, optional): Name of the MLflow experiment. Defaults to "Covid_Models".

    Returns:
    - dict: Contains information about the registered run (e.g., run name).
    """

    # TODO: consider making this async, so the caller doesn't time out
    # TODO: read architecture from model
    architecture = dict(
        {
            "layer0": "Conv2D(32, (3, 3), activation='relu')",
            "layer1": "MaxPooling2D((2, 2))",
        }
    )
    # TODO: evaulate metrics with evaluation set
    metrics = dict({"performance": 0.85})

    return log_mlflow_experiment(
        model_filepath=model_filepath,
        architecture=architecture,
        metrics=metrics,
        class_names=class_names,
        experiment_name=experiment_name,
        register_model=True,
        model_name=model_name,
    )
