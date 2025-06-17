import logging
import logging.config

import ml_train_hub.app.exceptions.client_exceptions as ce
import ml_train_hub.app.exceptions.service_exceptions as se
from fastapi import BackgroundTasks, FastAPI, Security
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from ml_train_hub.app.logging_config import LOGGING_CONFIG
from ml_train_hub.app.mlflow_util import (
    evaluate_and_log_metrics,
    get_mlflow_model,
    list_mlflow_models,
    log_mlflow_experiment,
)
from ml_train_hub.app.security import get_current_user
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
# init custom logging config
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# our singleton MLFlow API
app = FastAPI()

# setup Prometheus instrumentator
Instrumentator().instrument(app).expose(app)


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
async def list_models(
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer()),
):
    """
    This function returns a list of all available models in MLFlow
    """

    # Verify the JWT via ml_user_mgmt/verify-token endpoint
    get_current_user(credentials)

    return {"models": list_mlflow_models()}


@app.get("/models/{model_name}")
async def get_model(
    model_name: str, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
):
    """
    This function returns the information of the desired model and an MLFLow artifact pathname, from where to load.
    It also returns the human readable class names associated with the prediction index.

    Parameters:
    - model_name: The model to be retrieved
    """

    # Verify the JWT via ml_user_mgmt/verify-token endpoint
    get_current_user(credentials)

    return get_mlflow_model(model_name)


@app.post("/models/{model_name}/register")
async def register_model(
    model_filepath: str,
    model_name: str,
    class_names: list[str],
    experiment_name: str,
    max_num: int,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer()),
):
    """
    Evaluates the model using the evaluation set and registers it in MLflow as a new version.

    Args:
    - model_filepath (str): Path to the trained model file. Note: docker container shares folder 'file_exchange', put your model files into file_exchange and specify e.g. 'file_exchange/my_model.keras'. Only *.keras model files are supported!
    - model_name (str): Name under which the model will be registered.
    - experiment_name (str): Name of the MLflow experiment. If empty, it defaults to "Covid_Models".
    - max_num (int): Maximum number of predictions in order to respect server ressources, if 0 then all evaluation data is used.
    - class_names (list[str]): List of human-readable class names associated with the prediction indices in json-format in the request body, example: ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"].

    Returns:
    - dict: Contains information about the registered run (e.g., run name).
    """

    # Verify the JWT via ml_user_mgmt/verify-token endpoint
    get_current_user(credentials)

    # Optionally default to our standard experiment_name
    if experiment_name and experiment_name == "":
        experiment_name = "Covid_Models"

    # First register our model in a synchronous call (takes a few seconds)
    modelinfo = log_mlflow_experiment(
        model_filepath=model_filepath,
        class_names=class_names,
        experiment_name=experiment_name,
        register_model=True,
        model_name=model_name,
    )

    # If until here no exception occurred, the model is registered successfully
    # In a background task start the evaluation of the model, which adds additional data to it.
    # This takes some time, hence we let this do in a background task.
    background_tasks.add_task(evaluate_and_log_metrics, modelinfo, class_names, max_num)
    logger.info(
        f"Triggered background process for model architecture and metrics for run_id: {modelinfo.run_id}"
    )

    # Don't wait for this, just return the registered modelinfo to the caller
    return modelinfo
