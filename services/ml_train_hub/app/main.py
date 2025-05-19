import logging

from fastapi import FastAPI, File, UploadFile
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


@app.get("/ping")
def pong():
    """Ping Pong."""
    return {"ping": "pong!"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


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


@app.get("/models/list")
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
