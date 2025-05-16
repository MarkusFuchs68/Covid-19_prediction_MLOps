import logging

from fastapi import FastAPI, File, UploadFile

"""
from ml_train_hub.app.mlflow_util import (
    list_mlflow_models,
    load_mlflow_model,
    log_mlflow_experiment,
)
from ml_train_hub.app.model_util import evaluate_model
"""
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
async def register_model(model_name: str, file: UploadFile = File(...)):
    """
    This function evaluates the model via the evaluation set
    and registers it in MLFlow as a new version.
    """


@app.get("/models/list")
async def list_models():
    """
    This function returns a list of all available models in MLFlow
    """


@app.get("/models/{model_name}")
async def load_model():
    """
    This function returns the information of the desired model and a pathname, from where to load
    """
