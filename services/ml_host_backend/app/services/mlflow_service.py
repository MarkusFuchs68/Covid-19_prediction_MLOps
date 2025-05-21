import logging
import os

import requests
from ml_host_backend.app.exceptions.client_exceptions import InvalidArgumentException
from ml_host_backend.app.exceptions.service_exceptions import (
    MLFlowException,
    ModelNotFoundException,
)

logger = logging.getLogger(__name__)


def get_mlflow_host_and_port():
    mlflow_host = os.getenv("MLFLOW_HOST")
    mlflow_port = os.getenv("MLFLOW_PORT")
    if not mlflow_host or not mlflow_port:
        logger.error("MLFLOW_HOST or MLFLOW_PORT environment variable not set.")
        raise InvalidArgumentException(
            "MLFLOW_HOST or MLFLOW_PORT environment variable not set."
        )
    return mlflow_host, mlflow_port


def list_all_models_from_mlflow():
    logger.info("Fetching summary of all models from MLFlow.")
    mlflow_host, mlflow_port = get_mlflow_host_and_port()
    url = f"http://{mlflow_host}:{mlflow_port}/models"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "models" in data:
            return data["models"]
        return data
    except Exception as e:
        logger.error(f"Failed to fetch models summary from MLFlow: {e}", exc_info=True)
        raise MLFlowException(f"Failed to fetch models summary from MLFlow: {e}")


def get_single_model_summary_from_mlflow(model_name: str):
    logger.info(f"Fetching summary for model: {model_name} from MLFlow.")
    mlflow_host, mlflow_port = get_mlflow_host_and_port()
    url = f"http://{mlflow_host}:{mlflow_port}/models/{model_name}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            logger.warning(f"Model not found: {model_name}")
            raise ModelNotFoundException(message=f"Model '{model_name}' not found")
        response.raise_for_status()
        data = response.json()
        if not data:
            logger.warning(f"Model not found: {model_name}")
            raise ModelNotFoundException(message=f"Model '{model_name}' not found")
        logger.info(f"Model found: {model_name}")
        return data
    except ModelNotFoundException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch model summary from MLFlow: {e}", exc_info=True)
        raise MLFlowException(f"Failed to fetch model summary from MLFlow: {e}")
