import logging
import os

import requests
from fastapi import HTTPException, status

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_env_variable(name: str, default=None):
    value = os.environ.get(name, default)
    # if value is None:
    #     raise RuntimeError(f"Environment variable '{name}' is not set")
    return value


running_stage = get_env_variable("RUNNING_STAGE", "dev")
if running_stage == "prod":
    ML_HOST_BACKEND_PORT = 8080  # production port
    ML_TRAIN_HUB_PORT = 8082  # production port
else:
    ML_HOST_BACKEND_PORT = 8000  # dev or test port
    ML_TRAIN_HUB_PORT = 8002  # dev or test port

if os.path.exists(
    "/.dockerenv"
):  # When we are inside the container, we must use the container network name
    HOST_BACKEND_URL = f"http://ml_host_backend_{running_stage}:{ML_HOST_BACKEND_PORT}"
    TRAIN_HUB_URL = f"http://ml_train_hub_{running_stage}:{ML_TRAIN_HUB_PORT}"
else:
    HOST_BACKEND_URL = f"http://localhost:{ML_HOST_BACKEND_PORT}"
    TRAIN_HUB_URL = f"http://localhost:{ML_TRAIN_HUB_PORT}"

logger.info(f"Using ML Host Backend URL: {HOST_BACKEND_URL}")
logger.info(f"Using ML Train Hub URL: {TRAIN_HUB_URL}")

TRAIN_HUB_ENDPOINT_REGISTERMODEL = TRAIN_HUB_URL + "/models/{model_name}/register"
HOST_BACKEND_ENDPOINT_LISTMODELS = HOST_BACKEND_URL + "/api/models"
HOST_BACKEND_ENDPOINT_GETMODEL = HOST_BACKEND_URL + "/api/models/{model_name}"
HOST_BACKEND_ENDPOINT_PREDICT = HOST_BACKEND_URL + "/api/models/{model_name}/predict"
HOST_BACKEND_ENDPOINT_LOGIN = HOST_BACKEND_URL + "/api/models/login"


def login(username, password):
    # Call the ml_auth endpoint for token generation
    credentials = {"username": username, "password": password}
    resp = requests.post(HOST_BACKEND_ENDPOINT_LOGIN, json=credentials)
    if resp.status_code != status.HTTP_200_OK:
        logger.error("Failed to obtain JWT")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    logger.info(f"Obtained a JWT for user {username}")
    return resp.json().get("access_token")


def register_model(
    token: str,
    model_filepath: str,
    model_name: str,
    class_names: list[str],
    experiment_name: str,
    max_num: int,
):
    Headers = {"Authorization": f"Bearer {token}"}
    data = {
        "model_filepath": model_filepath,
        "model_name": model_name,
        "class_names": class_names,
        "experiment_name": experiment_name,
        "max_num": max_num,
    }
    resp = requests.post(
        TRAIN_HUB_ENDPOINT_REGISTERMODEL.format(model_name=model_name),
        headers=Headers,
        json=data,
    )
    if resp.status_code != status.HTTP_200_OK:
        logger.error(f"Failed to register model: {resp.status_code} - {resp.text}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to register model: {resp.text}",
        )
    logger.info(f"Model {model_name} registered successfully")
    return resp.json()


def list_models(token: str):
    Headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(HOST_BACKEND_ENDPOINT_LISTMODELS, headers=Headers)
    if resp.status_code != status.HTTP_200_OK:
        logger.error(f"Failed to list models: {resp.status_code} - {resp.text}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to list models: {resp.text}",
        )
    logger.info("Successfully retrieved list of models")
    return resp.json().get("models", [])


def get_model(token: str, model_name: str):
    Headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        HOST_BACKEND_ENDPOINT_GETMODEL.format(model_name=model_name), headers=Headers
    )
    if resp.status_code != status.HTTP_200_OK:
        logger.error(
            f"Failed to get model {model_name}: {resp.status_code} - {resp.text}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get model {model_name}: {resp.text}",
        )
    logger.info(f"Successfully retrieved model {model_name}")
    return resp.json()


def predict(token: str, model_name: str, data: dict):
    Headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(
        HOST_BACKEND_ENDPOINT_PREDICT.format(model_name=model_name),
        headers=Headers,
        json=data,
    )
    if resp.status_code != status.HTTP_200_OK:
        logger.error(
            f"Prediction failed for model {model_name}: {resp.status_code} - {resp.text}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed for model {model_name}: {resp.text}",
        )
    logger.info(f"Prediction successful for model {model_name}")
    return resp.json()
