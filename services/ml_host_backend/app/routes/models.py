import logging

from fastapi import APIRouter, File, Security, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from ml_host_backend.app.services.auth_service import get_current_user
from ml_host_backend.app.services.models_service import (
    download_latest_model_version,
    list_summary_of_all_models,
    predict_image_classification_4_classes,
    show_summary_of_single_model,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
def get_summary_of_all_models(
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer()),
):
    """
    Function to list details of all available models.
    """
    # Verify the JWT token
    get_current_user(credentials)

    logger.info("Fetching summary of all models.")
    return list_summary_of_all_models(credentials)


@router.get("/{model_name}")
def get_summary_of_single_model(
    model_name: str, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
):
    """
    Function to get details of a single model.
    """
    # Verify the JWT token
    get_current_user(credentials)

    return show_summary_of_single_model(model_name, credentials)


@router.post("/{model_name}/download")
def download_model(
    model_name: str, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
):
    """
    Function to download a model
    """
    # Verify the JWT token
    get_current_user(credentials)

    logger.info(f"Downloading model: {model_name}")
    download_latest_model_version(model_name)
    logger.info(f"Model {model_name} downloaded successfully.")
    return {"message": f"Model {model_name} downloaded successfully."}


@router.post("/{model_name}/predict/")
async def make_prediction_for_image(
    model_name: str,
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer()),
):
    # Verify the JWT token
    get_current_user(credentials)

    logger.info(f"Starting prediction for model: {model_name}")
    file_content = await file.read()
    logger.debug(
        f"File {file.filename} read successfully. Size: {len(file_content)} bytes"
    )

    logger.info(f"Performing prediction using model: {model_name}")
    result = predict_image_classification_4_classes(
        model_name, file_content, credentials
    )
    logger.info(f"Prediction completed for model: {model_name}")

    return result
