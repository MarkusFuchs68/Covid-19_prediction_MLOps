import logging
import os

import gdown
from dotenv import load_dotenv
from ml_host_backend.app.exceptions.service_exceptions import (
    GoogleDriveDownloadException,
    GoogleDriveFolderEmptyException,
    GoogleDriveServiceException,
    ModelNotFoundException,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_FOLDER = os.path.join(".", "data/models")
DRIVE_URL = os.getenv("GOOGLE_DRIVE_URL")


def get_list_of_models_from_google_drive():
    """
    Function to load a list of models from Google Drive.
    """

    # Ensure the model folder exists
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    logger.info("Starting to load the list of models from Google Drive.")
    logger.debug(f"Model folder: {MODEL_FOLDER}, Drive URL: {DRIVE_URL}")

    try:
        file_list = gdown.download_folder(
            DRIVE_URL,
            output=MODEL_FOLDER,
            quiet=True,
            use_cookies=False,
            remaining_ok=True,
            skip_download=True,
        )
        logger.info("Successfully retrieved the list of models.")
    except Exception as e:
        logger.error(
            "Error occurred while loading models from Google Drive.", exc_info=True
        )
        raise GoogleDriveServiceException(
            "Could not load models from Google Drive."
        ) from e

    if not file_list:
        logger.warning("No files found in the Google Drive folder.")
        raise GoogleDriveFolderEmptyException("No files found or invalid folder URL.")

    logger.debug(f"Retrieved file list: {file_list}")
    return file_list


def download_model_from_google_drive(model_file_name: str):
    """
    Function to load a model from Google Drive.
    """

    logger.info(f"Attempting to load model '{model_file_name}' from Google Drive.")
    logger.debug(f"Model folder: {MODEL_FOLDER}, Drive URL: {DRIVE_URL}")

    file_list = get_list_of_models_from_google_drive()

    file_to_download = None
    for file in file_list:
        if model_file_name == file[1]:
            file_to_download = file
            break

    if not file_to_download:
        logger.error(f"Model '{model_file_name}' not found in the Google Drive folder.")
        raise ModelNotFoundException(
            f"File '{model_file_name}' not found in the Google Drive folder."
        )

    # Construct the download URL using image ID
    output = os.path.join(MODEL_FOLDER, model_file_name)

    try:
        logger.info(
            f"Downloading model '{model_file_name}' with ID '{file_to_download[0]}'."
        )
        gdown.download(id=file_to_download[0], output=output, quiet=False)
        logger.info(f"Successfully downloaded model '{model_file_name}'.")
    except Exception as e:
        logger.error(
            f"Error occurred while downloading model '{model_file_name}'.",
            exc_info=True,
        )
        raise GoogleDriveDownloadException(
            "Could not download the model from Google Drive."
        ) from e

    model_path = os.path.join(MODEL_FOLDER, model_file_name)
    logger.debug(f"Model saved at: {model_path}")
    return model_path
