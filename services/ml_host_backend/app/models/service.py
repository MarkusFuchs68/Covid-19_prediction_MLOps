import gdown
import os
import logging
from app.exceptions.service_exceptions import GoogleDriveFolderEmptyException, ModelNotFoundException, GoogleDriveServiceException, GoogleDriveDownloadException
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

def Load_list_of_models_from_google_drive():
    """
    Function to load a list of models from Google Drive.
    """
    MODEL_FOLDER = os.path.join('.', 'models')
    DRIVE_URL = os.getenv("GOOGLE_DRIVE_URL")

    logger.info("Starting to load the list of models from Google Drive.")
    logger.debug(f"Model folder: {MODEL_FOLDER}, Drive URL: {DRIVE_URL}")

    try:
        file_list = gdown.download_folder(DRIVE_URL, output=MODEL_FOLDER,
                                          quiet=True, use_cookies=False,
                                          remaining_ok=True, skip_download=True)
        logger.info("Successfully retrieved the list of models.")
    except Exception as e:
        logger.error("Error occurred while loading models from Google Drive.", exc_info=True)
        raise GoogleDriveServiceException(
            "Could not load models from Google Drive."
        ) from e
    
    if not file_list:
        logger.warning("No files found in the Google Drive folder.")
        raise GoogleDriveFolderEmptyException(
            "No files found or invalid folder URL."
        )
    
    logger.debug(f"Retrieved file list: {file_list}")
    return file_list

def load_model_from_google_drive(model_name: str):
    """
    Function to load a model from Google Drive.
    """
    MODEL_FOLDER = os.path.join('.', 'models')
    DRIVE_URL = os.getenv("GOOGLE_DRIVE_URL")

    logger.info(f"Attempting to load model '{model_name}' from Google Drive.")
    logger.debug(f"Model folder: {MODEL_FOLDER}, Drive URL: {DRIVE_URL}")

    file_list = Load_list_of_models_from_google_drive()
    
    file_to_download = None
    for file in file_list:
        if model_name in file['name']:
            file_to_download = file
            break

    if not file_to_download:
        logger.error(f"Model '{model_name}' not found in the Google Drive folder.")
        raise ModelNotFoundException(f"File '{model_name}' not found in the Google Drive folder.")

    try:
        logger.info(f"Downloading model '{model_name}' with ID '{file_to_download['id']}'.")
        gdown.download(file_to_download['id'], os.path.join(MODEL_FOLDER, model_name), quiet=False)
        logger.info(f"Successfully downloaded model '{model_name}'.")
    except Exception as e:
        logger.error(f"Error occurred while downloading model '{model_name}'.", exc_info=True)
        raise GoogleDriveDownloadException(
            "Could not download the model from Google Drive."
        ) from e

    model_path = os.path.join(MODEL_FOLDER, model_name)
    logger.debug(f"Model saved at: {model_path}")
    return model_path