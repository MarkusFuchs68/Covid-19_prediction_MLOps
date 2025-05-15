import gdown
import os
import logging
from app.exceptions.service_exceptions import GoogleDriveFolderEmptyException, ModelNotFoundException, GoogleDriveServiceException, GoogleDriveDownloadException
from app.exceptions.client_exceptions import InvalidArgumentException
from dotenv import load_dotenv
import tensorflow as tf
from app.services.meta import classes_2, classes_4, models_summary
import pandas as pd
import numpy as np
from PIL import Image
import io
from PIL import UnidentifiedImageError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_FOLDER = os.path.join('.', 'data/models')
DRIVE_URL = os.getenv("GOOGLE_DRIVE_URL")

def read_and_prepare_image(file_content):
    try:
        image = Image.open(io.BytesIO(file_content))
        logger.debug("Image opened successfully.")
    except UnidentifiedImageError as e:
        logger.error(f"Failed to open image: {str(e)}", exc_info=True)
        raise InvalidArgumentException("Invalid image file format.")

    if image.mode != "L":
        image = image.convert("L")
        logger.debug("Image converted to grayscale.")

    processed_image = image.resize((224, 224))
    logger.debug("Image resized to 224x224.")

    image_array = np.array(processed_image) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # Ensure shape (224, 224, 1)
    logger.debug("Image converted to numpy array, normalized, and channel dimension added.")

    return image_array

def load_list_of_models_from_google_drive():
    """
    Function to load a list of models from Google Drive.
    """

    # Ensure the model folder exists
    os.makedirs(MODEL_FOLDER, exist_ok=True)

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

def load_model_from_google_drive(model_file_name: str):
    """
    Function to load a model from Google Drive.
    """

    logger.info(f"Attempting to load model '{model_file_name}' from Google Drive.")
    logger.debug(f"Model folder: {MODEL_FOLDER}, Drive URL: {DRIVE_URL}")

    file_list = load_list_of_models_from_google_drive()
    
    file_to_download = None
    for file in file_list:
        if model_file_name == file[1]:
            file_to_download = file
            break

    if not file_to_download:
        logger.error(f"Model '{model_file_name}' not found in the Google Drive folder.")
        raise ModelNotFoundException(f"File '{model_file_name}' not found in the Google Drive folder.")

    # Construct the download URL using image ID
    url = f"https://drive.google.com/file/d/{file_to_download[0]}/view?usp=sharing"
    output = os.path.join(MODEL_FOLDER, model_file_name)

    try:
        logger.info(f"Downloading model '{model_file_name}' with ID '{file_to_download[0]}'.")
        gdown.download(id = file_to_download[0], output = output, quiet=False)
        logger.info(f"Successfully downloaded model '{model_file_name}'.")
    except Exception as e:
        logger.error(f"Error occurred while downloading model '{model_file_name}'.", exc_info=True)
        raise GoogleDriveDownloadException(
            "Could not download the model from Google Drive."
        ) from e

    model_path = os.path.join(MODEL_FOLDER, model_file_name)
    logger.debug(f"Model saved at: {model_path}")
    return model_path

def load_latest_model_version(model_name: str):
    """
    TODO: Implement
    Function to load the latest version of a model from MLFlow.
    """
    return models_summary

def list_summary_of_all_models():
    """
    Function to list details of all available models from MLFlow.
    TODO: Implement
    """
    logger.info("Fetching summary of all models.")
    return models_summary

def show_summary_of_single_model(model_name: str):
    """
    Function to get details of a single model.
    """
    logger.info(f"Fetching summary for model: {model_name}")
    all_models = list_summary_of_all_models()

    for model in all_models:
        if model["name"] == model_name:
            logger.info(f"Model found: {model_name}")
            return model
    logger.warning(f"Model not found: {model_name}")
    raise ModelNotFoundException(message="Model not found")

def predict_image_classification_4_classes(model_name, file_content):
    """
    Function to predict image classification using the specified model.
    """

    logger.info(f"Preparing image for prediction with model: {model_name}")
    image_prepared = read_and_prepare_image(file_content)
    classes = classes_4

    logger.info("Identifying model for prediction.")
    models_summary = list_summary_of_all_models()
    
    if model_name not in [model["name"] for model in models_summary]:
        logger.error(f"Model '{model_name}' not found.")
        raise ModelNotFoundException(f"Model '{model_name}' not found.")
    logger.info(f"Predicting image classification with model: {model_name}")

    # TODO: Load model from MLFlow api once ready
    #model_path = load_latest_model_version(model_name)

    model_file_name = model_name + ".keras"
    model_path = os.path.join(MODEL_FOLDER, model_file_name)

    # teporary solution, the following function can be used to load the model from Google Drive
    # if already loaded, it can be skipped
    if not os.path.exists(model_path):
        logger.error(f"Model file '{model_path}' does not exist.")
        load_model_from_google_drive(model_file_name)

    model = tf.keras.models.load_model(model_path)

    image_batch = tf.expand_dims(image_prepared, axis=0)
    pred = model.predict(image_batch)[0]

    pred_df = pd.DataFrame(columns=["Predicted"] + classes)
    pred_df.loc[model_name] = [classes[np.argmax(pred)]] + pred.round(3).astype('str').tolist() # Write prediction into table
    pred_df.index.name = 'Model'

    logger.info(f"Prediction completed for model: {model_path}")
    # Return the prediction report
    return pred_df 