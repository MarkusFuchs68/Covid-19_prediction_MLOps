import io
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from ml_host_backend.app.exceptions.client_exceptions import InvalidArgumentException
from ml_host_backend.app.exceptions.service_exceptions import ModelNotFoundException
from ml_host_backend.app.services.meta import classes_4, models_summary
from PIL import Image, UnidentifiedImageError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_FOLDER = "/home/services/file_exchange"
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
    logger.debug(
        "Image converted to numpy array, normalized, and channel dimension added."
    )

    return image_array


def download_latest_model_version(model_name: str):
    """
    TODO: Implement
    Function to load the latest version of a model from MLFlow.
    """
    return models_summary


def list_summary_of_all_models():
    """
    Function to list details of all available models from MLFlow.
    """
    from ml_host_backend.app.services.mlflow_service import list_all_models_from_mlflow

    logger.info("Fetching summary of all models.")
    return list_all_models_from_mlflow()


def show_summary_of_single_model(model_name: str):
    """
    Function to get details of a single model.
    """
    from ml_host_backend.app.services.mlflow_service import (
        get_single_model_summary_from_mlflow,
    )

    logger.info(f"Fetching summary for model: {model_name}")
    return get_single_model_summary_from_mlflow(model_name)


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
    # model_path = load_latest_model_version(model_name)

    model_file_name = model_name + ".keras"
    model_path = os.path.join(MODEL_FOLDER, model_file_name)

    model = tf.keras.models.load_model(model_path)

    image_batch = tf.expand_dims(image_prepared, axis=0)
    pred = model.predict(image_batch)[0]

    pred_df = pd.DataFrame(columns=["Predicted"] + classes)
    pred_df.loc[model_name] = [classes[np.argmax(pred)]] + pred.round(3).astype(
        "str"
    ).tolist()  # Write prediction into table
    pred_df.index.name = "Model"

    logger.info(f"Prediction completed for model: {model_path}")
    # Return the prediction report
    return pred_df
