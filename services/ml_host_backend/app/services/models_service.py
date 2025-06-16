import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from ml_host_backend.app.exceptions.client_exceptions import InvalidArgumentException
from ml_host_backend.app.services.meta import classes_2, classes_4, models_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DRIVE_URL = os.getenv("GOOGLE_DRIVE_URL")


def prepare_image_for_model(
    image, model, normalize=False
):  # Convert the image to TensorFlow tensor
    # Copy the image to avoid overwriting
    img_tf = tf.convert_to_tensor(image.copy())
    channels = model.input_shape[-1]  # Expected number of channels (1=gray, 3=RGB)
    target_size = model.input_shape[1:3]  # Expected image size (height, width)

    # If shape is 2D (H, W), expand to 3D (H, W, 1)
    if img_tf.ndim == 2:  # Grayscale (H, W)
        img_tf = tf.expand_dims(img_tf, axis=-1)  # (H, W, 1)

    # Convert grayscale to RGB if model expects 3 channels
    if img_tf.shape[-1] == 1 and channels == 3:
        img_tf = tf.image.grayscale_to_rgb(img_tf)
    elif img_tf.shape[-1] == 3 and channels == 1:
        img_tf = tf.image.rgb_to_grayscale(img_tf)

    # Resize to target model input size
    img_tf = tf.image.resize(img_tf, target_size)

    # Normalize to range [0, 1] if specified
    if normalize:
        img_tf = img_tf / 255.0

    return img_tf  # Return the processed image


def read_and_prepare_image(file_content, model):
    try:
        image = tf.image.decode_image(file_content, channels=1).numpy()
        logger.debug("Image decoded successfully.")
    except Exception as e:
        logger.error(f"Failed to decode image: {str(e)}", exc_info=True)
        raise InvalidArgumentException("Could not decode image.")

    image_array = prepare_image_for_model(image, model, normalize=True)
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


def list_summary_of_all_models(credentials):
    """
    Function to list details of all available models from MLFlow.
    """
    from ml_host_backend.app.services.mlflow_service import list_all_models_from_mlflow

    logger.info("Fetching summary of all models.")
    return list_all_models_from_mlflow(credentials)


def show_summary_of_single_model(model_name: str, credentials):
    """
    Function to get details of a single model.
    """
    from ml_host_backend.app.services.mlflow_service import (
        get_single_model_summary_from_mlflow,
    )

    logger.info(f"Fetching summary for model: {model_name}")
    return get_single_model_summary_from_mlflow(model_name, credentials)


def predict_image_classification(model_name, file_content, credentials):
    """
    Function to predict image classification using the specified model.
    """

    logger.info("Identifying model for prediction.")
    model_summary = show_summary_of_single_model(model_name, credentials)

    logger.info(f"Predicting image classification with model: {model_name}")

    model_path = model_summary["model_filepath"]
    model = tf.keras.models.load_model(model_path)

    logger.info(f"Preparing image for prediction with model: {model_name}")
    image_prepared = read_and_prepare_image(file_content, model)

    class_names = model_summary["class_names"]
    # the class names are a list of strings, e.g. ["cat", "dog", "bird"], we must decode it into a python list
    if isinstance(class_names, str):
        class_names = class_names.strip().split(",")
    if len(class_names) == 2:
        classes = classes_2
    elif len(class_names) == 4:
        classes = classes_4

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
