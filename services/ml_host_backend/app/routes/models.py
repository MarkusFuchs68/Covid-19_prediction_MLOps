from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
import logging
from app.services.models_service import predict_image_classification_4_classes, list_summary_of_all_models
import numpy as np
from app.exceptions.client_exceptions import InvalidArgumentException

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
def get_summary_of_all_models():
    """
    Function to list details of all available models.
    """
    
    logger.info("Fetching summary of all models.")
    return list_summary_of_all_models()

@router.get("/{model_name}")
def get_summary_of_single_model(model_name: str):
    """
    Function to get details of a single model.
    """


@router.post("/{model_name}/predict/")
async def make_prediction_for_image(model_name: str, file: UploadFile = File(...)):
    logger.info(f"Starting prediction for model: {model_name}")
    file_content = await file.read()
    logger.debug(f"File {file.filename} read successfully. Size: {len(file_content)} bytes")

    logger.info(f"Performing prediction using model: {model_name}")
    result = predict_image_classification_4_classes(model_name, file_content)
    logger.info(f"Prediction completed for model: {model_name}")

    return result
