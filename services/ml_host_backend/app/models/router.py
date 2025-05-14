from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from PIL import Image
import io
import logging
from app.models.meta import models_summary
from app.models.service import load_model_from_google_drive

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
def list_summary_of_all_models():
    """
    Function to list details of all available models.
    """
    logger.info("Fetching summary of all models.")
    return models_summary

@router.get("/{model_name}")
def get_summary_of_single_model(model_name: str):
    """
    Function to get details of a single model.
    """
    logger.info(f"Fetching summary for model: {model_name}")
    for model in models_summary:
        if model["name"] == model_name:
            logger.info(f"Model found: {model_name}")
            return model
    logger.warning(f"Model not found: {model_name}")
    raise HTTPException(status_code=404, detail="Model not found")

@router.post("{model_name}/predict/")
async def process_image(model_name: str, file: UploadFile = File(...)):
    """
    API endpoint to upload an image and predict Covid19 classification.
    """
    logger.info(f"Predicting image classification using model: {model_name}")
    try:
        # Read the uploaded file
        contents = await file.read()
        logger.debug(f"File {file.filename} uploaded successfully.")

        # Open the image using PIL
        image = Image.open(io.BytesIO(contents))
        logger.debug("Image opened successfully.")

        # Perform an operation (e.g., convert to grayscale)
        grayscale_image = image.convert("L")
        logger.debug("Image converted to grayscale.")

        # Save the processed image to a buffer
        buffer = io.BytesIO()
        grayscale_image.save(buffer, format="PNG")
        buffer.seek(0)
        logger.info(f"Image processing completed for file: {file.filename}")

        load_model_from_google_drive(model_name)

        ## TODO: Add model prediction logic here
        ######### -> 

        # For demonstration, we will just return a dummy response
        return {"result": True, "message": "Covid19 detected", "model_name": model_name}
    except Exception as e:
        logger.error(f"Error processing image for model {model_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")