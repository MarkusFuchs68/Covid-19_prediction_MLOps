import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import patch
from app.routes.models import router
from unittest.mock import patch, MagicMock
import numpy as np

app = FastAPI()
app.include_router(router)

mock_model = MagicMock()    
mock_model.predict.return_value = np.array([[0.1, 0.0, 0.0, 0.0]])

dummy_mode_name = "dummy_model.keras"

@pytest.fixture
def client():
    return TestClient(app)

def test_get_summary_of_single_model_not_implemented(client):
    # This endpoint is not implemented, so it should return None or raise NotImplementedError
    response = client.get("/model1")
    assert response.status_code == 200
    assert response.json() is None

def test_make_prediction_for_image_success(client):
    # Create a dummy grayscale image
    from PIL import Image
    import io

    img = Image.new("L", (224, 224), color=128)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    with patch("app.services.models_service.tf.keras.models.load_model", return_value=mock_model):
        file = {"file": ("test.png", img_bytes, "image/png")}
        response = client.post(f"/{dummy_mode_name}/predict/", files=file)
        assert response.status_code == 200
        assert response.json()["Predicted"][dummy_mode_name] == "COVID"

def test_make_prediction_for_image_invalid_file(client):
    # Send a non-image file, mock prediction to raise an Exception
    with patch("app.services.models_service.tf.keras.models.load_model", return_value=mock_model):
        file = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post(f"/{dummy_mode_name}/predict/", files=file)
        assert response.status_code == 400
        # assert "unexpected error" in response.json()["detail"].lower()

# def test_make_prediction_for_image_missing_file(client):
#     # No file sent in the request
#     response = client.post("/model1/predict/")
#     assert response.status_code == 422  # Unprocessable Entity (missing required file)

# def test_make_prediction_for_large_image_file(client):
#     # Simulate a large file (but not actually large to avoid memory issues)
#     import io
#     large_content = b"\x00" * (1024 * 1024 * 2)  # 2MB dummy content
#     with patch("app.services.models_service.predict_image_classification_4_classes", return_value={"result": "classA"}):
#         files = {"file": ("large.png", io.BytesIO(large_content), "image/png")}
#         response = client.post("/model1/predict/", files=files)
#         # Depending on your service, this may succeed or fail; here we expect 200 due to mocking
#         assert response.status_code == 200

# def test_make_prediction_for_image_prediction_invalid_input(client):
#     # Simulate the prediction function raising an HTTPException
#     from fastapi import HTTPException
#     with patch("app.services.models_service.predict_image_classification_4_classes", side_effect=HTTPException(status_code=400, detail="Bad input")):
#         from PIL import Image
#         import io
#         img = Image.new("L", (224, 224), color=128)
#         img_bytes = io.BytesIO()
#         img.save(img_bytes, format="PNG")
#         img_bytes.seek(0)
#         files = {"file": ("test.png", img_bytes, "image/png")}
#         response = client.post("/model1/predict/", files=files)
#         assert response.status_code == 400
#         assert response.json()["detail"] == "Bad input"