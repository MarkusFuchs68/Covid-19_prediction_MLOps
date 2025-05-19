import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from ml_host_backend.app.main import app
from ml_host_backend.app.services.meta import models_summary
from PIL import Image

mock_model = MagicMock()
mock_model.predict.return_value = np.array([[0.1, 0.0, 0.0, 0.0]])
base_endpoint = "/api/models"


@pytest.fixture
def client():
    return TestClient(app)


def test_get_summary_of_single_model(client):
    available_model = models_summary[0]["name"]
    response = client.get(f"{base_endpoint}/{available_model}")
    print(response.json())
    assert response.status_code == 200
    assert response.json() == models_summary[0]


def test_make_prediction_for_image_success(client):
    img = Image.new("L", (224, 224), color=128)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    available_model = models_summary[0]["name"]
    file = {"file": ("test.png", img_bytes, "image/png")}
    with patch(
        "ml_host_backend.app.services.models_service.download_model_from_google_drive",
        return_value=None,
    ), patch(
        "ml_host_backend.app.services.models_service.tf.keras.models.load_model",
        return_value=mock_model,
    ):
        response = client.post(f"{base_endpoint}/{available_model}/predict", files=file)
        assert response.status_code == 200
        assert response.json()["Predicted"][available_model] == "COVID"


def test_make_prediction_for_image_invalid_file(client):
    with patch(
        "ml_host_backend.app.services.models_service.download_model_from_google_drive",
        return_value=None,
    ), patch(
        "ml_host_backend.app.services.models_service.tf.keras.models.load_model",
        return_value=mock_model,
    ):
        file = {"file": ("test.txt", b"not an image", "text/plain")}
        dummy_model_name = "some name"
        response = client.post(
            f"{base_endpoint}/{dummy_model_name}/predict", files=file
        )
        assert response.status_code == 400
        assert "invalid image file format." in response.json()["message"].lower()


def test_make_prediction_for_image_missing_file(client):
    available_model = models_summary[0]["name"]
    response = client.post(f"{base_endpoint}/{available_model}/predict")
    assert response.status_code == 422


def test_make_prediction_for_large_image_file(client):
    img = Image.new("L", (4000, 4000), color=128)  # large image
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    available_model = models_summary[0]["name"]
    with patch(
        "ml_host_backend.app.services.models_service.download_model_from_google_drive",
        return_value=None,
    ), patch(
        "ml_host_backend.app.services.models_service.tf.keras.models.load_model",
        return_value=mock_model,
    ):
        files = {"file": ("large.png", img_bytes, "image/png")}
        response = client.post(
            f"{base_endpoint}/{available_model}/predict", files=files
        )
        assert response.status_code == 200


def test_make_prediction_for_image_prediction_invalid_input(client):
    file_content = b"This is not an image file, just named as one."
    files = {"file": ("test.png", io.BytesIO(file_content))}
    available_model = models_summary[0]["name"]

    with patch(
        "ml_host_backend.app.services.models_service.download_model_from_google_drive",
        return_value=None,
    ), patch(
        "ml_host_backend.app.services.models_service.tf.keras.models.load_model",
        return_value=mock_model,
    ):
        response = client.post(
            f"{base_endpoint}/{available_model}/predict", files=files
        )
        assert response.status_code == 400
        assert response.json()["message"] == "Invalid image file format."
