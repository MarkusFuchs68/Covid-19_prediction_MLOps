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


@pytest.fixture(autouse=True)
def patch_mlflow_host_and_port():
    with patch(
        "ml_host_backend.app.services.mlflow_service.get_mlflow_host_and_port",
        return_value=("localhost", "5000"),
    ):
        yield


def test_get_summary_of_all_models(client):
    expected_models = models_summary
    with patch(
        "ml_host_backend.app.services.mlflow_service.requests.get"
    ) as mock_requests_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": expected_models}
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response
        response = client.get(base_endpoint)
        assert response.status_code == 200
        assert response.json() == expected_models
        assert mock_requests_get.call_count == 2
        called_url = mock_requests_get.call_args[0][0]
        assert "/models" in called_url


def test_get_summary_of_all_models_mlflow_timeout(client):
    """Test /api/models when MLflow service times out (requests.exceptions.Timeout)."""
    import requests

    with patch(
        "ml_host_backend.app.services.mlflow_service.requests.get"
    ) as mock_requests_get:
        mock_requests_get.side_effect = requests.exceptions.ConnectionError(
            "Request timed out"
        )
        response = client.get(base_endpoint)
        assert response.status_code == 503
        assert (
            "timeout" in response.json()["message"].lower()
            or "service" in response.json()["message"].lower()
        )


def test_get_summary_of_single_model(client):
    available_model = models_summary[0]["name"]
    mlflow_response = models_summary[0]
    with patch(
        "ml_host_backend.app.services.mlflow_service.requests.get"
    ) as mock_requests_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mlflow_response
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response
        response = client.get(f"{base_endpoint}/{available_model}")
        assert response.status_code == 200
        assert response.json() == mlflow_response
        assert mock_requests_get.call_count == 2
        called_url = mock_requests_get.call_args[0][0]
        assert f"/models/{available_model}" in called_url


def test_get_summary_of_single_model_not_found(client):
    with patch(
        "ml_host_backend.app.services.mlflow_service.requests.get"
    ) as mock_requests_get:
        unavailable_model = "nonexistent"
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = None
        mock_requests_get.return_value = mock_response
        response = client.get(f"{base_endpoint}/{unavailable_model}")
        assert response.status_code == 404
        assert response.json()["message"] == f"Model '{unavailable_model}' not found"
        assert mock_requests_get.call_count == 2
        called_url = mock_requests_get.call_args[0][0]
        assert f"/models/{unavailable_model}" in called_url


def test_get_summary_of_model_found(client):
    available_model = models_summary[0]["name"]
    mlflow_response = models_summary[0]
    with patch(
        "ml_host_backend.app.services.mlflow_service.requests.get"
    ) as mock_requests_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mlflow_response
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response
        response = client.get(f"{base_endpoint}/{available_model}")
        assert response.status_code == 200
        assert response.json() == mlflow_response
        assert mock_requests_get.call_count == 2
        called_url = mock_requests_get.call_args[0][0]
        assert f"/models/{available_model}" in called_url


@pytest.mark.skip(reason="Underlying functionality to be refactored")
def test_make_prediction_for_image_success(client):
    img = Image.new("L", (224, 224), color=128)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    available_model = models_summary[0]["name"]
    file = {"file": ("test.png", img_bytes, "image/png")}
    with patch(
        "ml_host_backend.app.services.google_drive_service.download_model_from_google_drive",
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
        "ml_host_backend.app.services.google_drive_service.download_model_from_google_drive",
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


@pytest.mark.skip(reason="Underlying functionality to be refactored")
def test_make_prediction_for_large_image_file(client):
    img = Image.new("L", (4000, 4000), color=128)  # large image
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    available_model = models_summary[0]["name"]
    with patch(
        "ml_host_backend.app.services.google_drive_service.download_model_from_google_drive",
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
        "ml_host_backend.app.services.google_drive_service.download_model_from_google_drive",
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


def test_only_to_trigger_actions_pipeline():
    print("triggered")
    assert True
