import os
import time

import jwt
import pytest
from ml_train_hub.app.mlflow_util import log_mlflow_experiment
from mlflow.models.model import ModelInfo

JWT_SECRET = "secret"  # this should be specified in a vault in a real application or otherwise mocked
JWT_ALGORITHM = "HS256"
active_token = jwt.encode(
    {"user_id": "user123", "expires": time.time() + 600},
    JWT_SECRET,
    algorithm=JWT_ALGORITHM,
)


@pytest.mark.integration
def test_log_experiment():
    model_filepath = os.path.join(
        os.path.dirname(__file__), "4_50ep_medparam_4xconv2d_dense128.keras"
    )
    class_names = list(["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"])
    modelinfo = log_mlflow_experiment(
        model_filepath=model_filepath,
        class_names=class_names,
        experiment_name="integration_testing",
    )
    assert isinstance(modelinfo, ModelInfo)


@pytest.mark.integration
def test_log_experiment_and_register_model():
    model_filepath = os.path.join(
        os.path.dirname(__file__), "4_50ep_medparam_4xconv2d_dense128.keras"
    )
    class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
    modelinfo = log_mlflow_experiment(
        model_filepath=model_filepath,
        class_names=class_names,
        experiment_name="integration_testing",
        register_model=True,
        model_name="Test",
    )
    assert isinstance(modelinfo, ModelInfo)


@pytest.mark.integration
def test_list_models(test_train_hub_client):
    """Test list models endpoint (e.g., GET /models)."""
    response = test_train_hub_client.get(
        "/models", headers={"Authorization": f"Bearer {active_token}"}
    )
    assert response.status_code == 200


@pytest.mark.integration
def test_get_model(test_train_hub_client):
    """Test get model endpoint (e.g., GET /models/{model_name})."""
    # Assuming "Test" is a valid model name in your MLFlow
    response = test_train_hub_client.get(
        "/models/Test", headers={"Authorization": f"Bearer {active_token}"}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "name" in response.json()
    assert response.json()["name"] == "Test"


@pytest.mark.integration
def test_get_model_not_found(test_train_hub_client):
    """Test get model endpoint with a non-existent model name."""
    response = test_train_hub_client.get(
        "/models/NonExistentModel", headers={"Authorization": f"Bearer {active_token}"}
    )
    assert response.status_code == 404
    assert response.json() == {
        "message": "No versions found for model 'NonExistentModel'"
    }


# For debugging
if __name__ == "__main__":
    # from fastapi.testclient import TestClient
    # from ml_train_hub.app.main import app

    # test_get_model(TestClient(app))

    test_log_experiment()
