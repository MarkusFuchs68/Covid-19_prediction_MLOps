from random import random

import pytest
from ml_train_hub.app.mlflow_util import log_mlflow_experiment
from mlflow.models.model import ModelInfo
from tensorflow.keras.models import Sequential


@pytest.mark.integration
def test_log_experiment():
    model = Sequential()  # create an empty model
    architecture = dict(
        {
            "layer0": "Conv2D(32, (3, 3), activation='relu')",
            "layer1": "MaxPooling2D((2, 2))",
        }
    )
    metrics = dict({"performance": random() * 0.29 + 0.7})
    class_names = list(["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"])
    modelinfo = log_mlflow_experiment(model, architecture, metrics, class_names)
    assert isinstance(modelinfo, ModelInfo)


@pytest.mark.integration
def test_log_experiment_and_register_model():
    model = Sequential()  # create an empty model
    architecture = {"architecture": {"layer0": "Conv2D(32, (3, 3), activation='relu')"}}
    class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
    metrics = {"performance": random() * 0.29 + 0.7}
    modelinfo = log_mlflow_experiment(
        model=model,
        architecture=architecture,
        metrics=metrics,
        class_names=class_names,
        register_model=True,
        model_name="Test",
    )
    assert isinstance(modelinfo, ModelInfo)


@pytest.mark.integration
def test_list_models(test_train_hub_client):
    """Test list models endpoint (e.g., GET /models)."""
    response = test_train_hub_client.get("/models")
    assert response.status_code == 200


@pytest.mark.integration
def test_get_model(test_train_hub_client):
    """Test get model endpoint (e.g., GET /models/{model_name})."""
    # Assuming "Test" is a valid model name in your MLFlow
    response = test_train_hub_client.get("/models/Test")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "name" in response.json()
    assert response.json()["name"] == "Test"


@pytest.mark.integration
def test_get_model_not_found(test_train_hub_client):
    """Test get model endpoint with a non-existent model name."""
    response = test_train_hub_client.get("/models/NonExistentModel")
    assert response.status_code == 404
    assert response.json() == {
        "message": "No versions found for model 'NonExistentModel'"
    }


# For debugging
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    from ml_train_hub.app.main import app

    test_get_model(TestClient(app))


"""
import requests
import pandas as pd
import json

# Prepare the data
data = pd.read_csv("data/fake_data.csv")
X = data.drop(columns=["date", "demand"])
X = X.astype('float')

# Convert data to JSON format
json_data = {
    "dataframe_split": {
        "columns": X.columns.tolist(),
        "data": X.head(2).values.tolist()  # Testing with 2 rows
    }
}

# Send request to the API
response = requests.post(
    url="http://localhost:5002/invocations",
    json=json_data,
    headers={"Content-Type": "application/json"}
)

# Display predictions
if response.status_code == 200:
    predictions = response.json()
    print("\nReceived predictions:")
    print(predictions)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
"""
