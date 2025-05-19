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
