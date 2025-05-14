import os
import sys
from random import random

from mlflow.models.model import ModelInfo
from tensorflow.keras.models import Sequential

# we need to tweak the path in order that import finds our app folder
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
app_dir = os.path.join(parent_dir, "app")
sys.path.append(app_dir)
from mlflow_svc import log_mlflow_experiment


def test_log_experiment():
    model = Sequential()  # create an empty model
    params = {"hyperparam": "test_param"}
    metrics = {"performance": random() * 0.29 + 0.7}
    modelinfo = log_mlflow_experiment(params, metrics, model)
    assert isinstance(modelinfo, ModelInfo)


def test_log_experiment_and_register_model():
    model = Sequential()  # create an empty model
    params = {"hyperparam": "test_param"}
    metrics = {"performance": random() * 0.29 + 0.7}
    modelinfo = log_mlflow_experiment(
        params=params,
        metrics=metrics,
        model=model,
        register_model=True,
        model_name="Test",
    )
    assert isinstance(modelinfo, ModelInfo)


def test_dummy():
    print("hello world")
