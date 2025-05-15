from random import random

from ml_train_hub.app.mlflow_svc import log_mlflow_experiment
from mlflow.models.model import ModelInfo
from tensorflow.keras.models import Sequential


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
