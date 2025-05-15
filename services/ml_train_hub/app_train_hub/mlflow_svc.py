from datetime import datetime

import mlflow
import mlflow.tensorflow
from mlflow import MlflowClient

# connect to our mlflow server
mlflow_client = MlflowClient(tracking_uri="http://127.0.0.1:8081")


# model experiment tracking
def log_mlflow_experiment(
    hyperparams,
    metrics,
    model,
    experiment_name="Covid_Models",
    register_model=False,
    model_name="Covid-19",
):
    """
    log_mlflow_experiment() is used to log a model and its params, metrics to MLFlow.
    It is possible to register the according model as well.

    parameters:
    - hyperparameters: a dictionary with the hyperparameters used in the model
    - metrics: a dictionary with the metrics of the model
    - model: the model itself
    - experiment_name (default="Covid_Models"): the name of the experiment run
    - register_model (default=False):
        if False, only the experiment run is logged
        if True, the experiment runs model is registered as a new version of that model, if the model is not yet existing, it will be created automatically
    - model_name (default="Covid-19"): name of the MLFlow model in which this experiments model will be registered (applies only if register_model=True)
    """
    mlflow.set_experiment(experiment_name)
    run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(hyperparams)
        mlflow.log_metrics(metrics)
        if register_model:  # log experiment and register its model
            modelinfo = mlflow.tensorflow.log_model(
                model=model, artifact_path=None, registered_model_name=model_name
            )
        else:  # just log the experiment
            modelinfo = mlflow.tensorflow.log_model(
                model=model,
                artifact_path=None,
            )
        return modelinfo


def load_mlflow_model(model_name="Covid-19"):
    return
