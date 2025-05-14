from datetime import datetime

import mlflow
import mlflow.tensorflow
from mlflow import MlflowClient

# connect to our mlflow server
mlflow_client = MlflowClient(tracking_uri="http://127.0.0.1:8081")

# set our experiment project name
experiment_name = "Covid_Models"
covid_experiment = mlflow.set_experiment(experiment_name)

# other variable definitions
artifact_path = "covid_cnn"  # the path to the models in mlflow


# model experiment tracking
def log_mlflow_experiment(
    params, metrics, model, register_model=False, model_name="Covid-19"
):

    run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if register_model:  # log experiment and register its model
            modelinfo = mlflow.tensorflow.log_model(
                model=model,
                artifact_path=artifact_path,
                registered_model_name=model_name,
            )
        else:  # just log the experiment
            modelinfo = mlflow.tensorflow.log_model(
                model=model, artifact_path=artifact_path
            )
        return modelinfo


def load_mlflow_model(model_name="Covid-19"):
    return
