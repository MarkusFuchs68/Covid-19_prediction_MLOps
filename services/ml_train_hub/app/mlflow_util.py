import logging
import os
from datetime import datetime

# import ml_train_hub.app.exceptions.client_exceptions as ce
import ml_train_hub.app.exceptions.service_exceptions as se
import mlflow
import mlflow.tensorflow

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# set our docker container running the local MLFlow service,
# otherwise mlflow will default to file://mlruns, which we don't want to
stage = os.getenv("RUNNING_STAGE")
# on which stage are we running
if stage == "prod":
    logger.info("Running MLFlow server on prod port 8081")
    mlflow.set_tracking_uri("http://localhost:8081")  # prod server on port 8081
else:
    logger.info("Running MLFlow server on dev/test port 8001")
    mlflow.set_tracking_uri("http://localhost:8001")  # dev server on port 8001


# model experiment tracking
def log_mlflow_experiment(
    model,
    architecture,
    metrics,
    class_names,
    experiment_name="Covid_Models",
    register_model=False,
    model_name="Covid-19",
):
    """
    log_mlflow_experiment() is used to log a model and its params, metrics to MLFlow.
    It is possible to register the according model as well.

    parameters:
    - model: the model itself
    - architecture: a dictionary with the architecture of the model
    - metrics: a dictionary with the metrics of the model
    - class_names: list of human readable class names associated with the prediction index
    - experiment_name (default="Covid_Models"): the name of the experiment run
    - register_model (default=False):
        if False, only the experiment run is logged
        if True, the experiment runs model is registered as a new version of that model, if the model is not yet existing, it will be created automatically
    - model_name (default="Covid-19"): name of the MLFlow model in which this experiments model will be registered (applies only if register_model=True)
    """
    # optionally restore a deleted experiments
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment and experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment.experiment_id)
            logger.info(f"Restored experiment '{experiment_name}'")
    except Exception as e:
        logger.error(f"Error restoring experiment '{experiment_name}': {e}")
        raise se.RegisterModelException(
            f"Failed to restore experiment '{experiment_name}': {e}"
        ) from e

    # Now register the new experiment
    logger.info(f"Logging experiment '{experiment_name}' to MLFlow")

    try:
        mlflow.set_experiment(experiment_name)
        run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("architecture", architecture)
            mlflow.log_param("class_names", class_names)  # convert list to json format
            mlflow.log_metrics(metrics)
            if register_model:  # log experiment and register its model
                modelinfo = mlflow.tensorflow.log_model(
                    model=model, artifact_path="model", registered_model_name=model_name
                )
                logger.info(
                    f"Logged experiment with run {run_name} and model {model_name}"
                )
            else:  # just log the experiment
                modelinfo = mlflow.tensorflow.log_model(
                    model=model, artifact_path="model"
                )
                logger.info(f"Logged experiment with run {run_name}")
            return modelinfo
    except Exception as e:
        logger.error(
            f"Error logging experiment '{experiment_name}' with model '{model_name}': {e}"
        )
        raise se.RegisterModelException(
            f"Failed to register model '{model_name}' in MLFlow: {e}"
        ) from e


def get_model_path(client, run):
    """
    Make the relative path to the model file stored in the MLFlow storage

    Args:
        client: MLflow client, connected to the MLFlow server
        run_id (str): ID of the run to inspect
    """
    artifacts = client.list_artifacts(run.info.run_id)
    for artifact in artifacts:
        if artifact.is_dir:
            nested_artifacts = client.list_artifacts(run.info.run_id, artifact.path)
            for nested in nested_artifacts:
                if nested.path.endswith(".keras"):
                    model_path = os.path.join(run.info.artifact_uri, nested.path)
                    logger.info(f"Found model in artifacts: {model_path}")
                    return model_path

    raise se.ModelNotFoundInArtifactsException(
        f"No model file found in artifacts for run ID: {run.info.run_id}"
    )


def get_model_params(run):
    """
    Get all parameters in a run.

    Args:
        client: MLflow client
        run_id: ID of the run to inspect
    """
    params = run.data.params
    return params


def get_model_metrics(run):
    """
    Get all metrics in a run.
    Args:
        client: MLflow client
        run_id: ID of the run to inspect
    """
    metrics = run.data.metrics
    return metrics


def get_mlflow_model(model_name):
    """
    Get a specific model from MLFlow.

    Args:
        model_name (str): Name of the model to retrieve.

    Returns a dictionary of model key/value pairs
    """

    logger.info(f"Retrieving model information for '{model_name}' from MLFlow")

    client = mlflow.tracking.MlflowClient()

    # Query the model versions
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise se.ModelNotFoundException(f"No versions found for model '{model_name}'")

    # Get the latest version of the model
    latest_version = versions[0]

    # and its run data/info
    run = client.get_run(latest_version.run_id)

    # Extract the relative path to the model in the mlruns storage
    model_path = get_model_path(client, run)

    # Get the parameters
    params = get_model_params(run)

    # Get the metrics
    metrics = get_model_metrics(run)

    # Put together the provided data
    model_data = {
        "name": model_name,
        "version": latest_version.version,
        "model_filepath": model_path,  # this is an absolute path matching our docker container file structure
        "status": latest_version.status,
        "architecture": params.get("architecture", None),
        "class_names": params.get("class_names", None),
        "metrics": metrics,
        "created_time": latest_version.creation_timestamp,
        "last_updated_time": latest_version.last_updated_timestamp,
        "run_name": run.info.run_name,
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
    }

    logger.info(f"Retrieved model data for '{model_name}': {model_data}")

    return model_data


def list_mlflow_models():
    """
    Return a list of all models in MLFlow and their latest version
    """

    logger.info("Listing all models in MLFlow")

    # Return a list of all available models in MLFlow
    models = mlflow.search_registered_models()
    model_list = []
    for model in models:

        logger.info(f"Found model: {model.name}")

        # Get the latest version of the model
        latest_version = model.latest_versions[0]

        # Query the desired data for this model, continue if not found or failed
        try:
            model_list.append(get_mlflow_model(latest_version.name))
        except Exception as e:
            logger.error(f"Error retrieving model {latest_version.name}: {e}")

    return model_list


# For debugging
if __name__ == "__main__":
    # get_mlflow_model("Test")
    from random import random

    from tensorflow.keras.models import Sequential

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
