from unittest.mock import MagicMock, patch

import pytest
from ml_host_backend.app.exceptions.service_exceptions import (
    MLFlowException,
    ModelNotFoundException,
)
from ml_host_backend.app.services.mlflow_service import (
    get_single_model_summary_from_mlflow,
    list_all_models_from_mlflow,
)


@patch(
    "ml_host_backend.app.services.mlflow_service.get_mlflow_host_and_port",
    return_value=("localhost", "5000"),
)
@patch("ml_host_backend.app.services.mlflow_service.requests.get")
def test_list_all_models_success(mock_requests_get, mock_get_host_port):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "models": [{"name": "model1"}, {"name": "model2"}]
    }
    mock_response.raise_for_status.return_value = None
    mock_requests_get.return_value = mock_response
    result = list_all_models_from_mlflow()
    assert result == [{"name": "model1"}, {"name": "model2"}]
    mock_requests_get.assert_called_once_with(
        "http://localhost:5000/models", timeout=10
    )


@patch(
    "ml_host_backend.app.services.mlflow_service.get_mlflow_host_and_port",
    return_value=("localhost", "5000"),
)
@patch("ml_host_backend.app.services.mlflow_service.requests.get")
def test_list_all_models_no_models_key(mock_requests_get, mock_get_host_port):
    mock_response = MagicMock()
    mock_response.json.return_value = [{"name": "model1"}]
    mock_response.raise_for_status.return_value = None
    mock_requests_get.return_value = mock_response
    result = list_all_models_from_mlflow()
    assert result == [{"name": "model1"}]


@patch(
    "ml_host_backend.app.services.mlflow_service.get_mlflow_host_and_port",
    return_value=("localhost", "5000"),
)
@patch("ml_host_backend.app.services.mlflow_service.requests.get")
def test_list_all_models_raises_mlflow_exception(mock_requests_get, mock_get_host_port):
    mock_requests_get.side_effect = Exception("Connection error")
    with pytest.raises(MLFlowException):
        list_all_models_from_mlflow()


@patch(
    "ml_host_backend.app.services.mlflow_service.get_mlflow_host_and_port",
    return_value=("localhost", "5000"),
)
@patch("ml_host_backend.app.services.mlflow_service.requests.get")
def test_get_single_model_summary_success(mock_requests_get, mock_get_host_port):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"name": "model1", "version": "1"}
    mock_response.raise_for_status.return_value = None
    mock_requests_get.return_value = mock_response
    result = get_single_model_summary_from_mlflow("model1")
    assert result == {"name": "model1", "version": "1"}
    mock_requests_get.assert_called_once_with(
        "http://localhost:5000/models/model1", timeout=10
    )


@patch(
    "ml_host_backend.app.services.mlflow_service.get_mlflow_host_and_port",
    return_value=("localhost", "5000"),
)
@patch("ml_host_backend.app.services.mlflow_service.requests.get")
def test_get_single_model_summary_not_found(mock_requests_get, mock_get_host_port):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.json.return_value = None
    mock_response.raise_for_status.side_effect = Exception("404 Not Found")
    mock_requests_get.return_value = mock_response
    with pytest.raises(ModelNotFoundException):
        get_single_model_summary_from_mlflow("unknown_model")


@patch(
    "ml_host_backend.app.services.mlflow_service.get_mlflow_host_and_port",
    return_value=("localhost", "5000"),
)
@patch("ml_host_backend.app.services.mlflow_service.requests.get")
def test_get_single_model_summary_other_exception(
    mock_requests_get, mock_get_host_port
):
    mock_requests_get.side_effect = Exception("Timeout")
    with pytest.raises(MLFlowException):
        get_single_model_summary_from_mlflow("model1")
