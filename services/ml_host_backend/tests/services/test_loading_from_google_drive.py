from unittest.mock import patch

import pytest
from ml_host_backend.app.exceptions.service_exceptions import (
    GoogleDriveFolderEmptyException,
    ModelNotFoundException,
)
from ml_host_backend.app.services.models_service import download_model_from_google_drive


@patch("ml_host_backend.app.services.models_service.gdown.download_folder")
@patch("ml_host_backend.app.services.models_service.gdown.download")
@patch("os.getenv")
def test_load_model_success(mock_getenv, mock_download, mock_download_folder):
    # Mock environment variable
    mock_getenv.return_value = "https://drive.google.com/folder_url"

    # Mock file list returned by gdown.download_folder
    mock_download_folder.return_value = [
        ("file_id_123", "test_model.h5", "./data/models/test_model.h5")
    ]

    # Mock gdown.download to simulate successful download
    mock_download.return_value = "models/test_model.h5"

    # Call the function
    result = download_model_from_google_drive("test_model.h5")

    # Assertions
    assert result == "./data/models/test_model.h5"
    mock_download_folder.assert_called_once()
    mock_download.assert_called_once_with(
        id="file_id_123", output="./data/models/test_model.h5", quiet=False
    )


@patch("ml_host_backend.app.services.models_service.gdown.download_folder")
@patch("os.getenv")
def test_load_model_not_found(mock_getenv, mock_download_folder):
    # Mock environment variable
    mock_getenv.return_value = "https://drive.google.com/folder_url"

    # Mock file list returned by gdown.download_folder
    mock_download_folder.return_value = [
        ("file_id_125", "another_model.h5", "./data/models/another_model.h5")
    ]

    # Call the function and expect an exception
    with pytest.raises(
        ModelNotFoundException,
        match="File 'test_model.h5' not found in the Google Drive folder.",
    ):
        download_model_from_google_drive("test_model.h5")


@patch("ml_host_backend.app.services.models_service.gdown.download_folder")
@patch("os.getenv")
def test_google_drive_folder_empty(mock_getenv, mock_download_folder):
    # Mock environment variable
    mock_getenv.return_value = "https://drive.google.com/folder_url"

    # Mock file list returned by gdown.download_folder
    mock_download_folder.return_value = []

    # Call the function and expect an exception
    with pytest.raises(
        GoogleDriveFolderEmptyException, match="No files found or invalid folder URL."
    ):
        download_model_from_google_drive("test_model.h5")
