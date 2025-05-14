import os
import pytest
from unittest.mock import patch, MagicMock
from app.models.service import load_model_from_google_drive
from app.exceptions.service_exceptions import GoogleDriveFolderEmptyException, ModelNotFoundException

@patch("app.models.service.gdown.download_folder")
@patch("app.models.service.gdown.download")
@patch("os.getenv")
def test_load_model_success(mock_getenv, mock_download, mock_download_folder):
    # Mock environment variable
    mock_getenv.return_value = "https://drive.google.com/folder_url"

    # Mock file list returned by gdown.download_folder
    mock_download_folder.return_value = [
        {"name": "test_model.h5", "id": "file_id_123"}
    ]

    # Mock gdown.download to simulate successful download
    mock_download.return_value = "models/test_model.h5"

    # Call the function
    result = load_model_from_google_drive("test_model.h5")

    # Assertions
    assert result == "./models/test_model.h5"
    mock_download_folder.assert_called_once()
    mock_download.assert_called_once_with("file_id_123", "./models/test_model.h5", quiet=False)

@patch("app.models.service.gdown.download_folder")
@patch("os.getenv")
def test_load_model_not_found(mock_getenv, mock_download_folder):
    # Mock environment variable
    mock_getenv.return_value = "https://drive.google.com/folder_url"

    # Mock file list returned by gdown.download_folder
    mock_download_folder.return_value = [
        {"name": "another_model.h5", "id": "file_id_456"}
    ]

    # Call the function and expect an exception
    with pytest.raises(ModelNotFoundException, match="File 'test_model.h5' not found in the Google Drive folder."):
        load_model_from_google_drive("test_model.h5")

@patch("app.models.service.gdown.download_folder")
@patch("os.getenv")
def test_google_drive_folder_empty(mock_getenv, mock_download_folder):
    # Mock environment variable
    mock_getenv.return_value = "https://drive.google.com/folder_url"

    # Mock file list returned by gdown.download_folder
    mock_download_folder.return_value = []

    # Call the function and expect an exception
    with pytest.raises(GoogleDriveFolderEmptyException, match="No files found or invalid folder URL."):
        load_model_from_google_drive("test_model.h5")