class GoogleDriveFolderEmptyException(Exception):
    """
    Google drive folder empty
    """

    def __init__(self, message="No files found or invalid folder URL."):
        self.message = message
        super().__init__(self.message)


class ModelNotFoundException(Exception):
    """
    Model not found
    """

    def __init__(self, message="Model not found."):
        self.message = message
        super().__init__(self.message)


class GoogleDriveServiceException(Exception):
    """
    Google drive exception
    """

    def __init__(self, message="Google drive exception."):
        self.message = message
        super().__init__(self.message)


class GoogleDriveDownloadException(Exception):
    """
    Google drive download exception
    """

    def __init__(self, message="Google drive download exception."):
        self.message = message
        super().__init__(self.message)


class MLFlowException(Exception):
    """
    MLFlow exception
    """

    def __init__(self, message="Coul not retrieve models from MLFlow."):
        self.message = message
        super().__init__(self.message)
