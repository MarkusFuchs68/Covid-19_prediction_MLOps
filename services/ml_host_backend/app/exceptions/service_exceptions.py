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


class MLUserMgmtException(Exception):
    """
    ML User Mgmt exception
    """

    def __init__(self, message="Could not retrieve token from the service."):
        self.message = message
        super().__init__(self.message)


class MLUserMgmtUnavailableException(Exception):
    """
    ML User Mgmt unavailable
    """

    def __init__(self, message="ML User Mgmt is unavailable."):
        self.message = message
        super().__init__(self.message)


class MLFlowUnavailableException(Exception):
    """
    MLFlow unavailable
    """

    def __init__(self, message="MLFlow is unavailable."):
        self.message = message
        super().__init__(self.message)


class MLUserMgmtConfigurationException(Exception):
    """
    ML User Mgmt service configuration exception
    """

    def __init__(self, message="ML User Mgmt service is not correctly configured"):
        self.message = message
        super().__init__(self.message)


class MLFlowConfigurationException(Exception):
    """
    MLFlow service configuration exception
    """

    def __init__(self, message="MLFlow service is not correctly configured"):
        self.message = message
        super().__init__(self.message)
