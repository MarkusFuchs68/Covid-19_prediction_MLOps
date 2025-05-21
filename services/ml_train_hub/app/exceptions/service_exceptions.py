class RegisterModelException(Exception):
    """
    Register model exception
    """

    def __init__(self, message="Failed to register model in MLFlow."):
        self.message = message
        super().__init__(self.message)


class ModelNotFoundException(Exception):
    """
    Model not found
    """

    def __init__(self, message="Model not found in MLFlow."):
        self.message = message
        super().__init__(self.message)


class ModelNotFoundInArtifactsException(Exception):
    """
    Model not found in artifacts
    """

    def __init__(self, message="Model not found in MLFlow artifacts."):
        self.message = message
        super().__init__(self.message)
