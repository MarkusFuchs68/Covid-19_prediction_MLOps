class InvalidEnvException(Exception):
    """Exception to handle invalid env variable values."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)
