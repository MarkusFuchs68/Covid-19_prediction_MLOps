class InvalidArgumentException(Exception):
    """
    Invalid argument exception
    """

    def __init__(self, message="Params are not correct or empty."):
        self.message = message
        super().__init__(self.message)


class UnauthroizedException(Exception):
    """
    Unauthorized exception
    """

    def __init__(self, message="Unauthorized."):
        self.message = message
        super().__init__(self.message)
