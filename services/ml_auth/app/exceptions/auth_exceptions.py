class InvalidArgumentException(Exception):
    """
    Invalid argument exception
    """

    def __init__(self, message="Params are not correct or empty."):
        self.message = message
        super().__init__(self.message)


class FailedAuthentification(Exception):
    """
    Wrong credentials or unknown user
    """

    def __init__(self, message="Wrong credentials or unknown user."):
        self.message = message
        super().__init__(self.message)
