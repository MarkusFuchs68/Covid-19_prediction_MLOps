import logging


class PrettyFormatter(logging.Formatter):
    """Shared standard format with tast of pretty."""

    def __init__(
        self,
        fmt="[%(levelname)s] [%(name)s] %(message)s",
        # fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ) -> None:
        super().__init__(fmt, datefmt)
