import logging
import os

from microlog.handlers.stream_handler import get_stream_handler


def init_logger(level: str = "INFO") -> logging.Logger:
    level = os.getenv("LOG_LEVEL", level).upper()
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers by marking ours
    existing = {type(h) for h in logger.handlers if getattr(h, "_microlog", False)}

    if logging.StreamHandler not in existing:
        handler = get_stream_handler()
        handler._microlog = True
        logger.addHandler(handler)

    return logger
