import logging
import os
import sys

from microlog.formatters.json_formatter import JsonFormatter
from microlog.formatters.pretty_formatter import PrettyFormatter
from microlog.utils.exceptions import InvalidEnvException


# TODO consider class implementation
def get_stream_handler() -> logging.Handler:
    log_format = os.getenv("LOG_FORMAT", "pretty").lower()
    handler = logging.StreamHandler(sys.stdout)

    if log_format == "pretty":
        handler.setFormatter(PrettyFormatter())
    elif log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        raise InvalidEnvException(f"Invalid value for LOG_FORMAT: {log_format}")

    return handler
