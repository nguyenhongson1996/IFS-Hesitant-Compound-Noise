import logging
import os
import sys
from typing import Any

from immutabledict import immutabledict

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_VERBOSITY = int(os.getenv("LOG_VERBOSITY", 1))
LOG_FORMAT = "[%(asctime)s] %(name)s %(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"
# Log format for process-level logs.
THREAD_LOG_FORMAT = ("[%(asctime)s] %(name)s %(process)s %(threadName)s {%(pathname)s:%(lineno)d} %(levelname)s - "
                     "%(message)s")
# Log format for thread-aware logs.
LOG_LEVEL_MAP: immutabledict[str, int] = immutabledict({
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
})


class CustomLogger(logging.Logger):
    def __init__(self, name: str, level: int = logging.NOTSET, verbosity: int = 0):
        super(CustomLogger, self).__init__(name, level=level)
        self.verbosity = verbosity

    def infov(self, msg: str, verbosity: int, *args: Any, **kwargs: Any):
        if self.isEnabledFor(logging.INFO) and self.verbosity >= verbosity:
            self._log(logging.INFO, msg, args, **kwargs)


class CustomRootLogger(CustomLogger):

    def __init__(self, level: int):
        super(CustomRootLogger, self).__init__("root", level)


def config_root_logger() -> logging.Logger:
    log_level = LOG_LEVEL_MAP[LOG_LEVEL]
    root_logger: CustomRootLogger = CustomRootLogger(log_level)
    root_logger.verbosity = LOG_VERBOSITY

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    return root_logger


logger = config_root_logger()
