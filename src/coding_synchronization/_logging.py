import logging
import sys
from pathlib import Path

_NAME = "coding_synchronization"
_FMT = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
_DATE = "%Y-%m-%dT%H:%M:%S"


def setup_logging(level: int | str = logging.DEBUG) -> None:
    log = logging.getLogger(_NAME)
    log.setLevel(level)
    log.propagate = False
    log.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter(_FMT, _DATE))
    log.addHandler(h)


def add_file_handler(log_file: Path) -> None:
    h = logging.FileHandler(log_file, encoding="utf-8")
    h.setFormatter(logging.Formatter(_FMT, _DATE))
    logging.getLogger(_NAME).addHandler(h)
