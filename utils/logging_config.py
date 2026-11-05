"""Logging configured once, imported everywhere (08_ENGINEERING_STANDARDS §2).

Console + rotating file under ``logs/`` (gitignored). Never use ``print`` except
for final CLI summaries.
"""

from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler

from utils.paths import LOGS_DIR

_CONFIGURED = False
ROTATE_RETRY_S = 600


class SafeRotatingFileHandler(RotatingFileHandler):
    """Rotating file log that tolerates other processes holding the file.

    On Windows a file open in another process (the orchestrator and each pipeline
    run log to the same file) cannot be renamed, so a plain RotatingFileHandler
    raises PermissionError on every record once the size limit is hit and drops
    that record. Here a blocked rotation keeps appending to the current file and
    retries only after ``ROTATE_RETRY_S`` seconds.
    """

    _blocked_until = 0.0

    def shouldRollover(self, record: logging.LogRecord) -> int:  # noqa: N802
        if time.monotonic() < self._blocked_until:
            return False
        return super().shouldRollover(record)

    def doRollover(self) -> None:  # noqa: N802
        try:
            super().doRollover()
        except PermissionError:
            self._blocked_until = time.monotonic() + ROTATE_RETRY_S
            if self.stream is None:
                self.stream = self._open()


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging idempotently.

    Args:
        level: Logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    fileh = SafeRotatingFileHandler(
        LOGS_DIR / "splitshield.log", maxBytes=5_000_000, backupCount=3
    )
    fileh.setFormatter(fmt)
    root.addHandler(fileh)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
