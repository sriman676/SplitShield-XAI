"""Log rotation must not raise or drop records when another process holds the file."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from utils import logging_config
from utils.logging_config import SafeRotatingFileHandler


def _record(msg: str) -> logging.LogRecord:
    return logging.LogRecord("t", logging.INFO, __file__, 1, msg, None, None)


def test_blocked_rotation_keeps_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "x.log"
    h = SafeRotatingFileHandler(log, maxBytes=10, backupCount=2)
    calls = []

    def locked(src: str, dst: str) -> None:
        calls.append(src)
        raise PermissionError(32, "in use by another process")

    monkeypatch.setattr(h, "rotate", locked)
    errors = []
    monkeypatch.setattr(h, "handleError", lambda r: errors.append(r))
    for i in range(5):
        h.emit(_record(f"line {i} long enough to exceed the limit"))
    h.close()
    assert not errors
    assert log.read_text().count("line") == 5
    assert len(calls) == 1  # retried only after ROTATE_RETRY_S, not per record


def test_rotation_still_works_when_free(tmp_path: Path) -> None:
    log = tmp_path / "x.log"
    h = SafeRotatingFileHandler(log, maxBytes=10, backupCount=2)
    for i in range(3):
        h.emit(_record(f"line {i} long enough to exceed the limit"))
    h.close()
    assert (tmp_path / "x.log.1").is_file()
    assert logging_config.ROTATE_RETRY_S > 0
