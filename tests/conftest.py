"""Shared test isolation: no test writes into the real SOC store.

2026-10-15 regression: tests that exercised the autopilot wrote fake decisions into
the real controller history and started real code-proposal trials (which re-ran the
test suite recursively). Every writable state file of the SOC is therefore sent to a
per-test temporary folder here; read-only inputs (models, config) stay real.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_soc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every writable SOC state path -> this test's temporary folder. The list lives in
    src/soc/sandbox.py, shared with the every-combination stress test."""
    from src.soc import allowlist, auth, eventindex, sandbox

    for mod, attr, value in sandbox.targets(tmp_path):
        monkeypatch.setattr(mod, attr, value)
    monkeypatch.setitem(allowlist._CACHE, "key", None)
    monkeypatch.delenv(auth.KEY, raising=False)
    # never start real code-proposal trials (they run the whole test suite)
    monkeypatch.setenv("SPLITSHIELD_IN_TRIAL", "1")
    eventindex.close()
    yield
    eventindex.close()  # open SQLite files would keep the temp folder locked


def engine_mod():  # noqa: ANN201 - the SOC engine module
    from src.soc import engine

    return engine
