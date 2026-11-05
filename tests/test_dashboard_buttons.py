"""Stress: every page, every button (owner 2026-10-16). Each page is opened and every
button on it is pressed once, against the temporary SOC store of tests/conftest.py.
Buttons that act outside the dashboard (firewall blocking, starting the monitor
process, sending to a SIEM, applying code, retraining for minutes, downloads) are
skipped and listed; everything else must run without an exception."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.test_dashboard import app

st_testing = pytest.importorskip("streamlit.testing.v1")
DASH = Path(__file__).resolve().parent.parent / "dashboard.py"
PATHS = re.findall(r'\(\s*"[^"]+",\s*"[^"]+",\s*page_\w+,\s*"[^"]+",\s*"([^"]+)",?\s*\)',
                   DASH.read_text("utf-8"))  # fmt: skip
# label words of buttons with effects outside the app or long training runs
SKIP = re.compile(r"block|start monitoring|stop|approve|apply|revert|reject|test connection|"
                  r"send|retrain|fine-tune|train|updat|check now|download|sign in|sign out|calibrate|save password|"
                  r"learn from the inboxes|check health|profile now|improve assistant|roll back|look for speed-ups", re.I)  # fmt: skip


@pytest.mark.parametrize("path", PATHS)
def test_every_button_on_page(
    path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # every writable SOC file is already redirected by tests/conftest.py
    at = app(path, timeout=180)
    at.run()
    assert not at.exception, f"{path}: {at.exception}"
    labels = [b.label for b in at.button]
    pressed = []
    for i, label in enumerate(labels):
        if SKIP.search(label or ""):
            continue
        at = app(path, timeout=180)
        at.run()
        if i >= len(at.button):
            continue
        at.button[i].click().run()
        assert not at.exception, f"{path} / {label!r}: {at.exception}"
        pressed.append(label)
    print(
        f"{path}: pressed {pressed}; skipped {[x for x in labels if x not in pressed]}"
    )
