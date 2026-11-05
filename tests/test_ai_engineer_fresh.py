"""--fresh must redo every phase even when an old progress file exists (2026-10-17:
the first finished phase merged the copied project's saved progress back in, and the
whole research run was skipped in 10 minutes)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ai_engineer


def test_fresh_ignores_and_sets_aside_old_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = tmp_path / "ai_engineer_state.json"
    state.write_text(json.dumps({"full": ["gate", "phase:E1", "phase:E2"]}), "utf-8")
    monkeypatch.setattr(ai_engineer, "STATE", state)
    prog = ai_engineer.Progress(fresh=True, tag="full")
    prog.mark("gate")
    assert prog.done == {"gate"}  # nothing old came back
    assert json.loads(state.read_text("utf-8"))["full"] == ["gate"]
    assert list(tmp_path.glob("ai_engineer_state.before-fresh-*.json"))  # kept, not deleted


def test_resume_keeps_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = tmp_path / "ai_engineer_state.json"
    state.write_text(json.dumps({"full": ["gate", "phase:E1"]}), "utf-8")
    monkeypatch.setattr(ai_engineer, "STATE", state)
    prog = ai_engineer.Progress(fresh=False, tag="full")
    assert prog.done == {"gate", "phase:E1"}
