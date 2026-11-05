"""Evolving or frozen (owner 2026-10-16): frozen components are never updated by the
controller, and their inbox files / ratings wait for later."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.soc import controller, evolution


def test_default_is_evolving() -> None:
    assert evolution.load()["mode"] == "evolving"
    assert evolution.allowed("flow") and evolution.allowed("rules")


def test_frozen_model_is_not_updated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[str] = []
    evolution.save("evolving", ["zeek"], who="tester")
    def update() -> dict:
        called.append("x")
        return {"kept": True}

    out = controller.guarded("zeek", update, "test")
    assert out["decision"].startswith("frozen") and called == []
    evolution.save("frozen")
    assert not evolution.allowed("flow") and not evolution.allowed("rules")
    evolution.save("evolving", [])
    assert evolution.allowed("zeek")


def test_frozen_inbox_files_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    box = controller.INBOX / "zeek"
    box.mkdir(parents=True, exist_ok=True)
    f = box / "new.log"
    f.write_text("x")
    monkeypatch.setattr(controller, "_intent_step", lambda: [])
    evolution.save("evolving", ["zeek"])
    controller.cycle()
    assert f.is_file()  # still waiting, not consumed while frozen


def test_unknown_names_refused() -> None:
    with pytest.raises(ValueError):
        evolution.save("sideways")
    with pytest.raises(ValueError):
        evolution.save("evolving", ["not-a-model"])
