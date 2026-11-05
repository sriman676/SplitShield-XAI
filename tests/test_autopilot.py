"""Autopilot decisions (scripts/autopilot.py): retrain on feedback, adapt to drift,
nightly assistant improvement, never while training runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.autopilot as ap
from src.soc import engine


@pytest.fixture()
def wired(tmp_path: Path, monkeypatch):  # noqa: ANN001, ANN201
    monkeypatch.setattr(ap, "SOC", tmp_path)
    monkeypatch.setattr(ap, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(ap, "LOG", tmp_path / "log.jsonl")
    monkeypatch.setattr(ap, "training_active", lambda: False)
    # 2026-10-15 regression: the autopilot's newer steps must never touch the real
    # project from a test (they wrote fake decisions to the real controller history,
    # and the nightly code proposals re-ran this test suite recursively)
    from src.soc import codeopt, controller, places, reports, riskboard

    ctl = tmp_path / "controller"
    monkeypatch.setattr(controller, "HOME", ctl)
    monkeypatch.setattr(controller, "HISTORY", ctl / "history.jsonl")
    monkeypatch.setattr(controller, "STATE", ctl / "state.json")
    monkeypatch.setattr(controller, "INBOX", tmp_path / "inbox")
    monkeypatch.setattr(controller, "_snapshot", lambda m: ctl / "snap")
    monkeypatch.setattr(controller, "cycle", lambda: [])
    monkeypatch.setattr(controller, "watch", lambda: {})
    monkeypatch.setattr(controller, "notify", lambda text: None)
    night_work: list[str] = []
    def profile() -> dict:
        night_work.append("profile")
        return {}

    def propose() -> list:
        night_work.append("propose")
        return []

    monkeypatch.setattr(codeopt, "profile", profile)
    monkeypatch.setattr(codeopt, "propose", propose)
    monkeypatch.setattr(places, "check", lambda events, force=False: {})
    monkeypatch.setattr(riskboard, "learn_from_reviews", lambda: {"status": "test"})
    monkeypatch.setattr(reports, "run_due", lambda now=None: [])
    rng = np.random.default_rng(0)
    train = rng.lognormal(2, 1, (3000, 4))
    det = SimpleNamespace(
        feature_names=list("abcd"),
        input_stats={
            "deciles": np.quantile(train, np.linspace(0.1, 0.9, 9), axis=0).T.tolist()
        },
    )
    monkeypatch.setattr(engine, "load_detector", lambda: det)
    calls: list[str] = []

    def retrain(_det: object) -> dict:
        calls.append("retrain")
        return {"kept": True}

    monkeypatch.setattr(engine, "retrain_from_feedback", retrain)
    state = {"reviews": 0}

    class FakeRCL:
        def rows(self):  # noqa: ANN202
            return [{"features": {"a": 1}}] * state["reviews"]

    monkeypatch.setattr("src.soc.rcl.RCL", FakeRCL)

    def flows(shift: float) -> None:
        x = rng.lognormal(2 + shift, 1, (500, 4))
        engine.FLOW_SAMPLE.write_text(
            "".join(json.dumps({"ts": 0, "x": r.tolist()}) + "\n" for r in x)
        )

    return SimpleNamespace(calls=calls, state=state, flows=flows, tmp=tmp_path,
                           night_work=night_work)  # fmt: skip


def test_retrains_after_enough_feedback_only(wired) -> None:  # noqa: ANN001
    noon = time.mktime((2026, 10, 14, 12, 0, 0, 0, 0, -1))
    wired.flows(0.0)
    wired.state["reviews"] = 5
    out = ap.run(now=noon)
    assert out["drift"] == "stable" and wired.calls == []
    wired.state["reviews"] = 25
    ap.run(now=noon)
    assert wired.calls == ["retrain"]
    ap.run(now=noon)  # same reviews -> nothing new to learn
    assert wired.calls == ["retrain"]


def test_drift_asks_for_reviews_then_adapts(wired) -> None:  # noqa: ANN001
    noon = time.mktime((2026, 10, 14, 12, 0, 0, 0, 0, -1))
    wired.flows(2.0)
    out = ap.run(now=noon)
    st = json.loads((wired.tmp / "state.json").read_text())
    assert out["drift"] == "drifted" and "Review recent alerts" in st["notice"]
    assert wired.calls == []
    wired.state["reviews"] = 12  # >= 10 fresh reviews while drifted -> adapt
    ap.run(now=noon)
    assert wired.calls == ["retrain"]


def test_skips_while_training(wired, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(ap, "training_active", lambda: True)
    assert ap.run() == {"skipped": True}


def test_assistant_improves_at_night_once(wired, monkeypatch) -> None:  # noqa: ANN001
    from src.nlp import selfimprove

    runs: list[int] = []
    monkeypatch.setattr(selfimprove, "ready", lambda: {"missing": []})

    def improve() -> dict:
        runs.append(1)
        return {"promoted": False}

    monkeypatch.setattr(selfimprove, "improve", improve)
    wired.flows(0.0)
    night = time.mktime((2026, 10, 14, 2, 0, 0, 0, 0, -1))
    ap.run(now=night)
    ap.run(now=night + 3600)  # same night: not again
    assert runs == [1]
    ap.run(now=night + 86400)
    assert runs == [1, 1]


def test_night_profiling_once_and_never_by_day(wired) -> None:  # noqa: ANN001
    noon = time.mktime((2026, 10, 14, 12, 0, 0, 0, 0, -1))
    night = time.mktime((2026, 10, 14, 3, 0, 0, 0, 0, -1))
    wired.flows(0.0)
    ap.run(now=noon)
    assert wired.night_work == []
    ap.run(now=night)
    ap.run(now=night + 3600)  # same night: not again
    assert wired.night_work == ["profile", "propose"]


def test_proposals_never_trial_inside_tests() -> None:
    import os

    from src.soc import codeopt

    assert (
        os.environ.get("SPLITSHIELD_IN_TRIAL") == "1"
    )  # set for every test (conftest)
    assert codeopt._CHILD_ENV()["SPLITSHIELD_IN_TRIAL"] == "1"


def test_fit_report_never_runs_inside_tests(wired, monkeypatch) -> None:  # noqa: ANN001
    """2026-10-17: a test faking 03:00 started the real 16-minute fit report."""
    import subprocess

    from src.eval import fitcheck

    calls: list[object] = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(fitcheck, "model_report", lambda path=None: {"models": {}, "flagged": [], "age_days": None})
    night = time.mktime((2026, 10, 18, 3, 0, 0, 0, 0, -1))
    ap.run(now=night)
    assert not [c for c in calls if "scripts.fit_report" in str(c)]
