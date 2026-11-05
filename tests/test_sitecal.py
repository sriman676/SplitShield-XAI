"""Per-site calibration of the flow detector (owner 2026-10-24)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from src.soc import sitecal


def test_threshold_keeps_alerts_at_target_and_never_looser() -> None:
    rng = np.random.default_rng(0)
    scores = rng.beta(2, 5, 100_000)  # a site whose normal traffic scores high-ish
    thr = sitecal.threshold_for(scores, validated=0.5)
    assert abs(float((scores >= thr).mean()) - sitecal.TARGET) < 0.002
    assert sitecal.threshold_for(scores * 0.1, validated=0.93) == 0.93  # never below validated


def test_reservoir_and_proposal_flow(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    p = tmp_path / "scores.json"
    sitecal._MEM.clear()
    sitecal.observe(np.linspace(0, 1, 1000), path=p, rng=random.Random(0), flush=True)
    st = sitecal.status(p)
    assert st["seen"] == 1000 and st["kept"] == 1000
    monkeypatch.setattr(sitecal, "MIN_FLOWS", 100)
    monkeypatch.setattr(sitecal, "MIN_DAYS", 0.0)
    monkeypatch.setattr("src.soc.custom.validated", lambda model: 0.5)
    prop = sitecal.propose(p)
    assert prop is not None and prop["rule"] == "sens:flow" and prop["change"]["value"] > 0.98
