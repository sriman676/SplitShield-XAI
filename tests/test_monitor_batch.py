"""The live monitor scores flows in batches (2026-10-17): same alerts as one flow at
a time, every flow counted, the last partial batch not lost."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from src.soc import engine

FIX = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "fixture.csv"


def test_replay_in_batches_matches_one_by_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    det = engine.load_detector()
    if det is None:
        pytest.skip("no deployed detector")
    from src.live.capture import replay_flows

    small = tmp_path / "replay.csv"
    fix = pd.read_csv(FIX)
    pd.concat([fix, fix]).to_csv(small, index=False)
    flows = [f for _m, f in replay_flows(small)]
    n_rows = len(flows)
    assert n_rows > engine.FLOW_BATCH  # one full batch + a partial one
    one_by_one = sum(len(engine.detect_flows(det, pd.DataFrame([f]), "x")) for f in flows)
    calls: list[int] = []
    real = engine.detect_flows

    def spy(d, frame, source, rcl=None):  # noqa: ANN001, ANN202
        calls.append(len(frame))
        return real(d, frame, source, rcl)

    monkeypatch.setattr(engine, "detect_flows", spy)
    monkeypatch.setattr(engine, "push_incidents", lambda *a, **k: 0)
    engine.monitor(argparse.Namespace(replay=str(small), live=False, iface=None, rate=0.0,
                                      block=False, keep_running=False))  # fmt: skip
    assert sum(calls) == n_rows and calls[0] == engine.FLOW_BATCH
    alerts = [a for a in engine.read_jsonl(engine.ALERTS) if a.get("source") == "detector"]
    assert len(alerts) == one_by_one
