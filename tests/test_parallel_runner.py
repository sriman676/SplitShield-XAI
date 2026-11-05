"""ai_engineer --parallel: chains run in order, stop on a failure, longest first."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import ai_engineer as ae


@pytest.fixture()
def prog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ae.Progress:
    monkeypatch.setattr(ae, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(ae, "LOG", tmp_path / "log.txt")
    return ae.Progress(fresh=True, tag="full")


def _cmd(code: int) -> list[str]:
    return [sys.executable, "-c", f"raise SystemExit({code})"]


def test_chain_stops_after_a_failure(prog: ae.Progress) -> None:
    ae.failures.clear()
    ae.run_chains(
        prog,
        [[("a", _cmd(0)), ("b", _cmd(1)), ("c", _cmd(0))], [("d", _cmd(0))]],
        workers=2,
        settle=0,
    )
    assert prog.done == {"a", "d"} and ae.failures == ["b"]
    ae.failures.clear()


def test_done_phases_are_skipped(prog: ae.Progress) -> None:
    prog.mark("a")
    ae.run_chains(prog, [[("a", _cmd(1))]], workers=1, settle=0)
    assert ae.failures == []


def test_past_minutes_and_thread_split(prog: ae.Progress) -> None:
    ae.LOG.write_text(
        "2026-10-14 03:22:37 OK    train:R2:seed0 in 87.9 min\n"
        "2026-10-14 03:22:38 FAIL  x (exit 1) after 2.0 min\n",
        "utf-8",
    )
    assert ae.past_minutes() == {"train:R2:seed0": 87.9}
    env = ae.thread_env(3)
    assert int(env["OMP_NUM_THREADS"]) >= 1 and env["LOKY_MAX_CPU_COUNT"]
