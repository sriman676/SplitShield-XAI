"""Parallel follow-up phases (ai_engineer.run_parallel): at most N at once, the
second waits for free RAM, and progress written by parallel phases is never lost."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import ai_engineer as ae


def test_parallel_runs_and_merges_progress(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(ae, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(ae, "free_ram_gb", lambda: 99.0)
    prog = ae.Progress(fresh=True, tag="full")
    now, peak = [0], [0]
    lock = threading.Lock()

    def fake_phase(p, key, cmd, **kw):  # noqa: ANN001, ANN003, ANN202
        with lock:
            now[0] += 1
            peak[0] = max(peak[0], now[0])
        time.sleep(0.2)
        with lock:
            now[0] -= 1
        p.mark(key)
        return True

    monkeypatch.setattr(ae, "phase", fake_phase)
    # another writer recorded a phase meanwhile: it must survive our marks
    (tmp_path / "state.json").write_text(json.dumps({"full": ["nlp"]}))
    ae.run_parallel(prog, [(f"k{i}", []) for i in range(4)], workers=2)
    done = set(json.loads((tmp_path / "state.json").read_text())["full"])
    assert done == {"nlp", "k0", "k1", "k2", "k3"}
    assert peak[0] == 2


def test_second_slot_waits_for_ram(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(ae, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(ae, "free_ram_gb", lambda: 1.0)  # never enough for two
    monkeypatch.setattr(ae.time, "sleep", lambda s: None if s >= 30 else None)
    prog = ae.Progress(fresh=True, tag="full")
    now, peak = [0], [0]
    lock = threading.Lock()

    def fake_phase(p, key, cmd, **kw):  # noqa: ANN001, ANN003, ANN202
        with lock:
            now[0] += 1
            peak[0] = max(peak[0], now[0])
        with lock:
            now[0] -= 1
        p.mark(key)
        return True

    monkeypatch.setattr(ae, "phase", fake_phase)
    ae.run_parallel(prog, [(f"k{i}", []) for i in range(3)], workers=2)
    assert peak[0] == 1 and prog.done == {"k0", "k1", "k2"}
