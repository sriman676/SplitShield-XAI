"""Parallel log analysis (src/soc/parallel.py) gives exactly the events of one process,
in the same order; the performance plan follows the chosen mode (src/soc/perf.py)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from src.soc import parallel, parsers, perf


def _lines(n: int) -> list[str]:
    rng = random.Random(3)
    out = []
    for i in range(n):
        k = rng.randrange(4)
        if k == 0:
            out.append(f"Oct 16 10:{i % 60:02d}:00 web1 sshd[{i}]: Failed password for root from "
                       f"10.0.{i % 200}.{i % 250} port {1000 + i} ssh2")  # fmt: skip
        elif k == 1:
            out.append(f'10.1.{i % 200}.7 - - [16/Oct/2026:10:00:{i % 60:02d} +0000] "GET /admin/{i}.php '
                       f'HTTP/1.1" {rng.choice([200, 404, 403])} 512 "-" "sqlmap/1.7"')  # fmt: skip
        elif k == 2:
            out.append(json.dumps({"EventID": 4625, "TimeCreated": "2026-10-16T10:00:00Z",
                                   "Message": f"An account failed to log on. Account Name: u{i % 30}",
                                   "IpAddress": f"192.168.1.{i % 250}"}))  # fmt: skip
        else:
            out.append(json.dumps({"timestamp": "2026-10-16T10:00:00Z", "event_type": "alert",
                                   "src_ip": f"172.16.0.{i % 250}", "dest_ip": "10.0.0.1", "dest_port": 445,
                                   "alert": {"signature": "ET SCAN Nmap", "category": "Attempted Recon"}}))  # fmt: skip
    return out


def test_parallel_equals_one_process(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = _lines(parallel.MIN_LINES + 7_000)
    text = "\n".join(lines)
    monkeypatch.setattr(perf, "plan", lambda s=None: {"workers": 1})
    one = parsers.parse_text(text, "mixed.log")
    monkeypatch.setattr(perf, "plan", lambda s=None: {"workers": 3})
    try:
        many = parsers.parse_text(text, "mixed.log")
        assert parallel.last["workers"] == 3
    finally:
        parallel.shutdown()
    assert len(one) > 20_000 and many == one


def test_plans_follow_the_mode() -> None:
    cal = perf.calibration()
    small = perf.plan({"mode": "memory", "memory_gb": 1.0, "target_lps": 0})
    big = perf.plan({"mode": "memory", "memory_gb": 8.0, "target_lps": 0})
    assert small["workers"] <= big["workers"] and small["memory_mb"] <= big["memory_mb"]
    assert big["predicted_lps"] >= small["predicted_lps"]
    reach = perf.plan(
        {"mode": "speed", "memory_gb": 4, "target_lps": cal["worker_lps"] * 0.5}
    )
    assert reach["workers"] == 1 and reach["target_reached"]
    far = perf.plan({"mode": "speed", "memory_gb": 4, "target_lps": 1e9})
    assert not far["target_reached"] and "maximum" in far["note"]


def test_settings_validated() -> None:
    assert perf.load_settings()["mode"] == "auto"
    perf.save_settings("memory", memory_gb=2)
    assert (
        perf.load_settings() | {} == perf.load_settings()
        and perf.plan()["mode"] == "memory"
    )
    with pytest.raises(ValueError):
        perf.save_settings("turbo")
    with pytest.raises(ValueError):
        perf.save_settings("speed", target_lps=-5)


def test_workers_never_rerun_the_dashboard_script() -> None:
    """2026-10-17: inside the dashboard every spawned worker re-ran dashboard.py and
    died (BrokenProcessPool on big uploads)."""
    import multiprocessing.spawn as spawn

    parallel._skip_dashboard_main()
    import sys

    main = sys.modules["__main__"]
    had = getattr(main, "__file__", None)
    try:
        main.__file__ = str(Path("dashboard.py").resolve())
        assert "init_main_from_path" not in spawn.get_preparation_data("w")
        main.__file__ = str(Path("other_script.py").resolve())
        data = spawn.get_preparation_data("w")
        assert data.get("init_main_from_path", "").endswith("other_script.py") or "init_main_from_name" in data
    finally:
        if had is None:
            main.__dict__.pop("__file__", None)
        else:
            main.__file__ = had
