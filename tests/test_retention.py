"""Local retention: old events archived (or deleted) in place, nothing new lost, an
interrupted run finishes, the index rebuilds, settings are checked and audited."""

from __future__ import annotations

import gzip
import json
import threading
import time
from pathlib import Path

import pytest

from src.soc import audit, engine, eventindex, retention

NOW = time.time()


def _fill(days_back: list[int]) -> None:
    rows = [{"ts": NOW - d * 86400, "source": "auth", "kind": "failed_logon", "detail": f"d{d}",
             "techniques": [], "src_ip": "203.0.113.9"} for d in days_back]  # fmt: skip
    engine._append(Path(engine.EVENTS), rows)


def test_archive_keeps_new_moves_old() -> None:
    _fill([60, 45, 31, 29, 1, 0])
    assert eventindex.count() == 6
    res = retention.apply(now=NOW)
    assert res["events"] == {"kept": 3, "moved": 3, "oldest": pytest.approx(NOW - 60 * 86400),
                             "newest_moved": pytest.approx(NOW - 31 * 86400)}  # fmt: skip
    left = [json.loads(ln)["detail"] for ln in Path(engine.EVENTS).read_text("utf-8").splitlines()]
    assert left == ["d29", "d1", "d0"]
    (arch,) = retention.archive_dir().glob("events-*.jsonl.gz")
    moved = [json.loads(ln)["detail"] for ln in gzip.open(arch, "rt", encoding="utf-8")]
    assert moved == ["d60", "d45", "d31"]
    assert eventindex.count() == 3  # the index followed the shrunk store
    assert any(r["action"] == "retention_applied" for r in audit.read())
    assert retention.apply(now=NOW)["events"]["moved"] == 0  # nothing twice


def test_delete_mode_and_settings_checks() -> None:
    retention.save({"days": 7, "mode": "delete"}, who="boss")
    _fill([10, 8, 2])
    assert retention.apply(now=NOW)["events"]["moved"] == 2
    assert not list(retention.archive_dir().glob("*.gz")) if retention.archive_dir().is_dir() else True
    for bad in ({"days": 0}, {"mode": "shred"}, {"days": 30, "archive_days": 10}):
        with pytest.raises(ValueError):
            retention.check(bad)
    retention.save({"enabled": False})
    assert "skipped" in retention.apply(now=NOW)


def test_lines_written_during_a_run_are_never_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    _fill([40] * 500 + [1] * 500)
    real = retention._ts
    stop = threading.Event()

    def writer() -> None:  # another process keeps appending meanwhile
        k = 0
        while not stop.is_set() and k < 200:
            engine._append(Path(engine.EVENTS), [{"ts": NOW, "detail": f"live{k}", "techniques": []}])
            k += 1

    th = threading.Thread(target=writer)
    def slow_ts(line: bytes) -> float | None:
        time.sleep(0.0005)
        return real(line)

    monkeypatch.setattr(retention, "_ts", slow_ts)
    th.start()
    retention.apply(now=NOW)
    stop.set()
    th.join()
    details = [json.loads(ln)["detail"] for ln in Path(engine.EVENTS).read_text("utf-8").splitlines()]
    live = [d for d in details if d.startswith("live")]
    assert len(live) == len(set(live)) and all(d != "d40" for d in details)
    assert sum(d == "d1" for d in details) == 500
    assert [int(d[4:]) for d in live] == sorted(int(d[4:]) for d in live)  # nothing torn or lost


def test_interrupted_copy_is_finished_on_next_start(tmp_path: Path) -> None:
    store = Path(engine.EVENTS)
    _fill([1, 0])
    keep = store.with_name(store.name + ".retention.tmp")
    keep.write_text(store.read_text("utf-8").splitlines(keepends=True)[1], "utf-8")
    retention._marker(store).write_text(json.dumps({"tmp": str(keep), "phase": "copy"}), "utf-8")
    store.write_text("", "utf-8")  # crashed half-way through the copy
    retention.recover()
    assert [json.loads(ln)["detail"] for ln in store.read_text("utf-8").splitlines()] == ["d0"]
    assert not keep.exists() and not retention._marker(store).exists()
