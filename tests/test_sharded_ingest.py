"""Sharded store path (owner 2026-10-24): workers parse AND store; same events as one process."""

from __future__ import annotations

import json
from collections import Counter

from src.soc import engine, parallel
from src.soc.parsers import parse_line


def _lines(n: int) -> list[str]:
    out = []
    for i in range(n):
        ip = f"203.0.113.{i % 50}"
        if i % 7 == 0:
            out.append(f"Oct 16 10:{i // 600 % 60:02d}:{i % 60:02d} web01 sshd[{i}]: Failed password for root from {ip} port 22 ssh2")
        else:
            out.append(f"Oct 16 10:{i // 600 % 60:02d}:{i % 60:02d} web01 CRON[{i}]: pam_unix(cron:session): session opened for user root")
    return out


def _stored() -> Counter[str]:
    return Counter(json.dumps({k: e[k] for k in ("ts", "kind", "src_ip", "techniques", "observer") if k in e},
                              sort_keys=True) for e in engine.read_jsonl(engine.EVENTS))  # fmt: skip


def test_workers_store_the_same_events_as_one_process() -> None:
    lines = _lines(parallel.MIN_LINES + 5_000)
    evs = [e for ln in lines if (e := parse_line(ln))]  # one process, the reference
    for e in evs:
        e.setdefault("observer", "web01")
    one = {"rows": len(evs), "alerts": len(engine.record(evs, send=False))}
    single = _stored()
    engine.EVENTS.unlink()
    engine.ALERTS.unlink()
    res = parallel.parse_and_record(lines, workers=2, send=False, observer="web01")
    parallel.shutdown()
    assert res["rows"] == one["rows"] and res["alerts"] == one["alerts"]
    assert _stored() == single
