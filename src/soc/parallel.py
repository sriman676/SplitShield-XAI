"""Parallel log analysis (owner 2026-10-16): big files are parsed by several worker
processes at once, as many as the performance plan allows (src/soc/perf.py).

Only line-by-line formats are split (syslog, web servers, Windows / Sysmon JSON
lines, IDS, firewall, ...): every line is parsed exactly as before, the slices are
put back in their original order, so the events are identical to one process
(tested). Formats whose lines need the file's header or the whole file (Zeek,
W3C, XML, CloudTrail) are parsed in one process as before. The pool stays up, so
the rules are loaded once per worker, not per file; small files skip it.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

MIN_LINES = 20_000  # below this, one process is faster than handing out work
SLICE = 10_000  # lines per task

_lock = threading.Lock()
_pool: ProcessPoolExecutor | None = None
_size = 0
last: dict[str, Any] = {}


def _warm() -> None:
    from src.soc.parsers import parse_line

    parse_line(
        "Oct 16 10:00:00 host sshd[1]: Failed password for root from 1.2.3.4 port 22 ssh2"
    )


def _parse_slice(lines: list[str]) -> list[dict[str, Any]]:
    from src.soc.parsers import parse_line

    return [ev for ln in lines if (ev := parse_line(ln))]


def _skip_dashboard_main() -> None:
    """Spawned workers re-run the parent's main *script* first. Inside the dashboard
    (Streamlit, or the dashboard tests) that script is dashboard.py, so every worker
    tried to start the dashboard and died - parsing a big upload failed with
    BrokenProcessPool (found 2026-10-17 by the research-run test gate). Workers only
    need this module (imported by name), so the dashboard script is not re-run."""
    import multiprocessing.spawn as spawn

    if getattr(spawn.get_preparation_data, "_splitshield", False):
        return
    original = spawn.get_preparation_data

    def prep(name: str) -> dict[str, Any]:
        data: dict[str, Any] = original(name)
        if Path(str(data.get("init_main_from_path", ""))).name == "dashboard.py":
            data.pop("init_main_from_path")
        return data

    prep._splitshield = True  # type: ignore[attr-defined]
    spawn.get_preparation_data = prep


def _get_pool(workers: int) -> ProcessPoolExecutor:
    _skip_dashboard_main()
    global _pool, _size
    with _lock:
        if _pool is None or _size != workers:
            if _pool is not None:
                _pool.shutdown(wait=False, cancel_futures=True)
            _pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                        initializer=_warm)  # fmt: skip
            _size = workers
        return _pool


def shutdown() -> None:
    global _pool, _size
    with _lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
        _pool, _size = None, 0


def parse_lines(lines: list[str], workers: int | None = None) -> list[dict[str, Any]]:
    """``[parse_line(l) for l in lines if it parses]`` - in parallel when the file is
    big and the plan allows more than one worker (``workers`` overrides the plan)."""
    from src.soc import perf

    t = time.perf_counter()
    if workers is None:
        workers = perf.plan()["workers"] if len(lines) >= MIN_LINES else 1
    if workers <= 1:
        out = _parse_slice(lines)
    else:
        pool = _get_pool(workers)
        parts = [lines[i : i + SLICE] for i in range(0, len(lines), SLICE)]
        out = [ev for chunk in pool.map(_parse_slice, parts) for ev in chunk]
    _note(len(lines), time.perf_counter() - t, workers)
    return out


def _record_slice(args: tuple[list[str], bool, str | None, list[tuple[str, str, Any]]]) -> tuple[int, int]:
    """Worker side of parse_and_record: parse a slice, run the whole per-event pipeline
    (cross-checks, rule changes, own rules, OWASP labels) and append it to the shared
    store under the store's cross-process lock. Only two counts go back."""
    import importlib

    lines, send, observer, state = args
    for mod, attr, value in state:  # the caller's store location (sandboxes, tests)
        setattr(importlib.import_module(mod), attr, value)
    evs = _parse_slice(lines)
    if observer:
        for e in evs:
            e.setdefault("observer", observer)
    from src.soc import engine

    return len(evs), len(engine.record(evs, send=send))


def store_state() -> list[tuple[str, str, Any]]:
    """(module, attribute, value) of every writable SOC path as this process sees it."""
    from src.soc import sandbox

    return [(mod.__name__, attr, getattr(mod, attr)) for mod, attr, _ in sandbox.targets(Path("."))]


def parse_and_record(lines: list[str], workers: int, send: bool = True,
                     observer: str | None = None) -> dict[str, int]:  # fmt: skip
    """The sharded store path (owner 2026-10-24): every worker parses AND stores its
    share, so more workers keep adding speed (shipping parsed events back to one
    storing process was the limit: 2 workers ~61k logs/s, 4 workers slower)."""
    t = time.perf_counter()
    pool = _get_pool(workers)
    state = store_state()
    parts = [(lines[i : i + SLICE], send, observer, state) for i in range(0, len(lines), SLICE)]
    n_ev = n_al = 0
    for a, b in pool.map(_record_slice, parts):
        n_ev += a
        n_al += b
    _note(len(lines), time.perf_counter() - t, workers)
    return {"rows": n_ev, "alerts": n_al, "workers": workers}


def _note(n: int, secs: float, workers: int) -> None:
    """Remember the speed of the last big parse (shown on the Performance page)."""
    if n < 1000:
        return
    last.update({"lines": n, "seconds": round(secs, 2), "workers": workers,
                 "lps": int(n / max(secs, 1e-9)), "when": time.strftime("%Y-%m-%d %H:%M:%S")})  # fmt: skip
    try:
        from src.soc import perf

        p: Path = perf._soc_dir() / "performance_last.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(last), "utf-8")
    except OSError:
        pass
