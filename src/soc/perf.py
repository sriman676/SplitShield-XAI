"""Performance settings: how much memory and CPU the SOC may use (owner 2026-10-16).

Three modes (Settings -> Performance on the dashboard):

  * ``auto`` (default)   up to half of the free memory and all but one CPU core,
                         re-planned as the PC gets busier or quieter;
  * ``memory``           a memory budget in GB (e.g. 4 or 8): more memory -> more
                         parallel log-analysis workers and a bigger search cache;
  * ``speed``            a target in logs per second: the smallest plan that reaches
                         it (or the fastest possible, with a note when it cannot).

What a plan sets: the number of log-analysis worker processes (src/soc/parallel.py,
each one parses a slice of a big file - the main cost of ingestion), the search-index
cache and memory map (src/soc/eventindex.py), and the predicted logs/s. Predictions
come from ``calibrate()``, which measures this PC once (about a minute): logs/s of
one worker and the memory one worker needs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

MODES = ("auto", "memory", "speed")
BASE_MB = 700  # the dashboard / monitor process itself (models, Sigma rules, caches)
MIN_CACHE_MB = 64  # search index: below this, searches at 1 M+ events slow down
MAX_CACHE_MB = 2048
# before calibration: conservative figures measured on a 2026 laptop CPU
DEFAULT_CAL: dict[str, Any] = {
    "worker_lps": 9000.0,
    "worker_mb": 450.0,
    "measured": None,
}


def _soc_dir() -> Path:
    from src.soc import engine

    return Path(engine.EVENTS).parent


def settings_path() -> Path:
    return _soc_dir() / "performance.json"


def calibration_path() -> Path:
    return _soc_dir() / "performance_calibration.json"


def load_settings() -> dict[str, Any]:
    try:
        doc = json.loads(settings_path().read_text("utf-8"))
    except (OSError, ValueError):
        doc = {}
    mode = doc.get("mode", "auto")
    return {
        "mode": mode if mode in MODES else "auto",
        "memory_gb": float(doc.get("memory_gb", 4.0)),
        "target_lps": float(doc.get("target_lps", 20000.0)),
        # lean search index (owner 2026-10-24): full-text rows for the last N days, 0 = all
        "fulltext_days": int(doc.get("fulltext_days", 0)),
    }


def save_settings(
    mode: str, memory_gb: float | None = None, target_lps: float | None = None,
    fulltext_days: int | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unknown performance mode {mode!r}")
    doc = load_settings() | {"mode": mode}
    if memory_gb is not None:
        if not 0.5 <= memory_gb <= total_ram_gb():
            raise ValueError(
                f"memory budget must be between 0.5 GB and {total_ram_gb():.0f} GB"
            )
        doc["memory_gb"] = float(memory_gb)
    if target_lps is not None:
        if target_lps <= 0:
            raise ValueError("target speed must be above 0 logs per second")
        doc["target_lps"] = float(target_lps)
    if fulltext_days is not None:
        if not 0 <= int(fulltext_days) <= 3650:
            raise ValueError("full-text days must be 0 (all) to 3650")
        doc["fulltext_days"] = int(fulltext_days)
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1), "utf-8")
    _apply_index()
    return doc


def calibration() -> dict[str, Any]:
    try:
        return DEFAULT_CAL | json.loads(calibration_path().read_text("utf-8"))
    except (OSError, ValueError):
        return dict(DEFAULT_CAL)


# --- the machine ------------------------------------------------------------------


def _mem() -> tuple[float, float]:
    """(total, available) physical memory in GB on Windows, macOS and Linux."""
    from utils.sysinfo import ram_gb

    return ram_gb()


def total_ram_gb() -> float:
    return _mem()[0]


def free_ram_gb() -> float:
    return _mem()[1]


def cores() -> int:
    return os.cpu_count() or 2


# --- the plan --------------------------------------------------------------------------


def _useful_workers(cal: dict[str, Any]) -> int:
    """More workers stop helping once storing the events is the slower part: the
    smallest measured count within 5 % of the best measured speed."""
    curve = {int(k): float(v) for k, v in (cal.get("curve") or {}).items()}
    if not curve:
        return max(1, cores() - 1)
    best = max(curve.values())
    return min(w for w, v in curve.items() if v >= 0.95 * best)


def _plan_for_memory(mb: float, cal: dict[str, Any]) -> dict[str, Any]:
    """Workers and index cache that fit in ``mb`` megabytes."""
    max_workers = min(max(1, cores() - 1), _useful_workers(cal))
    room = max(0.0, mb - BASE_MB - MIN_CACHE_MB)
    workers = int(max(1, min(max_workers, room // float(cal["worker_mb"]))))
    if room < float(cal["worker_mb"]):
        workers = 1  # parse in the main process only
    left = mb - BASE_MB - (workers * float(cal["worker_mb"]) if workers > 1 else 0)
    cache = int(max(MIN_CACHE_MB, min(MAX_CACHE_MB, left)))
    return {"workers": workers, "index_cache_mb": cache, "index_mmap_mb": int(min(4 * cache, 8192)),
            "memory_mb": int(BASE_MB + (workers * float(cal["worker_mb"]) if workers > 1 else 0) + cache),
            "predicted_lps": int(_lps(workers, cal))}  # fmt: skip


def _lps(workers: int, cal: dict[str, Any]) -> float:
    """Predicted end-to-end logs/s (parse + store + alerts). From the calibration's
    measured points: interpolated between them, never extrapolated above the best
    measured speed. Before calibration: a cautious 85 % per extra worker."""
    curve = sorted((int(k), float(v)) for k, v in (cal.get("curve") or {}).items())
    if not curve:
        one = float(cal["worker_lps"])
        return one if workers <= 1 else one * (1 + 0.85 * (workers - 1))
    if workers <= curve[0][0]:
        return curve[0][1]
    for (w0, v0), (w1, v1) in zip(curve, curve[1:], strict=False):
        if workers <= w1:
            return v0 + (v1 - v0) * (workers - w0) / (w1 - w0)
    return curve[-1][1]


def plan(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """The plan in force for the current settings and this PC."""
    s = settings or load_settings()
    cal = calibration()
    if s["mode"] == "memory":
        p = _plan_for_memory(s["memory_gb"] * 1024, cal)
        p["note"] = f"memory budget {s['memory_gb']:g} GB"
    elif s["mode"] == "speed":
        target = s["target_lps"]
        max_workers = min(max(1, cores() - 1), _useful_workers(cal))
        # the fewest workers predicted to reach the target (else the fastest plan)
        workers = next(
            (w for w in range(1, max_workers + 1) if _lps(w, cal) >= target),
            max_workers,
        )
        mb = BASE_MB + (workers * float(cal["worker_mb"]) if workers > 1 else 0) + 256
        p = _plan_for_memory(mb, cal)
        p["workers"] = workers
        p["predicted_lps"] = int(_lps(workers, cal))
        reach = p["predicted_lps"] >= target
        p["note"] = (f"target {target:,.0f} logs/s reached" if reach else
                     f"target {target:,.0f} logs/s is above this PC's maximum "
                     f"(about {p['predicted_lps']:,} logs/s with {workers} workers)")  # fmt: skip
        p["target_reached"] = reach
    else:
        p = _plan_for_memory(max(1024.0, free_ram_gb() * 1024 * 0.5), cal)
        p["note"] = "automatic: up to half of the free memory"
    p["mode"] = s["mode"]
    p["calibrated"] = cal.get("measured")
    return p


def _apply_index() -> None:
    """Reopen the search index so its cache follows the new plan."""
    from src.soc import eventindex

    eventindex.close()


# --- calibration ------------------------------------------------------------------------


def _worker_probe(lines: list[str]) -> dict[str, float]:
    """Run in a fresh process: parse ``lines``; logs/s and peak memory of a worker."""
    from src.soc.parsers import parse_line

    for ln in lines[:300]:  # Sigma rules compile once per process
        parse_line(ln)
    t = time.perf_counter()
    for ln in lines:
        parse_line(ln)
    lps = len(lines) / max(time.perf_counter() - t, 1e-9)
    return {"lps": lps, "peak_mb": _peak_mb()}


def _peak_mb() -> float:
    from src.soc.winmem import peak_mb

    v = peak_mb()
    return float(DEFAULT_CAL["worker_mb"]) if v is None else v


def _measure_curve(base: list[str], n_lines: int = 120_000) -> dict[str, float]:
    """End-to-end logs/s (parse in w workers + store + alerts, in a temporary store)
    for a few worker counts. Pool start-up is not timed (it happens once per process)."""
    import shutil
    import tempfile

    from src.soc import engine, eventindex, parallel

    lines = (base * (n_lines // max(1, len(base)) + 1))[:n_lines]
    saved = {n: getattr(engine, n) for n in ("EVENTS", "ALERTS", "PUSHED", "STATUS")}
    tmp = Path(tempfile.mkdtemp(prefix="perfcal_"))
    curve: dict[str, float] = {}
    try:
        for n in saved:
            setattr(engine, n, tmp / saved[n].name)
        # this process loads the rules / models once (not part of any speed)
        engine.record(parallel.parse_lines(lines[:2000], workers=1), send=False)
        for w in sorted({1, 2, 4, max(1, cores() - 1)}):
            for f in tmp.glob("*"):
                f.unlink()
            if w > 1:
                parallel.parse_lines(
                    lines[: parallel.SLICE * w], workers=w
                )  # warm the pool
            t = time.perf_counter()
            evs = parallel.parse_lines(lines, workers=w)
            engine.record(evs, send=False)
            curve[str(w)] = round(len(lines) / (time.perf_counter() - t), 1)
    finally:
        parallel.shutdown()
        eventindex.close()
        for n, v in saved.items():
            setattr(engine, n, v)
        shutil.rmtree(tmp, ignore_errors=True)
    return curve


def calibrate(n_lines: int = 40_000) -> dict[str, Any]:
    """Measure one log-analysis worker on this PC (fresh process, real parsers and
    Sigma rules, the SOC's fixed benchmark lines). About a minute."""
    from concurrent.futures import ProcessPoolExecutor

    from src.soc.codeopt import _bench_lines

    base = _bench_lines()
    lines = (base * (n_lines // max(1, len(base)) + 1))[:n_lines]
    import multiprocessing as mp

    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        r = pool.submit(_worker_probe, lines).result()
    cal = {"worker_lps": round(r["lps"], 1), "worker_mb": round(r["peak_mb"] + 50, 1),
           "measured": time.strftime("%Y-%m-%d %H:%M"), "cores": cores(),
           "ram_gb": round(total_ram_gb(), 1),
           "curve": _measure_curve(base)}  # fmt: skip
    p = calibration_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cal, indent=1), "utf-8")
    return cal
