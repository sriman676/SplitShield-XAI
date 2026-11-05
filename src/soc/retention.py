"""Local data retention (owner 2026-10-16, enterprise): keep the last N days of events
and alerts in the live store; older lines are moved to a compressed archive
(default) or deleted (the owner's choice in Settings -> Places & backup).

* Default: 30 days, archive - nothing is lost; ``archive/events-<from>-<to>.jsonl.gz``
  can be searched again by uploading it on Analyze logs.
* The store is rewritten IN PLACE under the same cross-process lock every writer uses
  (engine._append), so no line arriving meanwhile is lost or torn. A marker file
  makes an interrupted run finish on the next start before anything else.
* The search index (src/soc/eventindex.py) sees the shrunk store and rebuilds itself.
* The audit log is never trimmed (its hash chain must stay whole); archives are kept
  forever unless "delete archives older than" is set.
* Runs once a day from the autopilot (src/soc/autopilot) and from "Apply now".
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

DEFAULT: dict[str, Any] = {"enabled": True, "days": 30, "mode": "archive", "archive_days": 0}
MODES = {"archive": "Move to a compressed archive (recommended)", "delete": "Delete for good"}


def settings() -> dict[str, Any]:
    from src.soc import sinks

    return DEFAULT | dict(sinks.load_settings().get("retention") or {})


def check(s: dict[str, Any]) -> dict[str, Any]:
    out = DEFAULT | {k: s[k] for k in DEFAULT if k in s}
    out["days"] = int(out["days"])
    out["archive_days"] = int(out["archive_days"])
    if not 1 <= out["days"] <= 3650:
        raise ValueError("keep 1-3650 days")
    if out["mode"] not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if out["archive_days"] and out["archive_days"] < out["days"]:
        raise ValueError("archives must be kept at least as long as the live store")
    out["enabled"] = bool(out["enabled"])
    return out


def save(s: dict[str, Any], who: str = "") -> dict[str, Any]:
    from src.soc import audit, sinks

    s = check(s)
    full = sinks.load_settings()
    full["retention"] = s
    sinks.save_settings(full)
    audit.record(who or "system", "retention_settings", s)
    return s


def archive_dir() -> Path:
    from src.soc import engine

    return Path(engine.EVENTS).parent / "archive"


def _marker(path: Path) -> Path:
    return path.with_name(path.name + ".retention.json")


def _ts(line: bytes) -> float | None:
    try:
        return float(json.loads(line).get("ts") or 0)
    except (ValueError, AttributeError, TypeError):
        return None  # damaged line: kept (never silently dropped)


def _copy_back(path: Path, tmp: Path) -> None:
    """Overwrite ``path`` with ``tmp`` in place (works while readers have it open)."""
    with open(tmp, "rb") as src, open(path, "r+b") as dst:
        dst.truncate(0)
        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())


def recover() -> None:
    """Finish a rewrite that was interrupted after its kept-lines file was complete."""
    from src.soc import engine

    for path in (Path(engine.EVENTS), Path(engine.ALERTS)):
        m = _marker(path)
        if not m.is_file():
            continue
        info = json.loads(m.read_text("utf-8"))
        tmp = Path(info["tmp"])
        with engine._file_lock(path.with_name(path.name + ".lock")):
            if info.get("phase") == "copy" and tmp.is_file():
                _copy_back(path, tmp)
            tmp.unlink(missing_ok=True)
            m.unlink(missing_ok=True)


def trim(path: Path, cutoff: float, mode: str) -> dict[str, Any]:
    """Keep lines with ts >= cutoff; the rest go to the archive (or are dropped)."""
    from src.soc import engine

    if not path.is_file() or path.stat().st_size == 0:
        return {"kept": 0, "moved": 0}
    tmp = path.with_name(path.name + ".retention.tmp")
    arch = None
    kept = moved = 0
    lo: float | None = None
    hi: float | None = None
    with engine._file_lock(path.with_name(path.name + ".lock")):
        if mode == "archive":
            archive_dir().mkdir(parents=True, exist_ok=True)
            arch_tmp = archive_dir() / f".{path.stem}-{int(time.time())}.jsonl.gz.part"
            arch = gzip.open(arch_tmp, "wb", compresslevel=6)
        try:
            with open(path, "rb") as src, open(tmp, "wb") as keep:
                for line in src:
                    if not line.endswith(b"\n"):  # half-written last line: keep as is
                        keep.write(line)
                        continue
                    t = _ts(line)
                    if t is None or t >= cutoff:
                        keep.write(line)
                        kept += 1
                        continue
                    moved += 1
                    lo = t if lo is None or t < lo else lo
                    hi = t if hi is None or t > hi else hi
                    if arch is not None:
                        arch.write(line)
        finally:
            if arch is not None:
                arch.close()
        if moved == 0:
            tmp.unlink(missing_ok=True)
            if arch is not None:
                arch_tmp.unlink(missing_ok=True)
            return {"kept": kept, "moved": 0}
        if arch is not None:  # the archive is complete before the store shrinks
            day = lambda t: time.strftime("%Y%m%d", time.localtime(t))  # noqa: E731
            final = archive_dir() / f"{path.stem}-{day(lo)}-{day(hi)}-{int(time.time())}.jsonl.gz"
            arch_tmp.replace(final)
        m = _marker(path)
        m.write_text(json.dumps({"tmp": str(tmp), "phase": "copy"}), "utf-8")
        _copy_back(path, tmp)
        tmp.unlink(missing_ok=True)
        m.unlink(missing_ok=True)
    return {"kept": kept, "moved": moved, "oldest": lo, "newest_moved": hi}


def _old_archives(days: int) -> list[Path]:
    if not days or not archive_dir().is_dir():
        return []
    limit = time.time() - days * 86400
    return [p for p in archive_dir().glob("*.jsonl.gz") if p.stat().st_mtime < limit]


def apply(now: float | None = None, who: str = "") -> dict[str, Any]:
    """One retention pass (events + alerts). Returns what was kept / moved."""
    from src.soc import audit, engine, eventindex

    s = settings()
    if not s["enabled"]:
        return {"skipped": "retention is switched off"}
    recover()
    cutoff = (time.time() if now is None else now) - s["days"] * 86400
    eventindex.close()
    out: dict[str, Any] = {name: trim(Path(p), cutoff, s["mode"])
           for name, p in (("events", engine.EVENTS), ("alerts", engine.ALERTS))}  # fmt: skip
    engine._jsonl_cache.clear()
    removed = []
    for p in _old_archives(s["archive_days"]):
        p.unlink()
        removed.append(p.name)
    out["archives_removed"] = removed
    out["cutoff"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(cutoff))
    if any(v.get("moved") for v in out.values() if isinstance(v, dict)) or removed:
        audit.record(who or "system", "retention_applied", {
            "mode": s["mode"], "days": s["days"], "cutoff": out["cutoff"],
            "events_moved": out["events"]["moved"], "alerts_moved": out["alerts"]["moved"],
            "archives_removed": len(removed)})  # fmt: skip
    return out


def sizes() -> dict[str, float]:
    """MB of the live store, the index and the archive (for the dashboard)."""
    from src.soc import engine, eventindex

    def mb(p: Path) -> float:
        return round(p.stat().st_size / 2**20, 1) if p.is_file() else 0.0

    arch = sum(mb(p) for p in archive_dir().glob("*.jsonl.gz")) if archive_dir().is_dir() else 0.0
    return {"store": mb(Path(engine.EVENTS)) + mb(Path(engine.ALERTS)),
            "index": mb(eventindex.db_path()), "archive": round(arch, 1)}  # fmt: skip
