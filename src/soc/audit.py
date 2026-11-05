"""Audit log (owner 2026-10-16, enterprise): who did what, and when.

Every sign-in, review, rule / allow-list / settings change, user or role change and
evolution switch is appended to ``audit.jsonl`` next to the event store. Each entry
carries the SHA-256 of the previous entry: editing or deleting any past line breaks
the chain, and ``verify()`` reports the first broken line. Admins read it on the
dashboard (Settings -> Audit log); it is also sent to the SIEM with the alerts when
"forward audit events" is on.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def path() -> Path:
    from src.soc import engine

    return Path(engine.EVENTS).with_name("audit.jsonl")


def _digest(prev: str, body: dict[str, Any]) -> str:
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev + canon).encode("utf-8")).hexdigest()


def _last_hash(p: Path) -> str:
    """Hash of the last entry (reads only the file's tail)."""
    try:
        with open(p, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 65536))
            tail = fh.read().decode("utf-8", errors="replace").strip().splitlines()
        return str(json.loads(tail[-1])["hash"]) if tail else GENESIS
    except (OSError, ValueError, KeyError, IndexError):
        return GENESIS


def record(
    who: str, action: str, details: dict[str, Any] | None = None, ok: bool = True
) -> dict[str, Any]:
    """Append one entry (cross-process safe). Never raises into the caller's action."""
    from src.soc.engine import _file_lock

    p = path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock(p.with_name(p.name + ".lock")):
            prev = _last_hash(p)
            body = {"ts": round(time.time(), 3), "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "who": who or "unknown", "action": action, "ok": bool(ok),
                    "details": details or {}}  # fmt: skip
            entry = body | {"prev": prev, "hash": _digest(prev, body)}
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        return {}
    _forward(entry)
    return entry


def _forward(entry: dict[str, Any]) -> None:
    """Send the entry to the SIEM too when "forward audit events" is on."""
    try:
        from src.soc import sinks

        s = sinks.load_settings()
        if s.get("forward_audit"):
            sinks.send({"ts": entry["ts"], "kind": "audit", "title": f"audit: {entry['action']}",
                        "severity": 3 if entry["ok"] else 6, "user": entry["who"],
                        "reason": json.dumps(entry["details"], default=str)[:500],
                        "techniques": [], "audit_hash": entry["hash"]}, s)  # fmt: skip
    except Exception:  # noqa: BLE001 - forwarding never blocks the recorded action
        pass


def read(last: int | None = None) -> list[dict[str, Any]]:
    try:
        lines = path().read_text("utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for ln in lines[-last:] if last else lines:
        try:
            rows.append(json.loads(ln))
        except ValueError:
            rows.append({"broken": ln[:200]})
    return rows


def verify() -> dict[str, Any]:
    """Walk the whole chain: ok, entries, and the first line that does not match."""
    prev = GENESIS
    n = 0
    try:
        lines = path().read_text("utf-8").splitlines()
    except OSError:
        return {"ok": True, "entries": 0}
    for n, ln in enumerate(lines, 1):
        try:
            e = json.loads(ln)
            body = {k: v for k, v in e.items() if k not in ("prev", "hash")}
            if e.get("prev") != prev or e.get("hash") != _digest(prev, body):
                return {"ok": False, "entries": len(lines), "broken_at": n}
            prev = e["hash"]
        except (ValueError, KeyError):
            return {"ok": False, "entries": len(lines), "broken_at": n}
    return {"ok": True, "entries": n}
