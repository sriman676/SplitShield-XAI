"""Incident case management (owner 2026-10-22 gap audit): every incident can be
worked as a case - status, assignee, notes - next to its detections.

    status: new -> investigating -> contained -> resolved | false_positive

Stored in ``cases.json`` next to the event store; every change is in the audit log.
Changing a case never changes a model: marking an incident "false positive" here is a
case decision; teaching the detector stays the separate Real / False alarm review.
Shown on Incidents & review (permission ``review`` to change), in reports and through
the read-only API (``case`` field of each incident).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

STATUSES: dict[str, str] = {
    "new": "New",
    "investigating": "Investigating",
    "contained": "Contained",
    "resolved": "Resolved",
    "false_positive": "False positive",
}
OPEN = ("new", "investigating", "contained")
MAX_NOTE = 4000
_LOCK = threading.Lock()


def path() -> Path:
    from src.soc import engine

    return Path(engine.EVENTS).with_name("cases.json")


def _load() -> dict[str, dict[str, Any]]:
    try:
        return dict(json.loads(path().read_text("utf-8")))
    except (OSError, ValueError):
        return {}


def _save(d: dict[str, dict[str, Any]]) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1), "utf-8")
    tmp.replace(p)


def get(incident_id: str) -> dict[str, Any]:
    c = _load().get(incident_id)
    return c or {"status": "new", "assignee": "", "notes": [], "updated": None}


def all_cases() -> dict[str, dict[str, Any]]:
    return _load()


def _change(by: str, incident_id: str, action: str, details: dict[str, Any], fn: Any) -> dict[str, Any]:
    from src.soc import audit, users

    if not users.can(by, "review"):
        raise PermissionError("changing a case needs the 'review' permission")
    if not incident_id or len(incident_id) > 200:
        raise ValueError("unknown incident")
    with _LOCK:
        d = _load()
        case = d.get(incident_id) or {"status": "new", "assignee": "", "notes": [], "updated": None,
                                      "opened": time.strftime("%Y-%m-%d %H:%M")}  # fmt: skip
        fn(case)
        case["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        d[incident_id] = case
        _save(d)
    audit.record(by, action, {"incident": incident_id, **details})
    return case


def set_status(by: str, incident_id: str, status: str) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")

    def fn(c: dict[str, Any]) -> None:
        c["status"] = status
        if status in ("resolved", "false_positive"):
            c["closed"] = time.strftime("%Y-%m-%d %H:%M")

    return _change(by, incident_id, "case_status", {"status": status}, fn)


def assign(by: str, incident_id: str, assignee: str) -> dict[str, Any]:
    from src.soc import users

    if assignee and assignee not in users.load()["users"]:
        raise ValueError(f"no user {assignee!r}")

    def fn(c: dict[str, Any]) -> None:
        c["assignee"] = assignee
        if c.get("status") == "new" and assignee:
            c["status"] = "investigating"

    return _change(by, incident_id, "case_assigned", {"assignee": assignee}, fn)


def add_note(by: str, incident_id: str, text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty note")
    if len(text) > MAX_NOTE:
        raise ValueError(f"note longer than {MAX_NOTE} characters")

    def fn(c: dict[str, Any]) -> None:
        c.setdefault("notes", []).append({"who": by, "when": time.strftime("%Y-%m-%d %H:%M"), "text": text})

    return _change(by, incident_id, "case_note", {"chars": len(text)}, fn)


def summary() -> dict[str, int]:
    """Cases per status (incidents never touched count as new elsewhere)."""
    out = {s: 0 for s in STATUSES}
    for c in _load().values():
        out[c.get("status", "new")] = out.get(c.get("status", "new"), 0) + 1
    return out
