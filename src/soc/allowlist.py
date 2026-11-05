"""Allow-list: known-good IPs, users, hosts, files and domains (owner 2026-10-15).

Owner decision "mute, keep visible": an alert or incident that involves an
allow-listed entity is still STORED, searchable and shown on the dashboard (greyed,
with who/when/why), but it is not forwarded to the SIEM / webhooks and not counted
in report alert totals. CRITICAL incidents (severity >= 8) are still forwarded,
marked "allow-listed entity involved" - an attacker using a trusted account must
never become invisible.

Every entry needs a reason and expires (default 30 days, at most 365); expired
entries stop applying and are shown as expired. Stored in
artifacts/soc/allowlist.json.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from utils.paths import PROJECT_ROOT

PATH = PROJECT_ROOT / "artifacts" / "soc" / "allowlist.json"
TYPES = ("ip", "user", "host", "file", "domain")
DEFAULT_DAYS = 30
MAX_DAYS = 365
CRITICAL = 8
_DAY = 86400.0
_CACHE: dict[str, Any] = {"key": None, "entries": []}


def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def entries(path: Path | None = None) -> list[dict[str, Any]]:
    """All entries (active and expired), each with ``active`` set."""
    p = path or PATH
    try:
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
    except OSError:
        return []
    if _CACHE["key"] != key:
        try:
            rows = json.loads(p.read_text("utf-8"))
        except (OSError, ValueError):
            rows = []
        _CACHE.update(key=key, entries=rows if isinstance(rows, list) else [])
    now = _now()
    return [e | {"active": float(e.get("expires_ts", 0)) > now} for e in _CACHE["entries"]]


def _save(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    p = path or PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in r.items() if k != "active"} for r in rows]
    p.write_text(json.dumps(clean, indent=1), "utf-8")


def normalise(kind: str, value: str) -> str:
    v = str(value).strip()
    if kind in ("user", "host", "domain", "file"):
        v = v.lower()
    if kind == "domain":
        v = v.removeprefix("http://").removeprefix("https://").split("/")[0].lstrip(".")
    return v


def add(
    kind: str, value: str, reason: str, days: int = DEFAULT_DAYS, path: Path | None = None
) -> dict[str, Any]:
    """New entry; raises ValueError for a bad type, empty value/reason or expiry."""
    if kind not in TYPES:
        raise ValueError(f"type must be one of {', '.join(TYPES)}")
    v = normalise(kind, value)
    if not v:
        raise ValueError("value is empty")
    if not str(reason).strip():
        raise ValueError("a reason is required")
    if not 1 <= int(days) <= MAX_DAYS:
        raise ValueError(f"expiry must be 1-{MAX_DAYS} days")
    now = _now()
    row = {"id": uuid.uuid4().hex[:10], "type": kind, "value": v,
           "reason": str(reason).strip()[:300], "added": _iso(now), "added_ts": now,
           "expires": _iso(now + int(days) * _DAY), "expires_ts": now + int(days) * _DAY}  # fmt: skip
    rows = [e for e in entries(path)]
    rows.append(row)
    _save(rows, path)
    return row


def remove(entry_id: str, path: Path | None = None) -> bool:
    rows = entries(path)
    keep = [e for e in rows if e["id"] != entry_id]
    if len(keep) == len(rows):
        return False
    _save(keep, path)
    return True


def _keys_of_event(ev: dict[str, Any]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if ev.get("src_ip"):
        keys.add(("ip", str(ev["src_ip"]).strip()))
    if ev.get("user"):
        keys.add(("user", str(ev["user"]).strip().lower()))
    if ev.get("host"):
        keys.add(("host", str(ev["host"]).strip().lower()))
    for f in ("sha256", "file"):
        if ev.get(f):
            keys.add(("file", str(ev[f]).strip().lower()))
    try:
        from src.soc.riskboard import urls_in

        for u in urls_in(ev):
            keys.add(("domain", normalise("domain", u)))
    except Exception:  # noqa: BLE001 - URL extraction is best effort
        pass
    return keys


def _keys_of_incident(inc: dict[str, Any]) -> set[tuple[str, str]]:
    ent = str(inc.get("src_ip", ""))
    kind, sep, val = ent.partition(":")
    if sep and kind in ("user", "host"):
        return {(kind, val.lower())}
    return {("ip", ent)} if ent else set()


def _match(keys: set[tuple[str, str]], e: dict[str, Any]) -> bool:
    t, v = e["type"], e["value"]
    for kind, val in keys:
        if kind != t:
            continue
        if val == v or (t == "domain" and val.endswith("." + v)):
            return True
    return False


def is_muted(
    obj: dict[str, Any],
    path: Path | None = None,
    *,
    active: list[dict[str, Any]] | None = None,
) -> tuple[bool, str | None]:
    """(muted?, "allow-listed by you on <date>: <reason>") for an event, alert or
    incident. Used by the engine (no forwarding) and by reports (not counted).
    ``active``: the active entries, when checking many objects in one go."""
    if active is None:
        active = [e for e in entries(path) if e["active"]]
    if not active:
        return False, None
    keys = _keys_of_incident(obj) if "start" in obj and "end" in obj else _keys_of_event(obj)
    for e in active:
        if _match(keys, e):
            return True, (f"allow-listed by you on {e['added']}: {e['reason']} "
                          f"({e['type']} {e['value']}, until {e['expires']})")  # fmt: skip
    return False, None


def forward(obj: dict[str, Any], path: Path | None = None) -> tuple[bool, str | None]:
    """Should this alert / incident still be forwarded, and with which mark?
    Not muted -> (True, None); muted -> (False, reason); muted but critical ->
    (True, "allow-listed entity involved: ...")."""
    muted, reason = is_muted(obj, path)
    if not muted:
        return True, None
    if float(obj.get("severity") or 0) >= CRITICAL:
        return True, f"allow-listed entity involved: {reason}"
    return False, reason
