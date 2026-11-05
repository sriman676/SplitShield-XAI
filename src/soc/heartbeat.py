"""Silent sources and agent clocks (owner 2026-10-22 gap audit).

* An agent that has sent nothing for ``silence_minutes`` (default 30; Agents & API
  settings) raises ONE alert "source went silent" (ATT&CK T1562 Impair Defenses: a
  stopped agent can be an attacker switching logging off), and one "source is back"
  event when it sends again. A silent source must never look like "no attacks".
* Agents send their own clock time with every batch; the receiver keeps the difference.
  More than ``MAX_SKEW_S`` (2 minutes) is shown on the Agents & API page and in the
  health check, because wrong clocks put events outside the correlation windows.

``check()`` runs every minute inside the log receiver and hourly from the autopilot.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

MAX_SKEW_S = 120.0
DEFAULT_SILENCE_MIN = 30


def _state_path() -> Path:
    from src.soc import collector

    return collector.home() / "heartbeat.json"


def _load() -> dict[str, Any]:
    try:
        return dict(json.loads(_state_path().read_text("utf-8")))
    except (OSError, ValueError):
        return {}


def _save(d: dict[str, Any]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1), "utf-8")
    tmp.replace(p)


def silence_minutes() -> int:
    from src.soc import collector

    return int(collector.settings().get("silence_minutes", DEFAULT_SILENCE_MIN))


def _seen_ts(entry: dict[str, Any]) -> float | None:
    v = entry.get("last_seen")
    if not v:
        return None
    try:
        return time.mktime(time.strptime(str(v), "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return None


def check(now: float | None = None, send: bool = True) -> list[dict[str, Any]]:
    """Raise 'silent' / 'back' events for agents; returns the events it stored."""
    from src.soc import collector, engine

    now = time.time() if now is None else now
    limit = silence_minutes() * 60
    st = _load()
    out: list[dict[str, Any]] = []
    for name, e in collector.listing("agents").items():
        if e.get("disabled"):
            continue
        seen = _seen_ts(e)
        if seen is None:
            continue  # never connected: nothing to miss yet
        silent = now - seen > limit
        was = bool(st.get(name, {}).get("silent"))
        if silent and not was:
            out.append({"ts": now, "source": "soc_health", "kind": "source_silent", "host": name,
                        "detail": f"agent {name} sent no logs for {int((now - seen) // 60)} minutes "
                                  f"(last from {e.get('last_ip') or '?'} at {e.get('last_seen')})",
                        "techniques": ["T1562"], "severity": 5})  # fmt: skip
        elif was and not silent:
            out.append({"ts": now, "source": "soc_health", "kind": "source_back", "host": name,
                        "detail": f"agent {name} is sending again", "techniques": []})  # fmt: skip
        st[name] = {"silent": silent, "checked": now}
    _save(st)
    if out:
        engine.record(out, send)
    return out


def skew(agent_time: Any, now: float | None = None) -> float | None:
    """Receiver time minus the agent's clock, in seconds (None if not sent)."""
    try:
        t = float(agent_time)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(t):  # "inf" / "nan" from an agent (pentest re-check 2026-10-23)
        return None
    return round((time.time() if now is None else now) - t, 1)


def clock_problems() -> dict[str, float]:
    """Agents whose clock is more than MAX_SKEW_S off."""
    from src.soc import collector

    return {n: float(e["skew_s"]) for n, e in collector.listing("agents").items()
            if e.get("skew_s") is not None and abs(float(e["skew_s"])) > MAX_SKEW_S}  # fmt: skip
