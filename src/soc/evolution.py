"""Evolving or frozen (owner 2026-10-16): the user decides whether the models keep
learning by themselves.

    mode "evolving" (default)  every model may be updated by the self-improvement
                               controller - and only kept when held-out precision
                               and recall do not drop and it is not slower;
    mode "frozen"              no automatic update of any model.
    frozen list                with "evolving", single components can be frozen.

Components: the 9 models of config/models.yaml plus "rules" (the log-rule engine's
self-tuning loop, src/soc/evolve.py). Frozen components still detect, alert and keep
collecting reviews and inbox files, so evolution can resume later with all of it.
Manual retrain buttons stay available to users allowed to change models.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

MODES = ("evolving", "frozen")


def path() -> Path:
    from src.soc import engine

    return Path(engine.EVENTS).with_name("evolution.json")


def components() -> list[str]:
    from src.soc import controller

    return [*controller.registry(), "rules"]


def load() -> dict[str, Any]:
    try:
        d = json.loads(path().read_text("utf-8"))
    except (OSError, ValueError):
        d = {}
    mode = d.get("mode", "evolving")
    return {"mode": mode if mode in MODES else "evolving",
            "frozen": sorted(set(d.get("frozen", []))), "changed": d.get("changed")}  # fmt: skip


def allowed(component: str) -> bool:
    """May ``component`` be updated automatically right now?"""
    s = load()
    return s["mode"] == "evolving" and component not in s["frozen"]


def save(mode: str, frozen: list[str] | None = None, who: str = "") -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unknown evolution mode {mode!r}")
    known = set(components())
    frozen = sorted(set(frozen or []))
    if bad := [c for c in frozen if c not in known]:
        raise ValueError(f"unknown component(s): {', '.join(bad)}")
    doc = {
        "mode": mode,
        "frozen": frozen,
        "changed": time.strftime("%Y-%m-%d %H:%M"),
        "by": who,
    }
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1), "utf-8")
    try:
        from src.soc import audit

        audit.record(who or "system", "evolution", {"mode": mode, "frozen": frozen})
    except ImportError:
        pass
    return doc
