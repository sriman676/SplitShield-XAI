"""Where the SOC runs, and how alerting adapts to it (owner 2026-10-15).

Modes (owner decision):
  * confirm (default) - the place is GUESSED from what the SOC sees and shown on the
    dashboard; the owner confirms or changes it. Re-checked weekly; a different
    guess raises a notice instead of switching.
  * manual            - the owner's pick always wins; no guessing.
  * auto (opt-in)     - the guess is applied by itself and the owner is notified.
In every mode a profile changes only push budgets and risk-board escalation
(config/places.yaml); it never switches a detector off.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from utils.paths import PROJECT_ROOT

PROFILES = PROJECT_ROOT / "config" / "places.yaml"
STATE = PROJECT_ROOT / "artifacts" / "soc" / "place.json"
DEFAULT_PLACE = "home_pc"
RECHECK_S = 7 * 24 * 3600


_PROFILES_CACHE: dict[str, tuple[int, dict[str, dict[str, Any]]]] = {}


def profiles(path: Path = PROFILES) -> dict[str, dict[str, Any]]:
    """config/places.yaml, parsed again only when the file changed (it is read on
    every incident view)."""
    import copy

    import yaml

    mtime = path.stat().st_mtime_ns
    hit = _PROFILES_CACHE.get(str(path))
    if hit is None or hit[0] != mtime:
        hit = _PROFILES_CACHE[str(path)] = (
            mtime,
            yaml.safe_load(path.read_text("utf-8")),
        )
    return copy.deepcopy(hit[1])


def state(path: Path | None = None) -> dict[str, Any]:
    try:
        st: dict[str, Any] = json.loads((path or STATE).read_text("utf-8"))
    except (OSError, ValueError):
        st = {}
    return {"mode": "confirm", "place": None, "guess": None, "reasons": []} | st


def _save(st: dict[str, Any], path: Path | None = None) -> None:
    path = path or STATE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st, indent=1), "utf-8")


def guess(events: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Best guess of the place from stored events, with the reasons."""
    if not events:
        return DEFAULT_PLACE, ["no logs ingested yet: assuming a personal computer"]
    from src.soc.riskboard import family_of

    fams = Counter(family_of(e) for e in events)
    hosts = {str(e["host"]).lower() for e in events if e.get("host")}
    users = {str(e["user"]).lower() for e in events if e.get("user")}
    now = time.time()
    old = sum(1 for e in events if now - float(e.get("ts") or now) > 30 * 86400)
    attack = sum(1 for e in events if e.get("techniques"))
    n = len(events)
    if old / n > 0.5 and attack / n > 0.2:
        return "lab", [f"{old / n:.0%} of events are older than 30 days and {attack / n:.0%} "
                       "carry attack techniques: looks like replayed datasets"]  # fmt: skip
    if fams["web"] + fams["mail"] > 0.3 * n:
        return "server", [f"{(fams['web'] + fams['mail']) / n:.0%} of events are web / mail "
                          "server logs"]  # fmt: skip
    if len(hosts) > 3 or len(users) > 5:
        return "small_office", [f"{len(hosts)} hosts and {len(users)} users seen"]
    return "home_pc", [f"{max(1, len(hosts))} host(s) and {len(users)} user(s) seen"]


def current() -> dict[str, Any]:
    """The active profile (+ its name)."""
    st = state()
    name = (
        st["place"] or (st["guess"] if st["mode"] == "auto" else None) or DEFAULT_PLACE
    )
    profs = profiles()
    return {"name": name} | profs.get(name, profs[DEFAULT_PLACE])


def board_scale() -> float:
    try:
        return float(current().get("board_scale", 1.0))
    except Exception:  # noqa: BLE001 - a broken profile file must not stop correlation
        return 1.0


def set_place(place: str, mode: str = "manual") -> None:
    if place not in profiles():
        raise ValueError(f"unknown place {place!r}")
    if mode not in ("manual", "confirm", "auto"):
        raise ValueError(f"unknown mode {mode!r}")
    st = state()
    st.update(place=place, mode=mode, confirmed=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _save(st)


def set_mode(mode: str) -> None:
    if mode not in ("manual", "confirm", "auto"):
        raise ValueError(f"unknown mode {mode!r}")
    st = state()
    st["mode"] = mode
    _save(st)


def check(events: list[dict[str, Any]], force: bool = False) -> dict[str, Any]:
    """Weekly re-guess. confirm: a different guess -> notice; auto: switch + notify;
    manual: nothing changes."""
    st = state()
    if not force and time.time() - float(st.get("checked_ts", 0)) < RECHECK_S:
        return st
    g, why = guess(events)
    st.update(guess=g, reasons=why, checked_ts=time.time())
    st.pop("notice", None)
    if st["mode"] == "auto" and g != st.get("place"):
        old, st["place"] = st.get("place"), g
        from src.soc.controller import notify

        notify(f"place changed automatically: {old or DEFAULT_PLACE} -> {g} ({why[0]})")
    elif st["mode"] == "confirm" and st.get("place") and g != st["place"]:
        st["notice"] = (f"This looks like a '{g}' now ({why[0]}), but '{st['place']}' is "
                        "set. Confirm or change it on Getting started.")  # fmt: skip
    _save(st)
    return st


# --- push budget -----------------------------------------------------------------------

BUDGET = PROJECT_ROOT / "artifacts" / "soc" / "push_budget.json"
CRITICAL = 8


def may_push(severity: float, path: Path | None = None) -> bool:
    """Within today's push budget? Critical incidents always pass (and are counted)."""
    path = path or BUDGET
    prof = current()
    if severity < float(prof.get("notify_min_severity", 6)):
        return False
    day = time.strftime("%Y-%m-%d")
    try:
        b = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        b = {}
    if b.get("day") != day:
        b = {"day": day, "n": 0}
    ok = severity >= CRITICAL or b["n"] < int(prof.get("max_pushes_per_day", 50))
    if ok:
        b["n"] += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(b), "utf-8")
    return ok
