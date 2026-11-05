"""Users adapt the built-in rules and engine settings to their site (owner 2026-10-24).

Every detection records which rule produced which technique (``ev["rule_hits"]``:
``sigma:<id>``, ``crs:<id>``, ``own:<kind>``, ``lolbas:<program>``, ``ids:<signature>``).
On top of that, a user can - from the dashboard's Customise page or a rule pack:

  * switch any built-in rule off (or back on);
  * override its severity (1-10) or its ATT&CK technique(s);
  * add exceptions to it ("not when host is backup01", contains / equals / regex);
  * change engine settings (correlation gap, risk board, OWASP CRS paranoia level and
    threshold), each shown next to its validated value;
  * export / import everything above plus their own rules as one rule pack;
  * accept or reject PROPOSALS the system makes from the analysts' reviews (a rule
    whose reviewed alerts were all false alarms, a rule whose false alarms all come
    from one host or user). Nothing changes until a user approves; every change is
    undoable and written to the tamper-evident audit log.

Overrides apply to events from now on (stored events keep what they had); ``apply``
never raises - a broken override is skipped.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from utils.logging_config import get_logger
from utils.paths import PROJECT_ROOT

_log = get_logger(__name__)
HOME = PROJECT_ROOT / "artifacts" / "soc"
OVERRIDES = HOME / "rule_overrides.json"
SETTINGS = HOME / "engine_settings.json"
PROPOSALS = HOME / "rule_proposals.json"
HISTORY = HOME / "rule_history.jsonl"
Event = dict[str, Any]
_TID = re.compile(r"^T\d{4}(?:\.\d{3})?$")
EX_FIELDS = ("any", "detail", "kind", "source", "user", "host", "observer", "src_ip", "dst_ip", "dst_port",
             "file", "url")  # fmt: skip
EX_OPS = ("contains", "equals", "regex")
MIN_REVIEWS = 5  # reviewed alerts of one rule before a proposal is made
EXCEPTION_SHARE = 0.8  # false alarms concentrated on one value -> propose an exception

# engine settings: key -> (validated / default value, min, max, what it does)
ENGINE: dict[str, tuple[float, float, float, str]] = {
    "correlation_gap_s": (900.0, 60.0, 86_400.0, "events of one entity further apart than this start a new incident"),
    "riskboard_weight": (-1.0, 0.0, 3.0, "severity points per other sensor family that agrees (risk board)"),
    "riskboard_cap": (-1.0, 0.0, 4.0, "most points the risk board may add"),
    "riskboard_hops": (-1.0, 0.0, 1.0, "0 = same entity only, 1 = also entities one link away"),
    "riskboard_follow_s": (-1.0, 0.0, 86_400.0, "follow an attacker onto the machine it attacked for this long"),
    "crs_paranoia": (1.0, 1.0, 4.0, "OWASP CRS paranoia level (higher = more rules, more false alarms)"),
    "crs_threshold": (5.0, 2.0, 50.0, "OWASP CRS inbound anomaly score that raises an alert"),
}  # fmt: skip


ACTOR = ["local"]  # the dashboard sets the signed-in user before a change


def _who() -> str:
    return ACTOR[0] or "local"


def _read(path: Path, default: Any) -> Any:  # noqa: ANN401 - JSON
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, doc: Any) -> None:  # noqa: ANN401 - JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str), "utf-8")
    tmp.replace(path)


def _audit(action: str, details: dict[str, Any]) -> None:
    from src.soc import audit

    audit.record(_who(), action, details)
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.time(), "when": time.strftime("%Y-%m-%d %H:%M"), "who": _who(),
                             "action": action, **details}, default=str) + "\n")  # fmt: skip


# --- attribution (called by the parsers) ------------------------------------------------


def hit(ev: Event, rule_id: str, techniques: list[str] | tuple[str, ...], title: str = "") -> None:
    """Record that ``rule_id`` gave ``techniques`` to this event."""
    if not techniques:
        return
    ev.setdefault("rule_hits", []).append({"id": rule_id, "title": title[:120], "techniques": list(techniques)})


# --- overrides ----------------------------------------------------------------------------

_CACHE: dict[str, Any] = {"key": None, "doc": {}}


def overrides() -> dict[str, dict[str, Any]]:
    try:
        key = (str(OVERRIDES), OVERRIDES.stat().st_mtime_ns)
    except OSError:
        return {}
    if _CACHE["key"] != key:
        _CACHE["key"], _CACHE["doc"] = key, _read(OVERRIDES, {})
    doc: dict[str, dict[str, Any]] = _CACHE["doc"]
    return doc


def _check_override(o: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "enabled" in o:
        out["enabled"] = bool(o["enabled"])
    if o.get("severity") is not None:
        s = int(o["severity"])
        if not 1 <= s <= 10:
            raise ValueError("severity must be 1-10")
        out["severity"] = s
    if o.get("techniques"):
        t = [str(x).strip().upper() for x in o["techniques"]]
        bad = [x for x in t if not _TID.match(x)]
        if bad:
            raise ValueError(f"not ATT&CK technique ids: {bad}")
        out["techniques"] = t
    exc = []
    for e in o.get("exceptions") or []:
        f, op, v = str(e.get("field", "any")), str(e.get("op", "contains")), str(e.get("value", ""))
        if f not in EX_FIELDS or op not in EX_OPS or not v:
            raise ValueError(f"bad exception {e}")
        if op == "regex":
            from src.soc.custom import nested_repetition, regex_runs_in_time

            re.compile(v)
            if nested_repetition(v) or not regex_runs_in_time(v):
                raise ValueError("regex too slow (nested repetition)")
        exc.append({"field": f, "op": op, "value": v})
    if exc:
        out["exceptions"] = exc
    return out


def set_override(rule_id: str, change: dict[str, Any] | None, reason: str = "") -> dict[str, Any]:
    """Set (or with ``None`` remove) the override of one rule; audited."""
    doc = dict(_read(OVERRIDES, {}))
    before = doc.get(rule_id)
    if change is None:
        doc.pop(rule_id, None)
    else:
        doc[rule_id] = _check_override(change) | {"by": _who(), "when": time.strftime("%Y-%m-%d %H:%M"),
                                                  "reason": reason[:300]}  # fmt: skip
    _write(OVERRIDES, doc)
    _CACHE.update(key=None, doc={})
    _audit("rule_override", {"rule": rule_id, "before": before, "after": doc.get(rule_id), "reason": reason})
    return doc.get(rule_id) or {}


def _excepted(ev: Event, exceptions: list[dict[str, Any]]) -> bool:
    for e in exceptions:
        val = json.dumps(ev, default=str) if e["field"] == "any" else str(ev.get(e["field"], ""))
        v, low = e["value"], val.lower()
        if (e["op"] == "contains" and v.lower() in low) or (e["op"] == "equals" and low == v.lower()) or (
            e["op"] == "regex" and re.search(v, val)
        ):
            return True
    return False


def apply(events: list[Event]) -> int:
    """Apply the users' overrides to fresh events; returns how many events changed."""
    ov = overrides()
    if not ov:
        # nothing overridden: events re-checked after an undo get their own techniques
        for ev in events:
            if ev.get("rule_hits"):
                ev["techniques"] = list(dict.fromkeys([*ev.get("techniques", []),
                                                       *(t for h in ev["rule_hits"] for t in h["techniques"])]))  # fmt: skip
        return 0
    changed = 0
    for ev in events:
        hits = ev.get("rule_hits")
        if not hits:
            continue
        try:
            attributed = {t for h in hits for t in h["techniques"]}
            keep = [t for t in ev.get("techniques", []) if t not in attributed]
            sev: list[int] = []
            off: list[str] = []
            for h in hits:
                o = ov.get(h["id"])
                if o is None:
                    keep += h["techniques"]
                    continue
                if o.get("enabled") is False or _excepted(ev, o.get("exceptions") or []):
                    off.append(h["id"])
                    continue
                keep += o.get("techniques") or h["techniques"]
                if o.get("severity"):
                    sev.append(int(o["severity"]))
            new = list(dict.fromkeys(keep))
            if new != ev.get("techniques") or off or sev:
                changed += 1
                ev["techniques"] = new
                if off:
                    ev["rules_off"] = off
                if sev:
                    ev["severity_hint"] = max(sev)
        except (KeyError, TypeError, re.error) as exc:  # a broken override never stops ingest
            _log.warning("rule override skipped: %s", exc)
    return changed


def reapply_store() -> dict[str, int]:
    """Re-check every stored event with the current rule changes (owner 2026-10-25).
    Events keep their original rule hits, so a rule switched back on gets its
    techniques back too. The store is rewritten in place under its lock (like
    retention), then the search index is rebuilt. Alerts already sent stay as sent."""
    import json as _json
    import os

    from src.soc import engine, eventindex, retention

    path = Path(engine.EVENTS)
    if not path.is_file() or path.stat().st_size == 0:
        return {"events": 0, "changed": 0}
    tmp = path.with_name(path.name + ".reapply.tmp")
    n = changed = 0
    with engine._file_lock(path.with_name(path.name + ".lock")):
        with open(path, "rb") as src, open(tmp, "wb") as out:
            for line in src:
                if not line.endswith(b"\n") or b'"rule_hits"' not in line:
                    out.write(line)
                    n += line.endswith(b"\n")
                    continue
                n += 1
                try:
                    ev = _json.loads(line)
                except ValueError:
                    out.write(line)
                    continue
                before = (ev.get("techniques"), ev.get("rules_off"), ev.get("severity_hint"))
                ev.pop("rules_off", None)
                ev.pop("severity_hint", None)
                apply([ev])
                if (ev.get("techniques"), ev.get("rules_off"), ev.get("severity_hint")) != before:
                    changed += 1
                    out.write((_json.dumps(ev, default=str) + "\n").encode("utf-8"))
                else:
                    out.write(line)
            out.flush()
            os.fsync(out.fileno())
        if changed:
            m = retention._marker(path)
            m.write_text(_json.dumps({"tmp": str(tmp), "phase": "copy"}), "utf-8")
            retention._copy_back(path, tmp)
            m.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    if changed:
        eventindex.invalidate()
    _audit("rules_reapplied", {"events": n, "changed": changed})
    return {"events": n, "changed": changed}


# --- engine settings -------------------------------------------------------------------------


def validated_engine() -> dict[str, float]:
    from src.soc import riskboard

    rb = riskboard.validated_settings()
    out = {k: v[0] for k, v in ENGINE.items()}
    out |= {"riskboard_weight": float(rb["weight"]), "riskboard_cap": float(rb["cap"]),
            "riskboard_hops": float(rb["hops"]), "riskboard_follow_s": float(rb["follow_s"])}  # fmt: skip
    return out


_ES: dict[str, Any] = {"key": None, "val": {}}


def _mtime(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return 0


def engine_settings() -> dict[str, float]:
    """Validated values with the users' changes on top (cached until either file changes)."""
    from src.soc import riskboard

    key = (str(SETTINGS), _mtime(SETTINGS), str(riskboard.VALIDATED), _mtime(riskboard.VALIDATED))
    if _ES["key"] == key:
        cached: dict[str, float] = _ES["val"]
        return dict(cached)
    _ES["key"], _ES["val"] = key, _engine_settings()
    return dict(_ES["val"])


def _engine_settings() -> dict[str, float]:
    user = _read(SETTINGS, {})
    out = validated_engine()
    for k, v in user.items():
        if k in ENGINE:
            lo, hi = ENGINE[k][1], ENGINE[k][2]
            out[k] = float(min(max(float(v), lo), hi))
    return out


def set_engine(key: str, value: float | None, reason: str = "") -> dict[str, float]:
    if key not in ENGINE:
        raise ValueError(f"unknown setting {key}")
    doc = dict(_read(SETTINGS, {}))
    before = doc.get(key)
    if value is None:
        doc.pop(key, None)
    else:
        lo, hi = ENGINE[key][1], ENGINE[key][2]
        if not lo <= float(value) <= hi:
            raise ValueError(f"{key} must be within {lo:g}-{hi:g}")
        doc[key] = float(value)
    _write(SETTINGS, doc)
    _ES.update(key=None, val={})  # two writes within one file-time tick must not hit a stale cache
    _audit("engine_setting", {"setting": key, "before": before, "after": doc.get(key),
                              "validated": validated_engine()[key], "reason": reason})  # fmt: skip
    return engine_settings()


# --- catalogue -------------------------------------------------------------------------------


def catalog() -> list[dict[str, Any]]:
    """Every built-in rule a user can switch: Sigma, OWASP CRS, LOLBAS / GTFOBins."""
    out: list[dict[str, Any]] = []
    try:
        from src.soc import sigma

        for r in sigma.load()["all"]:
            ls = r.source or ("windows", None, r.service)
            out.append({"id": f"sigma:{r.rule_id}", "kind": "Sigma", "title": r.title, "level": r.level,
                        "techniques": list(r.techniques), "logsource": " / ".join(x for x in ls if x) or "windows"})  # fmt: skip
    except Exception as exc:  # noqa: BLE001 - rules folder missing
        _log.warning("sigma catalogue: %s", exc)
    try:
        from src.soc import crs

        level: dict[str, int] = {}
        for pl in (4, 3, 2, 1):
            for r in crs.parse_rules(crs.RULES_DIR, pl)[0]:
                level[r.id] = pl
        for r in crs.parse_rules(crs.RULES_DIR, 4)[0]:
            out.append({"id": f"crs:{r.id}", "kind": "OWASP CRS", "title": r.msg, "level": f"PL{level.get(r.id, 4)}",
                        "techniques": list(crs.FILE_TECHNIQUES.get(r.file, [])), "logsource": "web"})  # fmt: skip
    except Exception as exc:  # noqa: BLE001
        _log.warning("crs catalogue: %s", exc)
    try:
        from src.soc import lolbins

        seen = set()
        for (osn, b), pats in lolbins._index().items():
            if (osn, b) in seen:
                continue
            seen.add((osn, b))
            out.append({"id": f"lolbas:{b}", "kind": "LOLBAS" if osn == "windows" else "GTFOBins",
                        "title": f"misuse of {b}", "level": "medium",
                        "techniques": sorted({p["technique"] for p in pats}), "logsource": osn})  # fmt: skip
    except Exception as exc:  # noqa: BLE001
        _log.warning("lolbins catalogue: %s", exc)
    return out


def disabled_ids(prefix: str) -> set[str]:
    """Rule ids (without the prefix) a user switched off, e.g. disabled_ids("crs:")."""
    return {k[len(prefix):] for k, v in overrides().items() if k.startswith(prefix) and v.get("enabled") is False}


def fired(events: list[Event]) -> Counter[str]:
    """How often each rule fired in these stored events (own rules appear here)."""
    return Counter(h["id"] for e in events for h in e.get("rule_hits") or [])


# --- rule packs --------------------------------------------------------------------------------


def export_pack() -> dict[str, Any]:
    from src.soc import custom

    return {"format": "splitshield-rule-pack/1", "exported": time.strftime("%Y-%m-%d %H:%M"), "by": _who(),
            "custom_rules": custom.load_docs(), "overrides": _read(OVERRIDES, {}),
            "engine_settings": _read(SETTINGS, {})}  # fmt: skip


def import_pack(pack: dict[str, Any], events: list[Event], apply_now: bool = False,
                accept_noisy: bool = False) -> dict[str, Any]:  # fmt: skip
    """Check a pack (every custom rule compiled and tested on stored events, every
    override and setting validated); with ``apply_now`` switch it on."""
    from src.soc import custom

    if pack.get("format") != "splitshield-rule-pack/1":
        raise ValueError("not a SplitShield rule pack")
    report: dict[str, Any] = {"custom_rules": [], "overrides": [], "engine_settings": [], "ok": True}
    for doc in pack.get("custom_rules") or []:
        try:
            r = custom.test_rule(doc, events)
            noisy = r.get("rate", 0) > custom.MAX_RATE
            report["custom_rules"].append({"name": doc.get("name"), "ok": not noisy or accept_noisy, **r})
            report["ok"] &= not noisy or accept_noisy
        except ValueError as exc:
            report["custom_rules"].append({"name": doc.get("name"), "ok": False, "error": str(exc)})
            report["ok"] = False
    for rid, o in (pack.get("overrides") or {}).items():
        try:
            _check_override(o)
            report["overrides"].append({"rule": rid, "ok": True})
        except (ValueError, re.error) as exc:
            report["overrides"].append({"rule": rid, "ok": False, "error": str(exc)})
            report["ok"] = False
    for k, v in (pack.get("engine_settings") or {}).items():
        ok = k in ENGINE and ENGINE[k][1] <= float(v) <= ENGINE[k][2]
        report["engine_settings"].append({"setting": k, "value": v, "ok": ok})
        report["ok"] &= ok
    if apply_now and report["ok"]:
        for doc in pack.get("custom_rules") or []:
            custom.save_rule(doc, events, accept_noisy=accept_noisy)
        for rid, o in (pack.get("overrides") or {}).items():
            set_override(rid, o, reason="rule pack import")
        for k, v in (pack.get("engine_settings") or {}).items():
            set_engine(k, float(v), reason="rule pack import")
        report["applied"] = True
    return report


# --- proposals from reviews ----------------------------------------------------------------------


def propose(alerts: list[Event] | None = None, reviews: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Rule changes suggested by the analysts' reviews (Real / False alarm). Kept in
    rule_proposals.json until a user accepts or rejects them."""
    from src.soc import engine
    from src.soc.rcl import RCL

    alerts = alerts if alerts is not None else engine.read_jsonl(engine.ALERTS)
    reviews = reviews if reviews is not None else RCL().rows()
    by_id = {a.get("id"): a for a in alerts}
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"fp": 0, "tp": 0, "values": Counter()})
    for r in reviews:
        a = by_id.get(r.get("alert_id"))
        if not a:
            continue
        for h in a.get("rule_hits") or []:
            s = stats[h["id"]]
            if r.get("correct"):
                s["tp"] += 1
            else:
                s["fp"] += 1
                for f in ("host", "user", "src_ip", "observer"):
                    if a.get(f):
                        s["values"][(f, str(a[f]))] += 1
    old = {p["id"]: p for p in _read(PROPOSALS, [])}
    ov = overrides()
    out = []
    for rid, s in stats.items():
        n = s["fp"] + s["tp"]
        if n < MIN_REVIEWS or s["fp"] == 0 or ov.get(rid, {}).get("enabled") is False:
            continue
        (f, v), cnt = s["values"].most_common(1)[0] if s["values"] else (("", ""), 0)
        if s["tp"] == 0 and cnt < EXCEPTION_SHARE * s["fp"]:
            change: dict[str, Any] = {"enabled": False}
            why = f"all {s['fp']} reviewed alerts of this rule were false alarms"
        elif cnt >= EXCEPTION_SHARE * s["fp"] and s["fp"] >= MIN_REVIEWS:
            change = {"exceptions": [*(ov.get(rid, {}).get("exceptions") or []), {"field": f, "op": "equals", "value": v}]}
            why = f"{cnt} of its {s['fp']} false alarms come from {f} {v} ({s['tp']} real alerts kept)"
        else:
            continue
        pid = f"{rid}|{json.dumps(change, sort_keys=True)}"
        prev = old.get(pid, {})
        out.append({"id": pid, "rule": rid, "change": change, "why": why, "false_alarms": s["fp"], "real": s["tp"],
                    "status": prev.get("status", "open"), "made": prev.get("made", time.strftime("%Y-%m-%d %H:%M"))})  # fmt: skip
    out += [p for p in old.values() if (p["status"] != "open" or p["rule"].startswith(("engine:", "sens:")))
            and p["id"] not in {q["id"] for q in out}]  # fmt: skip
    _write(PROPOSALS, out)
    return out


def add_proposal(target: str, change: dict[str, Any], why: str) -> dict[str, Any]:
    """A proposal from another part of the system (e.g. the risk board's review check);
    ``target`` "engine:<setting>" changes an engine setting when accepted."""
    props = _read(PROPOSALS, [])
    pid = f"{target}|{json.dumps(change, sort_keys=True)}"
    p = next((x for x in props if x["id"] == pid), None)
    if p is None:
        p = {"id": pid, "rule": target, "change": change, "why": why, "false_alarms": 0, "real": 0,
             "status": "open", "made": time.strftime("%Y-%m-%d %H:%M")}  # fmt: skip
        props.append(p)
        _write(PROPOSALS, props)
    return p


def decide(proposal_id: str, accept: bool) -> dict[str, Any]:
    props = _read(PROPOSALS, [])
    p = next((x for x in props if x["id"] == proposal_id), None)
    if p is None:
        raise ValueError("no such proposal")
    if accept and p["rule"].startswith("engine:"):
        set_engine(p["rule"].split(":", 1)[1], float(p["change"]["value"]), reason=f"accepted proposal: {p['why']}")
    elif accept and p["rule"].startswith("sens:"):  # a model threshold (Customise -> Sensitivity)
        from src.soc import custom

        model = p["rule"].split(":", 1)[1]
        base = custom.validated(model)
        if base is None:
            raise ValueError(f"model {model} not available")
        custom.set_threshold(model, float(p["change"]["value"]), base)
        _audit("sensitivity", {"model": model, "threshold": p["change"]["value"], "from_proposal": p["why"]})
    elif accept:
        cur = {k: v for k, v in overrides().get(p["rule"], {}).items() if k in ("enabled", "severity", "techniques", "exceptions")}
        set_override(p["rule"], cur | p["change"], reason=f"accepted proposal: {p['why']}")
    p["status"] = "accepted" if accept else "rejected"
    p["decided"] = time.strftime("%Y-%m-%d %H:%M")
    _write(PROPOSALS, props)
    _audit("rule_proposal", {"proposal": proposal_id, "accepted": accept})
    return p


def history(last: int = 200) -> list[dict[str, Any]]:
    try:
        lines = HISTORY.read_text("utf-8").splitlines()[-last:]
    except OSError:
        return []
    return [json.loads(x) for x in lines if x.strip()]
