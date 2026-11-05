"""The owner's own detection rules + per-model sensitivity (owner 2026-10-15).

Custom rules live in artifacts/soc/custom_rules/<name>.yml, made on the dashboard's
⚙️ Customise page in one of two ways:

  * simple form: log family (or any), a field (or the whole event text), contains /
    regex / equals a value, an optional "N times in M minutes per entity" threshold,
    an ATT&CK technique and a severity;
  * Sigma YAML, compiled with the project's Sigma engine (src/soc/sigma.py). It is
    matched against the stored event fields (source, kind, detail, user, host,
    src_ip, dst_ip, dst_port, file, sha256, url) and keywords in any value.

Before a rule is saved it is TESTED on the owner's stored events; a rule that would
fire on more than ``MAX_RATE`` of them is refused unless the owner explicitly
accepts it. ``apply`` runs every enabled rule on each ingested batch (from
engine.record): a hit adds the rule's techniques and a "custom:<name>" marker. It
never raises - a broken rule is skipped and reported on the page.

Sensitivity (artifacts/soc/sensitivity.json): a model may be made stricter (fewer
alerts) or - since 2026-10-24, owner decision - looser than its validated threshold
(more alerts; the page shows the expected extra alerts on the stored data and the
change is audited), never below LOOSEST. ``threshold(model, validated)`` returns the
value to use. Wired in engine.record for: the flow detector (confirmed tier), the
phishing check of URLs in logs and the malware file scan (cross-check events).
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.logging_config import get_logger
from utils.paths import PROJECT_ROOT

_log = get_logger(__name__)
RULES_DIR = PROJECT_ROOT / "artifacts" / "soc" / "custom_rules"
SENSITIVITY = PROJECT_ROOT / "artifacts" / "soc" / "sensitivity.json"
MAX_RATE = 0.05  # refuse rules firing on > 5 % of stored events unless accepted
LOOSEST = 0.5  # a threshold below this would alert on a coin flip
FIELDS = ("any", "detail", "kind", "source", "user", "host", "src_ip", "dst_ip",
          "dst_port", "file", "sha256", "url")  # fmt: skip
OPS = ("contains", "regex", "equals")
_NAME = re.compile(r"^[A-Za-z0-9 _.-]{3,60}$")
_TID = re.compile(r"^T\d{4}(?:\.\d{3})?$")
Event = dict[str, Any]

# --- rules ---------------------------------------------------------------------------


@dataclass
class CustomRule:
    name: str
    doc: dict[str, Any]
    pred: Callable[[Event], bool]
    techniques: list[str]
    severity: int | None
    family: str = "any"
    count: int = 1
    minutes: float = 0.0
    seen: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def fires(self, ev: Event) -> bool:
        if self.family != "any":
            from src.soc.riskboard import family_of

            if family_of(ev) != self.family:
                return False
        if not self.pred(ev):
            return False
        if self.count <= 1:
            return True
        from src.soc.correlate import entity_of

        q = self.seen[entity_of(ev)]
        t = float(ev.get("ts") or time.time())
        q.append(t)
        while q and t - q[0] > self.minutes * 60:
            q.popleft()
        return len(q) >= self.count


REGEX_MAX = 4000  # characters of an event a regex rule looks at
_UNBOUNDED = re.compile(r"[+*]|\{\d*,\}")


def nested_repetition(pattern: str) -> bool:
    """True for a group repeated without bound that itself repeats something without
    bound - (a+)+, (\\w*)*, ((ab)*)+, (x+|y){2,} - the shape behind catastrophic
    backtracking. Bounded counts ({3}, {1,3}) cannot explode and are allowed."""
    stack: list[bool] = []  # per open group: does it contain unbounded repetition?
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "\\":
            i += 2
            continue
        if c == "[":  # a character class is one atom
            j = i + 1
            while j < n and pattern[j] != "]":
                j += 2 if pattern[j] == "\\" else 1
            i = j + 1
            continue
        if c == "(":
            stack.append(False)
        elif c == ")" and stack:
            inner = stack.pop()
            m = _UNBOUNDED.match(pattern, i + 1)
            if m and inner:
                return True
            if stack and (inner or m):
                stack[-1] = True
        elif (_UNBOUNDED.match(pattern, i) or c == "|") and stack:
            # alternatives inside a repeated group are just as ambiguous: (a|aa)+
            stack[-1] = True
        i += 1
    return False


_PROBE = """
import re, sys
p = sys.argv[1]
lits = "".join(dict.fromkeys(c for c in re.sub(r"\\\\.|\\[[^]]*\\]|[(){}|*+?.^$]", "", p))) or "a"
probes = [lits[0] * 40 + "!", lits * 15 + "!", (lits[:2] or "a") * 25 + "\\n!", " " * 40 + "!"]
rx = re.compile(p, re.I)
for s in probes:
    rx.search(s)
"""


def regex_runs_in_time(pattern: str, limit_s: float = 2.0) -> bool:
    """Run the pattern on worst-case text in a separate process; False if it does not
    finish within ``limit_s`` (catastrophic backtracking the shape check missed)."""
    import subprocess
    import sys

    try:
        subprocess.run([sys.executable, "-c", _PROBE, pattern], timeout=limit_s,
                       capture_output=True, check=False)  # fmt: skip
        return True
    except subprocess.TimeoutExpired:
        return False


def _text(ev: Event) -> str:
    return " ".join(str(v) for k, v in ev.items() if not k.startswith("_")).lower()


def _simple_pred(doc: dict[str, Any]) -> Callable[[Event], bool]:
    fld, op, val = doc["field"], doc["op"], str(doc["value"])
    if fld not in FIELDS or op not in OPS or not val:
        raise ValueError("field, operation and value are required")
    if op == "regex":
        if nested_repetition(val):  # 2026-10-16 hostile-input stress test
            raise ValueError(
                "this pattern repeats a group that itself repeats or has alternatives "
                "(like (a+)+ or (a|aa)+); a crafted log line could make it run for "
                "minutes and stall the SOC - write the words without repeating the group"
            )
        rx = re.compile(val, re.I)  # raises re.error for a bad pattern
        # only the first REGEX_MAX characters: bounded time per event, whatever the line
        test: Callable[[str], bool] = lambda s: rx.search(s[:REGEX_MAX]) is not None  # noqa: E731
    elif op == "contains":
        low = val.lower()
        test = lambda s: low in s.lower()  # noqa: E731
    else:
        low = val.lower()
        test = lambda s: s.strip().lower() == low  # noqa: E731

    def pred(ev: Event) -> bool:
        s = _text(ev) if fld == "any" else str(ev.get(fld) or "")
        return bool(s) and test(s)

    return pred


def _sigma_pred(text: str) -> tuple[Callable[[Event], bool], list[str], str]:
    import yaml

    from src.soc import sigma

    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or "detection" not in doc:
        raise ValueError("not a Sigma rule (no 'detection' section)")
    doc = dict(doc)
    doc.setdefault("logsource", {"product": "custom"})  # matched on any event
    rule = sigma.compile_rule(doc)

    def pred(ev: Event) -> bool:
        fields = {str(k).lower(): v for k, v in ev.items()
                  if not str(k).startswith("_") and not isinstance(v, dict | list)}  # fmt: skip
        sigma._BLOB[0] = None  # new event for the keyword cache
        try:
            return bool(rule.match(fields))
        except (TypeError, ValueError):
            return False

    return pred, list(rule.techniques), rule.title


def compile_doc(doc: dict[str, Any]) -> CustomRule:
    """A saved rule document -> runnable rule (raises ValueError when invalid)."""
    name = str(doc.get("name", "")).strip()
    if not _NAME.match(name):
        raise ValueError("name: 3-60 letters, digits, spaces, _ . -")
    techs = [t.strip().upper() for t in doc.get("techniques") or [] if str(t).strip()]
    if doc.get("kind") == "sigma":
        pred, sig_techs, _title = _sigma_pred(str(doc.get("sigma", "")))
        techs = techs or sig_techs
    else:
        pred = _simple_pred(doc)
    bad = [t for t in techs if not _TID.match(t)]
    if bad:
        raise ValueError(f"not ATT&CK technique ids: {bad}")
    if not techs:
        raise ValueError("pick at least one ATT&CK technique")
    sev = doc.get("severity")
    if sev is not None and not 1 <= int(sev) <= 10:
        raise ValueError("severity must be 1-10")
    count, minutes = int(doc.get("count") or 1), float(doc.get("minutes") or 0)
    if count > 1 and minutes <= 0:
        raise ValueError("a threshold needs a time window in minutes")
    return CustomRule(name, doc, pred, techs, None if sev is None else int(sev),
                      str(doc.get("family") or "any"), count, minutes)  # fmt: skip


def _file(name: str, folder: Path | None = None) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())[:60]
    return (folder or RULES_DIR) / f"{safe}.yml"


def load_docs(folder: Path | None = None) -> list[dict[str, Any]]:
    import yaml

    out = []
    for p in sorted((folder or RULES_DIR).glob("*.yml")):
        try:
            doc = yaml.safe_load(p.read_text("utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            out.append(
                {"name": p.stem, "error": f"unreadable: {exc}"[:200], "enabled": False}
            )
            continue
        if isinstance(doc, dict):
            out.append(doc)
    return out


_COMPILED: dict[str, Any] = {"key": None, "rules": [], "errors": {}}


def _state_key(folder: Path) -> tuple[Any, ...]:
    try:
        return tuple(
            (p.name, p.stat().st_mtime_ns) for p in sorted(folder.glob("*.yml"))
        )
    except OSError:
        return ()


def rules(folder: Path | None = None) -> tuple[list[CustomRule], dict[str, str]]:
    """Enabled, compiled rules (cached until a rule file changes) + compile errors."""
    folder = folder or RULES_DIR
    key = (str(folder), _state_key(folder))
    if _COMPILED["key"] != key:
        ok, errs = [], {}
        for d in load_docs(folder):
            if not d.get("enabled", True) or "error" in d:
                if "error" in d:
                    errs[str(d.get("name"))] = d["error"]
                continue
            try:
                ok.append(compile_doc(d))
            except Exception as exc:  # noqa: BLE001 - one bad rule must not stop the rest
                errs[str(d.get("name"))] = f"{type(exc).__name__}: {exc}"[:200]
        _COMPILED.update(key=key, rules=ok, errors=errs)
    return _COMPILED["rules"], _COMPILED["errors"]


def mark(ev: Event, rule: CustomRule) -> None:
    techs = list(ev.get("techniques") or [])
    ev["techniques"] = techs + [t for t in rule.techniques if t not in techs]
    ev.setdefault("custom_rules", [])
    marker = f"custom:{rule.name}"
    if marker not in ev["custom_rules"]:
        ev["custom_rules"].append(marker)
    if rule.severity is not None:
        ev["severity"] = max(int(ev.get("severity") or 0), rule.severity)


def apply(events: list[Event], folder: Path | None = None) -> int:
    """Run every enabled custom rule on a batch (in place). Returns hits; never raises."""
    try:
        active, _ = rules(folder)
    except Exception as exc:  # noqa: BLE001
        _log.warning("custom rules unavailable: %s", exc)
        return 0
    hits = 0
    for ev in events:
        for r in active:
            try:
                if r.fires(ev):
                    mark(ev, r)
                    hits += 1
            except Exception:  # noqa: BLE001 - never stop ingestion
                continue
    return hits


def test_rule(
    doc: dict[str, Any], events: list[Event], examples: int = 5
) -> dict[str, Any]:
    """Dry run on stored events: how often would the rule have fired?"""
    rule = compile_doc(doc)
    fired = [
        ev
        for ev in sorted(events, key=lambda e: float(e.get("ts") or 0))
        if rule.fires(ev)
    ]
    n = len(events)
    rate = len(fired) / n if n else 0.0
    return {"events": n, "fired": len(fired), "rate": rate, "too_noisy": rate > MAX_RATE,
            "examples": [{"ts": e.get("ts"), "source": e.get("source"), "kind": e.get("kind"),
                          "detail": str(e.get("detail", ""))[:160]} for e in fired[:examples]]}  # fmt: skip


def save_rule(
    doc: dict[str, Any],
    events: list[Event],
    accept_noisy: bool = False,
    folder: Path | None = None,
) -> dict[str, Any]:
    """Test, then save (refused when too noisy and not accepted). Returns the test."""
    import yaml

    if doc.get("op") == "regex" and not regex_runs_in_time(str(doc.get("value", ""))):
        raise ValueError("this regular expression takes too long on worst-case text "
                         "(it could stall the SOC on a crafted log line) - simplify it")  # fmt: skip
    result = test_rule(doc, events)
    if result["too_noisy"] and not accept_noisy:
        raise ValueError(f"would fire on {result['rate']:.1%} of your stored events "
                         f"(limit {MAX_RATE:.0%}); tick 'I understand' to save anyway")  # fmt: skip
    folder = folder or RULES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    doc = dict(doc) | {"enabled": doc.get("enabled", True),
                       "created": doc.get("created") or time.strftime("%Y-%m-%d %H:%M"),
                       "tested": {k: result[k] for k in ("events", "fired", "rate")}}  # fmt: skip
    _file(doc["name"], folder).write_text(yaml.safe_dump(doc, sort_keys=False), "utf-8")
    return result


def set_enabled(name: str, on: bool, folder: Path | None = None) -> None:
    import yaml

    p = _file(name, folder)
    doc = yaml.safe_load(p.read_text("utf-8"))
    doc["enabled"] = bool(on)
    p.write_text(yaml.safe_dump(doc, sort_keys=False), "utf-8")


def delete_rule(name: str, folder: Path | None = None) -> bool:
    """Delete one of the owner's own custom rule files (dashboard button)."""
    p = _file(name, folder)
    if p.is_file():
        p.unlink()
        return True
    return False


# --- sensitivity (stricter only) -----------------------------------------------------


_SENS_CACHE: dict[str, tuple[tuple[int, int], dict[str, float]]] = {}


def _sens(path: Path | None = None) -> dict[str, float]:
    """The owner's stricter thresholds; read again only when the file changed
    (2026-10-16: it was read once per stored event - a third of the ingest time)."""
    p = path or SENSITIVITY
    try:
        st = p.stat()
    except OSError:
        return {}
    key = (st.st_mtime_ns, st.st_size)
    hit = _SENS_CACHE.get(str(p))
    if hit is None or hit[0] != key:
        try:
            d = json.loads(p.read_text("utf-8"))
        except (OSError, ValueError):
            d = {}
        hit = _SENS_CACHE[str(p)] = (
            key,
            {k: float(v) for k, v in d.items()} if isinstance(d, dict) else {},
        )
    return dict(hit[1])


def threshold(model: str, validated: float, path: Path | None = None) -> float:
    """The threshold to use: the user's value (stricter or looser), else ``validated``."""
    v = _sens(path).get(model)
    return validated if v is None else max(LOOSEST, min(float(v), 0.9999))


def set_threshold(model: str, value: float | None, validated: float,
                  path: Path | None = None) -> float:  # fmt: skip
    """Store a user threshold (``None`` = back to validated); looser than validated is
    allowed down to LOOSEST (owner 2026-10-24: with a warning on the page)."""
    p = path or SENSITIVITY
    d = _sens(p)
    # the dashboard slider rounds to 4 decimals: the validated value shown as 0.9548
    # is "unchanged", not "looser" (2026-10-16, every-button stress test)
    if value is None or abs(float(value) - validated) < 5e-4:
        d.pop(model, None)
    else:
        if float(value) < LOOSEST:
            raise ValueError(f"{model}: {value} is below {LOOSEST}; that would alert on a coin flip")
        d[model] = min(float(value), 0.9999)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1), "utf-8")
    return threshold(model, validated, p)


def extra_alerts(model: str, value: float, events: list[Event]) -> dict[str, int]:
    """What a new threshold would change on these stored events (for the warning):
    alerts that would be added (looser) or removed (stricter), flow model only exact."""
    base = validated(model)
    if base is None:
        return {"added": 0, "removed": 0}
    added = removed = 0
    for e in events:
        if model == "flow" and e.get("source") == "detector":
            c = float(e.get("confidence") or 0)
            if e.get("tier") == "low" and c >= value:
                added += 1
            elif e.get("tier") == "confirmed" and c < value:
                removed += 1
        elif model in ("phishing", "malware") and e.get("source") == model:
            m = _P.search(str(e.get("detail", "")))
            if m and float(m.group(1)) < value:
                removed += 1
    return {"added": added, "removed": removed}


def validated(model: str) -> float | None:
    """The deployed (validated) threshold of a model the SOC applies live."""
    try:
        if model == "flow":
            from src.soc import engine

            det = engine.load_detector()
            v = getattr(det, "alert_threshold_high", None) if det is not None else None
            return float(v) if v is not None else 0.5
        if model == "phishing":
            from src.mlareas import phishing

            return float(phishing.load()["threshold"])
        if model == "malware":
            from src.mlareas import malware

            return float(malware.load_champion()[1]["thresholds"]["fpr_0.01"])
    except Exception:  # noqa: BLE001 - model missing: no control shown
        return None
    return None


LIVE_MODELS = {
    "flow": "Flow detector: attack probability needed for a confirmed alert",
    "phishing": "Phishing check of URLs found in logs: phishing probability needed",
    "malware": "Malware scan of files found in logs: malware probability needed",
}
_P = re.compile(r"\(p=([0-9.]+)\)")


def demote(
    ev: Event, sens_path: Path | None = None, sens: dict[str, float] | None = None
) -> bool:
    """Apply the owner's stricter thresholds to one event (in place): a flow alert,
    phishing-URL or malware-scan finding below it keeps its record but loses its
    techniques (no alert). Returns True when demoted. Never raises."""
    try:
        if sens is None:  # a batch passes the thresholds it loaded once
            sens = _sens(sens_path)
        if not sens:
            return False
        src = str(ev.get("source", ""))
        if src == "detector" and ev.get("tier") == "low" and "flow" in sens:
            # a looser user threshold promotes a "possible" flow alert to a normal one
            if float(ev.get("confidence") or 0) >= max(LOOSEST, sens["flow"]):
                from src.soc.engine import CLASS_TECHNIQUES

                ev.update(techniques=list(CLASS_TECHNIQUES.get(str(ev.get("label")), [])), tier="confirmed",
                          kind="flow_detection", title=f"{ev.get('label')} detected (your looser threshold)")  # fmt: skip
                return False
        if src == "detector" and ev.get("tier") == "confirmed" and "flow" in sens:
            if float(ev.get("confidence") or 0) < sens["flow"]:
                ev.update(techniques=[], tier="low", kind="flow_detection_low", severity=2,
                          title=f"{ev.get('title', 'Flow alert')} (below your stricter threshold)")  # fmt: skip
                return True
        if src in ("phishing", "malware") and src in sens and ev.get("techniques"):
            m = _P.search(str(ev.get("detail", "")))
            if m and float(m.group(1)) < sens[src]:
                ev["techniques"] = []
                ev["detail"] = f"{ev.get('detail', '')} [below your stricter threshold]"
                return True
    except Exception:  # noqa: BLE001
        return False
    return False
