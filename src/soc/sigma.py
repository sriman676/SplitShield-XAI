"""Sigma rule engine for Windows events (owner 2026-10-14).

Evaluates the SigmaHQ community rules (third_party/sigma, release r2026-07-01, the
official "core+" bundle: stable/test rules of level medium and above; Detection
Rule License 1.1, authors credited in every rule file) directly on parsed events,
without a SIEM.

Fair-test exclusions: rules written by the authors of the two attack-log test
collections (Samir Bousseaden: EVTX-ATTACK-SAMPLES; Roberto Rodriguez: OTRF) or
citing those collections are skipped, so the reported accuracy is not measured on
data the rules were built from.

Supported Sigma features: selections (maps, lists of maps, keyword lists), the
modifiers contains / startswith / endswith / all / re / windash / cidr /
base64offset / cased / exists / fieldref, wildcards, null, and conditions with
and / or / not / parentheses / "1 of x*" / "all of x*" / "of them". Rules using
anything else are skipped and counted.
"""

from __future__ import annotations

import base64
import fnmatch
import ipaddress
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import yaml

from utils.logging_config import get_logger
from utils.paths import PROJECT_ROOT

_log = get_logger(__name__)
# another rules folder / levels for a whole run, worker processes included (comparisons,
# owner 2026-10-24): SPLITSHIELD_SIGMA_ROOT (must hold windows/ ...), SPLITSHIELD_SIGMA_LEVELS
RULES_ROOT = Path(os.environ.get("SPLITSHIELD_SIGMA_ROOT") or PROJECT_ROOT / "third_party" / "sigma" / "rules" / "rules")
RULES_DIR = RULES_ROOT / "windows"
EXCLUDE_AUTHORS = ("bousseaden", "roberto rodriguez", "cyb3rward0g")
# also the other held-out test sets (Splunk attack_data, AIT-LDS, IoT-23, Invictus)
EXCLUDE_REFS = (
    "attack_data",
    "ait-lds",
    "aecid",
    "iot-23",
    "invictus-ir",
    "flaws.cloud",
    "evtx-attack-samples",
    "sbousseaden",
    "security-datasets",
    "otrf",
    "mordor",
)
LEVELS: tuple[str, ...] = tuple(os.environ.get("SPLITSHIELD_SIGMA_LEVELS", "medium high critical").split())
_TACTIC_COMPAT = {"stealth": "defense-evasion", "defense-impairment": "defense-evasion"}

Fields = dict[str, Any]  # lower-case field name -> value
Pred = Callable[[Fields], bool]

# Sigma category -> (source, event id) pairs it covers
CATEGORIES: dict[str, set[tuple[str, int]]] = {
    "process_creation": {("sysmon", 1), ("security", 4688)},
    "network_connection": {("sysmon", 3)},
    "process_access": {("sysmon", 10)},
    "file_event": {("sysmon", 11)},
    "registry_add": {("sysmon", 12)},
    "registry_delete": {("sysmon", 12)},
    "registry_set": {("sysmon", 13)},
    "registry_rename": {("sysmon", 14)},
    "registry_event": {("sysmon", 12), ("sysmon", 13), ("sysmon", 14)},
    "create_remote_thread": {("sysmon", 8)},
    "image_load": {("sysmon", 7)},
    "pipe_created": {("sysmon", 17), ("sysmon", 18)},
    "wmi_event": {("sysmon", 19), ("sysmon", 20), ("sysmon", 21)},
    "driver_load": {("sysmon", 6)},
    "dns_query": {("sysmon", 22)},
    "create_stream_hash": {("sysmon", 15)},
    "process_tampering": {("sysmon", 25)},
    "file_delete": {("sysmon", 23), ("sysmon", 26)},
    "ps_script": {("powershell", 4104)},
    "ps_module": {("powershell", 4103)},
    "ps_classic_start": {("powershell", 400)},
}
# Sigma service -> text expected in the event's provider / channel
SERVICES = {
    "security": "security",
    "system": "system",
    "application": "application",
    "sysmon": "sysmon",
    "windefend": "windows defender",
    "wmi": "wmi-activity",
    "firewall-as": "firewall",
    "codeintegrity-operational": "codeintegrity",
}


class Unsupported(ValueError):
    """A rule uses a Sigma feature this engine does not implement."""


@dataclass(frozen=True)
class Rule:
    title: str
    level: str
    techniques: tuple[str, ...]
    tactics: tuple[str, ...]
    keys: frozenset[tuple[str, int]] | None  # Windows category rules
    service: str | None
    match: Pred
    # non-Windows rules: (product, category, service), None = any
    source: tuple[str | None, str | None, str | None] | None = None
    rule_id: str = ""  # the rule's own id (or its file name): users switch rules by it


# --- values -------------------------------------------------------------------------


def _parts(v: str) -> list[str]:
    """Sigma value -> literal chunks and wildcard tokens ('*' / '?' as their own
    items; escaped \\*, \\? and \\\\ become literal text)."""
    out: list[str] = []
    lit, i = "", 0
    while i < len(v):
        c = v[i]
        if c == "\\" and i + 1 < len(v) and v[i + 1] in "*?\\":
            lit += v[i + 1]
            i += 2
            continue
        if c in "*?":
            out += [lit, c] if lit else [c]
            lit = ""
        else:
            lit += c
        i += 1
    return [*out, lit] if lit else out


def _compile_value(v: str, mode: str, cased: bool) -> Callable[[str], bool]:
    """Test for one value; ``mode`` is '', 'contains', 'startswith' or 'endswith'."""
    parts = _parts(v)
    if all(p not in ("*", "?") for p in parts):  # plain text: fast string ops
        lit = "".join(parts) if cased else "".join(parts).lower()
        op = {"contains": _has, "startswith": _starts, "endswith": _ends}.get(mode, _eq)
        return partial(op, lit) if cased else partial(_lower, partial(op, lit))
    body = "".join(
        ".*" if p == "*" else "." if p == "?" else re.escape(p) for p in parts
    )
    if mode in ("contains", "endswith"):
        body = ".*" + body
    if mode in ("contains", "startswith"):
        body += ".*"
    return partial(_full, re.compile(body, re.S if cased else re.S | re.I))


def _has(lit: str, x: str) -> bool:
    return lit in x


def _starts(lit: str, x: str) -> bool:
    return x.startswith(lit)


def _ends(lit: str, x: str) -> bool:
    return x.endswith(lit)


def _lower(test: Callable[[str], bool], x: str) -> bool:
    return test(x.lower())


_DASHES = ("-", "/", "\u2013", "\u2014", "\u2015")


def _windash(v: str) -> list[str]:
    """Variants with every word-leading dash as -, /, en/em/horizontal-bar dash."""
    return list(dict.fromkeys(re.sub(r"(?<!\w)-", d, v) for d in _DASHES))


def _b64offsets(v: str) -> list[str]:
    raw = v.encode()
    out = []
    for i in range(3):
        enc = base64.b64encode(b" " * i + raw).decode()
        start = (0, 2, 3)[i]
        end = len(enc) - (0, 3, 2)[(len(raw) + i) % 3]
        out.append(enc[start:end])
    return out


def _value_pred(field: str, mods: list[str], values: list[Any]) -> Pred:
    f = field.lower()
    known = {"contains", "startswith", "endswith", "all", "re", "windash", "cidr",
             "base64offset", "cased", "exists", "fieldref", "i", "m", "s"}  # fmt: skip
    if set(mods) - known:
        raise Unsupported(f"modifier {set(mods) - known}")
    cased = "cased" in mods
    if "exists" in mods:
        want = bool(values[0])
        return lambda e: (f in e and e[f] not in (None, "")) == want
    if "fieldref" in mods:
        refs = [str(v).lower() for v in values]
        return lambda e: any(
            str(e.get(f, "")).lower() == str(e.get(r, "")).lower() and f in e
            for r in refs
        )
    if "windash" in mods:
        values = [x for v in values for x in _windash(str(v))]
    if "base64offset" in mods:
        values = [x for v in values for x in _b64offsets(str(v))]
    tests: list[Callable[[str], bool]] = []
    for v in values:
        if v is None:
            tests.append(lambda s: s == "")
            continue
        s = str(v)
        if "re" in mods:
            flags = (
                (re.I if "i" in mods else 0)
                | (re.M if "m" in mods else 0)
                | (re.S if "s" in mods else 0)
            )
            rx = re.compile(s, flags)
            tests.append(partial(_search, rx))
        elif "cidr" in mods:
            net = ipaddress.ip_network(s, strict=False)
            tests.append(partial(_in_net_t, net))
        else:
            mode = next(
                (m for m in ("contains", "startswith", "endswith") if m in mods), ""
            )
            tests.append(_compile_value(s, mode, cased))
    if _REFERENCE:  # original, unoptimised semantics (equivalence test only)
        combine = all if "all" in mods else any

        def ref_pred(e: Fields) -> bool:
            if f not in e or e[f] is None:
                return any(v is None for v in values) and "all" not in mods
            x = str(e[f])
            return combine(t(x) for t in tests)

        return ref_pred
    # case-insensitive plain-text tests are _lower(op): unwrap them so the value is
    # lower-cased once per event instead of once per test (same results)
    plan = [
        (t.args[0], True) if isinstance(t, partial) and t.func is _lower else (t, False)
        for t in tests
    ]
    need_lower = any(low for _, low in plan)
    use_all = "all" in mods
    none_ok = any(v is None for v in values) and not use_all

    def pred(e: Fields) -> bool:
        if f not in e or e[f] is None:
            return none_ok
        x = str(e[f])
        xl = x.lower() if need_lower else x
        if use_all:
            for fn, low in plan:
                if not fn(xl if low else x):
                    return False
            return True
        for fn, low in plan:
            if fn(xl if low else x):
                return True
        return False

    return pred


def _all_of(preds: tuple[Pred, ...], e: Fields) -> bool:
    for p in preds:
        if not p(e):
            return False
    return True


def _any_of(preds: tuple[Pred, ...], e: Fields) -> bool:
    for p in preds:
        if p(e):
            return True
    return False


def _search(rx: re.Pattern[str], x: str) -> bool:
    return rx.search(x) is not None


def _full(rx: re.Pattern[str], x: str) -> bool:
    return rx.fullmatch(x) is not None


def _eq(lit: str, x: str) -> bool:
    return x == lit


def _in_net_t(net: Any, x: str) -> bool:
    return _in_net(x, net)


def _in_net(x: str, net: Any) -> bool:
    try:
        return ipaddress.ip_address(x) in net
    except ValueError:
        return False


def _selection(sel: Any) -> Pred:
    if isinstance(sel, dict):
        preds = []
        for key, val in sel.items():
            field, *mods = str(key).split("|")
            vals = val if isinstance(val, list) else [val]
            preds.append(_value_pred(field, mods, vals))
        if _REFERENCE:
            return lambda e: all(p(e) for p in preds)
        return partial(_all_of, tuple(preds))
    if isinstance(sel, list):
        if all(isinstance(x, dict) for x in sel):
            subs = [_selection(x) for x in sel]
            if _REFERENCE:
                return lambda e: any(p(e) for p in subs)
            return partial(_any_of, tuple(subs))
        words = [str(x).lower() for x in sel]  # keyword search over every value
        if _REFERENCE or any(_SEP in w for w in words):  # naive search
            return lambda e: any(w in str(v).lower() for v in e.values() for w in words)
        return partial(_any_keyword, tuple(words))
    raise Unsupported("selection type")


# Keyword selections search every value of the event. All rules are evaluated against
# the same event dict, so its lower-cased values are joined ONCE per event (keyed by
# identity; the reference held here keeps the id from being reused) instead of once
# per rule and keyword. A keyword without the separator matches the joined text iff
# it matches one of the values, so results are identical.
_SEP = "\x00"
_BLOB: list[Any] = [None, ""]
# True -> rules compile with the original, unoptimised predicates. Only the
# equivalence test sets it (tests/test_sigma.py: both compilations must agree).
_REFERENCE = False


def _blob(e: Fields) -> str:
    if _BLOB[0] is not e:
        _BLOB[0] = e
        _BLOB[1] = _SEP.join(str(v).lower() for v in e.values())
    text: str = _BLOB[1]
    return text


def _any_keyword(words: tuple[str, ...], e: Fields) -> bool:
    if not e:
        return False
    text = _blob(e)
    for w in words:  # plain loop: noticeably cheaper than any(<genexpr>) here
        if w in text:
            return True
    return False


# --- conditions -----------------------------------------------------------------------


def _condition(text: str, sels: dict[str, Pred]) -> Pred:
    toks = re.findall(r"\(|\)|[^\s()]+", text)
    pos = 0

    def peek() -> str | None:
        return toks[pos].lower() if pos < len(toks) else None

    def take() -> str:
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def group(pattern: str) -> list[Pred]:
        names = (
            list(sels)
            if pattern == "them"
            else [n for n in sels if fnmatch.fnmatchcase(n, pattern)]
        )
        if not names:
            raise Unsupported(f"no selection matches {pattern}")
        return [sels[n] for n in names]

    def atom() -> Pred:
        t = take()
        lt = t.lower()
        if lt == "(":
            p = expr()
            if take() != ")":
                raise Unsupported("parenthesis")
            return p
        if lt == "not":
            inner = atom()
            return lambda e: not inner(e)
        if peek() == "of":
            take()
            ps = group(take())
            if lt == "all":
                return lambda e: all(p(e) for p in ps)
            if lt == "1" or lt == "any":
                return lambda e: any(p(e) for p in ps)
            raise Unsupported(f"{t} of")
        if t not in sels:
            raise Unsupported(f"unknown selection {t}")
        return sels[t]

    def conj() -> Pred:
        p = atom()
        while peek() == "and":
            take()
            q = atom()
            p = (lambda a, b: lambda e: a(e) and b(e))(p, q)
        return p

    def expr() -> Pred:
        p = conj()
        while peek() == "or":
            take()
            q = conj()
            p = (lambda a, b: lambda e: a(e) or b(e))(p, q)
        return p

    out = expr()
    if pos != len(toks):
        raise Unsupported("trailing condition tokens")
    return out


# --- loading ------------------------------------------------------------------------


def _excluded(doc: dict[str, Any]) -> bool:
    author = str(doc.get("author", "")).lower()
    refs = " ".join(str(r) for r in doc.get("references") or []).lower()
    return any(a in author for a in EXCLUDE_AUTHORS) or any(
        r in refs for r in EXCLUDE_REFS
    )


def compile_rule(doc: dict[str, Any]) -> Rule:
    det = dict(doc["detection"])
    cond = det.pop("condition")
    if isinstance(cond, list):
        cond = " or ".join(f"({c})" for c in cond)
    if "|" in str(cond) or "timeframe" in det:
        raise Unsupported("aggregation")
    sels = {k: _selection(v) for k, v in det.items()}
    ls = doc.get("logsource") or {}
    cat, svc = ls.get("category"), ls.get("service")
    product = ls.get("product")
    keys = None
    source = None
    if product != "windows" and (product or cat in GENERIC_CATEGORIES):
        source = (product, cat, svc)
        cat = svc = None
    elif cat:
        if cat not in CATEGORIES:
            raise Unsupported(f"category {cat}")
        keys = frozenset(CATEGORIES[cat])
    elif svc:
        svc = SERVICES.get(svc, svc)
    elif source is None:
        raise Unsupported("no logsource")
    tags = [str(t).lower() for t in doc.get("tags") or []]
    techs = sorted(
        {m.group(1).upper() for t in tags if (m := re.match(r"attack\.(t\d{4})", t))}
    )
    tactics = sorted(
        {_TACTIC_COMPAT.get(t[7:].replace("_", "-"), t[7:].replace("_", "-"))
         for t in tags if t.startswith("attack.") and not re.match(r"attack\.(t\d|g\d|s\d)", t)}
    )  # fmt: skip
    return Rule(
        title=str(doc.get("title", "")),
        level=str(doc.get("level", "")),
        techniques=tuple(techs),
        tactics=tuple(tactics),
        keys=keys,
        service=None if (cat or source) else svc,
        match=_condition(str(cond), sels),
        source=source,
        rule_id=str(doc.get("id") or ""),
    )


@lru_cache(maxsize=1)
def load(rules_dir: str = str(RULES_ROOT)) -> dict[str, Any]:
    """Compile every usable rule once per process; returns the index and stats."""
    by_key: dict[tuple[str, int], list[Rule]] = {}
    by_service: dict[str, list[Rule]] = {}
    by_source: dict[tuple[str | None, str | None, str | None], list[Rule]] = {}
    stats: dict[str, Any] = {
        "files": 0,
        "loaded": 0,
        "excluded_fair_test": 0,
        "unsupported": 0,
        "low_level": 0,
    }
    every: list[Rule] = []
    for p in sorted(Path(rules_dir).rglob("*.yml")):
        stats["files"] += 1
        try:
            doc = yaml.safe_load(p.read_text("utf-8"))
        except yaml.YAMLError:
            stats["unsupported"] += 1
            continue
        if not isinstance(doc, dict) or "detection" not in doc:
            continue
        if _excluded(doc):
            stats["excluded_fair_test"] += 1
            continue
        if str(doc.get("level", "")).lower() not in LEVELS:
            stats["low_level"] += 1
            continue
        try:
            rule = compile_rule(doc)
        except (Unsupported, re.error, ValueError, KeyError, TypeError):
            stats["unsupported"] += 1
            continue
        if not rule.rule_id:
            from dataclasses import replace as _replace

            rule = _replace(rule, rule_id=p.stem)
        every.append(rule)
        stats["loaded"] += 1
        if rule.source is not None:
            by_source.setdefault(rule.source, []).append(rule)
        elif rule.keys is not None:
            for k in rule.keys:
                by_key.setdefault(k, []).append(rule)
        else:
            by_service.setdefault(rule.service or "", []).append(rule)
    _log.info("sigma: %s", stats)
    stats["by_product"] = dict(
        sorted(
            {
                str(k[0] or k[1]): sum(
                    len(v)
                    for kk, v in by_source.items()
                    if (kk[0] or kk[1]) == (k[0] or k[1])
                )
                for k in by_source
            }.items()
        )
    )
    return {
        "by_key": by_key,
        "by_service": by_service,
        "all": every,
        "by_source": by_source,
        "stats": stats,
    }


_DIRS: dict[str, bool] = {}


def _is_dir(p: Path) -> bool:
    """Rules folder present? Checked once per path and process (was a disk stat per
    event); keyed by path, so a swapped rules folder (update agent) is re-checked."""
    key = str(p)
    if key not in _DIRS:
        _DIRS[key] = p.is_dir()
    return _DIRS[key]


def source_of(provider: str, channel: str, is_sysmon: bool) -> str:
    if is_sysmon:
        return "sysmon"
    p = f"{provider} {channel}".lower()
    if "security-auditing" in p or channel.lower() == "security":
        return "security"
    if "powershell" in p:
        return "powershell"
    return "other"


def match(
    fields: dict[str, Any], eid: int, provider: str, channel: str, is_sysmon: bool
) -> list[Rule]:
    """Rules that fire on one event (``fields``: event data by field name)."""
    if not _is_dir(RULES_DIR):
        return []
    idx = load()
    e = {str(k).lower(): v for k, v in fields.items()}
    e.setdefault("eventid", eid)
    _BLOB[0] = None  # new event: drop the joined-values cache
    src = source_of(provider, channel, is_sysmon)
    cands = list(idx["by_key"].get((src, eid), []))
    where = f"{provider} {channel}".lower()
    for svc, rules in idx["by_service"].items():
        if svc and svc in where:
            cands += rules
    out = []
    for r in cands:
        try:
            if r.match(e):
                out.append(r)
        except (TypeError, ValueError):
            continue
    return out


# non-Windows logsource categories that rules use without a product
GENERIC_CATEGORIES = {"webserver", "proxy", "dns", "antivirus", "database", "firewall"}


def flatten(
    obj: Any, prefix: str = "", out: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Nested JSON -> {"useridentity.arn": ...} (lower-case dotted keys, as rules use)."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}{str(k).lower()}.", out)
    elif (
        isinstance(obj, list)
        and obj
        and all(not isinstance(x, dict | list) for x in obj)
    ):
        out[prefix[:-1]] = " ".join(str(x) for x in obj)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:20]):
            flatten(v, f"{prefix}{i}.", out)
    elif prefix:
        out[prefix[:-1]] = obj
    return out


def match_source(
    fields: dict[str, Any],
    product: str | None,
    category: str | None,
    service: str | None,
) -> list[Rule]:
    """Rules for a non-Windows event: every rule whose logsource parts are unset or
    equal to the event's (product, category, service)."""
    if not _is_dir(RULES_ROOT):
        return []
    idx = load()
    e = {str(k).lower(): v for k, v in fields.items()}
    _BLOB[0] = None  # new event: drop the joined-values cache
    out = []
    for (rp, rc, rs), rules in idx["by_source"].items():
        if (rp and rp != product) or (rc and rc != category) or (rs and rs != service):
            continue
        for r in rules:
            try:
                if r.match(e):
                    out.append(r)
            except (TypeError, ValueError):
                continue
    return out
