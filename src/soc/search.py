"""Search the stored logs in plain English (owner 2026-10-15).

    "show failed logins for bob yesterday"
    "find powershell -enc"
    "events from host PC1 after 3 pm"
    "alerts from 203.0.113.45 in the last 2 hours"
    "list T1110 events today"

The question is turned into explicit filters (always shown back, so the reader sees
exactly what was searched): entity (IP / user / host / file hash), time window,
event kind, ATT&CK technique or tactic, log source, and literal text. Matching is
deterministic - no model guesses which events "look like" the question.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from datetime import datetime
from typing import Any

Event = dict[str, Any]
MAX_ROWS = 200

_VERB = re.compile(r"^\s*(please\s+)?(show|find|search|list|give me|get|display|look for|grep|filter)\b", re.I)  # fmt: skip
_NOT_SEARCH = re.compile(r"\b(how (to|do|can|should)|what is|what are|explain|why|define|meaning of|quer(y|ies)|sigma|splunk|kql|spl|event ids?|detection|detect it|rules? for)\b", re.I)  # fmt: skip
_CUE = re.compile(
    r"\b(events?|logs?|log ?ins?|log ?ons?|logins?|logons?|alerts?|connections?|"
    r"requests?|processes?|commands?|activity|entries|lines|hits?)\b|\"[^\"]+\"|'[^']+'",
    re.I,
)
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_SHA = re.compile(r"\b[0-9a-fA-F]{64}\b")
_TID = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.I)
_USER = re.compile(
    r"\b(?:user|account|for user|by user|by|for)\s+([A-Za-z][\w.@-]{1,40})\b", re.I
)
_HOST = re.compile(
    r"\b(?:host|computer|machine|server|on|from host)\s+([A-Za-z][\w.-]{1,60})\b", re.I
)
_QUOTED = re.compile(r"\"([^\"]+)\"|'([^']+)'")
_LAST = re.compile(r"\b(?:last|past)\s+(\d+)\s*(minute|min|hour|hr|day|week)s?\b", re.I)
_CLOCK = re.compile(
    r"\b(after|before|since)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.I
)
_STOP = frozenset(
    """show find search list give me get display look for grep filter which what all the
    a an of in on at for from by with to and or me my any events event logs log entries
    lines today yesterday last past hours hour minutes minute days day week weeks after
    before since am pm user account host computer machine server alerts alert please""".split()
)

# plain words -> event kinds produced by the parsers
KINDS = {
    "failed login": "failed_logon", "failed logon": "failed_logon", "failed logins": "failed_logon",
    "failed logons": "failed_logon", "login failure": "failed_logon", "logon failure": "failed_logon",
    "successful login": "successful_logon", "successful logon": "successful_logon",
    "successful logins": "successful_logon", "logins": "successful_logon", "logons": "successful_logon",
    "account lockout": "account_locked_out", "locked out": "account_locked_out",
    "new account": "account_created", "account created": "account_created",
    "service install": "service_installed", "new service": "service_installed",
    "log cleared": "audit_log_cleared", "logs cleared": "audit_log_cleared",
    "process": "process_creation", "processes": "process_creation",
    "web request": "http_request", "web requests": "http_request", "http": "http_request",
    "firewall drop": "fw_drop", "blocked": "fw_drop",
}  # fmt: skip
SOURCES = ("sysmon", "winsec", "suricata", "zeek", "cloudtrail", "auditd", "firewall",
           "syslog", "detector", "phishing", "intel", "malware", "k8s_audit", "osquery")  # fmt: skip
TACTICS = ("reconnaissance", "initial-access", "execution", "persistence", "privilege-escalation",
           "defense-evasion", "credential-access", "discovery", "lateral-movement",
           "collection", "command-and-control", "exfiltration", "impact")  # fmt: skip


_START = re.compile(
    r"^\s*(all\s+)?(events?|logs?|alerts?|failed|successful|log ?ins?|log ?ons?|"
    r"logins?|logons?|processes|connections|requests)\b",
    re.I,
)
_FILTER = re.compile(
    r"\b(today|yesterday|last \d+|past \d+|after \d|before \d|since \d|user \w|host \w)",
    re.I,
)
_CMDLIKE = re.compile(r"\s-\w|\\|\.exe\b|/")


def is_search(q: str) -> bool:
    """A search request (not a how-to / definition question)? Either it starts with
    what to list ("events from host PC1", "failed logins today"), or a search verb
    comes with something concrete to look for (log words, quoted text, an IP, a
    technique id, a hash, a time window, or a command-line-like token)."""
    if _NOT_SEARCH.search(q):
        return False
    if _START.search(q):
        return True
    concrete = (_CUE.search(q) or _IP.search(q) or _TID.search(q) or _SHA.search(q)
                or _FILTER.search(q) or _CMDLIKE.search(q))  # fmt: skip
    return bool(_VERB.search(q) and concrete)


def _day_start(ts: float) -> float:
    d = datetime.fromtimestamp(ts)
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def parse(q: str, now: float | None = None) -> dict[str, Any]:
    """Plain-English question -> explicit filters."""
    now = time.time() if now is None else now
    low = q.lower()
    f: dict[str, Any] = {}
    if ips := _IP.findall(q):
        f["ips"] = sorted(set(ips))
    if shas := _SHA.findall(q):
        f["sha256"] = sorted({s.lower() for s in shas})
    if tids := _TID.findall(q):
        f["techniques"] = sorted({t.upper() for t in tids})
    if tac := [t for t in TACTICS if t.replace("-", " ") in low or t in low]:
        f["tactics"] = tac
    if src := [s for s in SOURCES if re.search(rf"\b{s}\b", low)]:
        f["sources"] = src
    kinds = {k: v for k, v in KINDS.items() if re.search(rf"\b{re.escape(k)}\b", low)}
    # a match inside a longer match does not count ("logins" in "failed logins")
    kinds = {
        k: v for k, v in kinds.items() if not any(k != o and k in o for o in kinds)
    }
    if kinds:
        f["kinds"] = sorted(set(kinds.values()))
    if "alert" in low:
        f["alerts_only"] = True
    if m := _USER.search(q):
        name = m.group(1)
        if (
            name.lower() not in _STOP
            and not _IP.fullmatch(name)
            and not _TID.fullmatch(name)
            and name.lower() not in KINDS
        ):
            f["user"] = name.lower()
    if m := _HOST.search(q):
        name = m.group(1)
        if (
            name.lower() not in _STOP
            and name.lower() != f.get("user")
            and not _IP.fullmatch(name)
        ):
            f["host"] = name.lower()
    # time window
    start, end = None, None
    if "yesterday" in low:
        end = _day_start(now)
        start = end - 86400
    elif "today" in low:
        start = _day_start(now)
    if m := _LAST.search(q):
        n, unit = int(m.group(1)), m.group(2).lower()
        secs = {
            "minute": 60,
            "min": 60,
            "hour": 3600,
            "hr": 3600,
            "day": 86400,
            "week": 604800,
        }[unit]
        # pentest 2026-10-17: "last 999999999 hours" overflowed the date conversion
        start = max(0.0, now - min(n * secs, 100 * 365 * 86400))
    for m in _CLOCK.finditer(q):
        hh, mm, ap = int(m.group(2)), int(m.group(3) or 0), (m.group(4) or "").lower()
        if ap == "pm" and hh < 12:
            hh += 12
        if ap == "am" and hh == 12:
            hh = 0
        base = _day_start(start if start is not None else now)
        t = base + hh * 3600 + mm * 60
        if m.group(1).lower() in ("after", "since"):
            start = t
        else:
            end = t
    if start is not None:
        f["start"] = start
    if end is not None:
        f["end"] = end
    # literal text: quoted phrases, else command-line-looking leftovers ("powershell -enc")
    texts = [a or b for a, b in _QUOTED.findall(q)]
    if not texts:
        rest = _IP.sub(" ", _TID.sub(" ", q))
        for v in (f.get("user"), f.get("host")):
            if v:
                rest = re.sub(rf"\b{re.escape(v)}\b", " ", rest, flags=re.I)
        for k in KINDS:
            rest = re.sub(rf"\b{re.escape(k)}\b", " ", rest, flags=re.I)
        words = [w for w in re.findall(r"[\w.:/\\-]+", rest)
                 if w.lower() not in _STOP and w.lower() not in SOURCES and not w.isdigit()
                 and w.lower() not in TACTICS]  # fmt: skip
        texts = words  # unquoted: every word must appear (quoted text: exact phrase)
    if texts:
        f["text"] = [t.lower() for t in texts]
    return f


def _match(ev: Event, f: dict[str, Any]) -> bool:
    ts = float(ev.get("ts") or 0)
    if "start" in f and ts < f["start"]:
        return False
    if "end" in f and ts >= f["end"]:
        return False
    if "ips" in f and not ({ev.get("src_ip"), ev.get("dst_ip")} & set(f["ips"])):
        return False
    if "user" in f and str(ev.get("user", "")).lower() != f["user"]:
        return False
    if "host" in f and f["host"] not in str(ev.get("host", "")).lower():
        return False
    if "sha256" in f and str(ev.get("sha256", "")).lower() not in f["sha256"]:
        return False
    techs = [str(t).upper() for t in ev.get("techniques", [])]
    if f.get("alerts_only") and not techs:
        return False
    if "techniques" in f and not any(
        t.split(".")[0] in {x.split(".")[0] for x in f["techniques"]} for t in techs
    ):
        return False
    if "tactics" in f:
        from src.soc.correlate import technique_info

        if not {technique_info(t)[1] for t in techs} & set(f["tactics"]):
            return False
    if "sources" in f and not any(
        s in str(ev.get("source", "")).lower() for s in f["sources"]
    ):
        return False
    if "kinds" in f and ev.get("kind") not in f["kinds"]:
        return False
    if "text" in f:
        hay = (
            f"{ev.get('detail', '')} {ev.get('kind', '')} {ev.get('file', '')}".lower()
        )
        if not all(t in hay for t in f["text"]):
            return False
    return True


def describe(f: dict[str, Any]) -> str:
    """The filters in words (shown above the results)."""
    parts = []
    for k, label in (("kinds", "kind"), ("techniques", "technique"), ("tactics", "tactic"),
                     ("sources", "source"), ("ips", "IP")):  # fmt: skip
        if k in f:
            parts.append(f"{label} {', '.join(x.replace('_', ' ') for x in f[k])}")
    for k in ("user", "host"):
        if k in f:
            parts.append(f"{k} {f[k]}")
    if "sha256" in f:
        parts.append("file hash " + ", ".join(h[:12] + "…" for h in f["sha256"]))
    if f.get("alerts_only"):
        parts.append("alerts only")
    if "text" in f:
        parts.append("text " + " + ".join(f'"{t}"' for t in f["text"]))
    fmt = "%Y-%m-%d %H:%M"
    if "start" in f or "end" in f:
        a = (
            time.strftime(fmt, time.localtime(f["start"]))
            if "start" in f
            else "the beginning"
        )
        b = time.strftime(fmt, time.localtime(f["end"])) if "end" in f else "now"
        parts.append(f"from {a} to {b}")
    return " · ".join(parts) or "everything (no filter recognised)"


def run(q: str, events: list[Event], now: float | None = None) -> dict[str, Any]:
    f = parse(q, now)
    hits = [e for e in events if _match(e, f)]
    hits.sort(key=lambda e: -float(e.get("ts") or 0))
    return result(f, hits, len(hits))


def run_store(q: str, now: float | None = None, exact: bool = False) -> dict[str, Any]:
    """Search the SOC's own store through its index (src/soc/eventindex.py): the same
    matches as ``run(q, all stored events)`` in milliseconds at millions of events;
    above 10,000 matches the count is shown as "10,000+" unless ``exact``."""
    from src.soc import eventindex

    return eventindex.search(q, now, exact=exact)


def result(
    f: dict[str, Any],
    hits: list[Event],
    count: int,
    by_source: dict[str, int] | None = None,
    by_kind: dict[str, int] | None = None,
) -> dict[str, Any]:
    """The answer for filters ``f`` and its matching events (newest first); the index
    passes the per-source / per-kind counts and only the rows it shows."""
    if by_source is None:
        by_source = dict(
            Counter(str(e.get("source", "?")) for e in hits).most_common(8)
        )
    if by_kind is None:
        by_kind = dict(Counter(str(e.get("kind", "?")) for e in hits).most_common(8))
    return {
        "filters": f,
        "understood": describe(f),
        "count": count,
        "capped": False,
        "sample": len(hits),
        "by_source": by_source,
        "by_kind": by_kind,
        "rows": [
            {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(e.get("ts") or 0))),
             "source": e.get("source"), "kind": e.get("kind"), "ip": e.get("src_ip"),
             "user": e.get("user"), "host": e.get("host"),
             "techniques": ", ".join(e.get("techniques", [])), "detail": str(e.get("detail", ""))[:200]}
            for e in hits[:MAX_ROWS]
        ],
    }  # fmt: skip


def answer_text(res: dict[str, Any]) -> str:
    """Answer-first text for the assistant."""
    n = res["count"]
    big = res.get("capped", False)
    head = (f"Found {n:,}{'+' if big else ''} matching event{'s' if n != 1 else ''}." if n
            else "No stored events match that search.")  # fmt: skip
    lines = [head, f"🔎 **Searched for:** {res['understood']}."]
    if n:
        from src.soc.safe import md  # log fields are untrusted Markdown

        src = ", ".join(f"{md(k)} ({v})" for k, v in res["by_source"].items())
        among = f" (among the newest {n:,})" if big else ""
        lines.append(f"**By log source{among}:** {src}.")
        lines.append("**Newest first:**")
        for r in res["rows"][:10]:
            who = " · ".join(md(x) for x in (r["ip"], r["user"], r["host"]) if x)
            lines.append(f"- {r['time']} · {md(r['source'])} · {md(r['kind'])}"
                         + (f" · {who}" if who else "") + (f" · {md(r['techniques'])}" if r["techniques"] else "")
                         + f" — {md(r['detail'], 120)}")  # fmt: skip
        if n > 10:
            more = f"{n - 10:,}{'+' if big else ''}"
            lines.append(
                f"…and {more} more (the full table is shown below the answer)."
            )
            if big:
                lines.append("Use **Count exactly** for the precise number.")
    else:
        lines.append("Try a wider time window, or fewer words.")
    return "\n".join(lines)
