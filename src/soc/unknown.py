"""Logs of a type the SOC has never seen (owner 2026-10-15: "detect + suggest").

1. Detect the structure: JSON, key=value, CSV with a header, or free text.
2. Turn each record into fields and check every value against generic attack
   indicators (encoded PowerShell, credential tools, /etc/shadow, SQL injection,
   path traversal, reverse shells, ...) -> ATT&CK techniques.
3. Score rarity: a line whose *shape* (numbers, IPs, hashes masked) is rare in the
   file is flagged as unusual - not an alert on its own, but worth a look.
4. Suggest a parser: which fields look like the time, source IP, user and message.
   Suggestions are stored for the owner to approve; nothing is adopted
   automatically.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
from collections import Counter
from typing import Any

from utils.paths import PROJECT_ROOT

SUGGESTIONS = PROJECT_ROOT / "artifacts" / "soc" / "parser_suggestions.jsonl"

# value pattern -> (kind, technique); generic, product-independent indicators
INDICATORS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"mimikatz|sekurlsa|lsadump|procdump.{0,40}lsass|comsvcs.{0,20}minidump", re.I),
     "credential_dumping", "T1003"),
    (re.compile(r"/etc/shadow|\bSAM\b.{0,20}\bSYSTEM\b|ntds\.dit", re.I), "credential_store_access", "T1003"),
    (re.compile(r"powershell.{0,60}\s-(e|enc|encodedcommand)\s+[A-Za-z0-9+/=]{20,}", re.I),
     "encoded_powershell", "T1027"),
    (re.compile(r"\b(nc|ncat|netcat)\b.{0,40}\s-[ec]\s|/dev/tcp/|bash\s+-i\s*>", re.I), "reverse_shell", "T1059"),
    (re.compile(r"union(\s|%20|\+)+select|'\s*or\s*'?1'?\s*=\s*'?1|sleep\(\d+\)--", re.I), "sql_injection", "T1190"),
    (re.compile(r"(\.\./|\.\.%2f|%2e%2e/){2,}", re.I), "path_traversal", "T1190"),
    (re.compile(r"<script|javascript:|onerror\s*=", re.I), "xss", "T1190"),
    (re.compile(r"\b(wget|curl)\b.{0,80}(/tmp/|/dev/shm/|\|\s*(ba)?sh)", re.I), "download_and_run", "T1105"),
    (re.compile(r"vssadmin.{0,20}delete\s+shadows|wbadmin.{0,20}delete|bcdedit.{0,40}recoveryenabled\s+no", re.I),
     "recovery_inhibited", "T1490"),
    (re.compile(r"\b(failed password|authentication failure|invalid user|login failed|logon failure)\b", re.I),
     "failed_logon", "T1110"),
    (re.compile(r"\b(nikto|sqlmap|nmap|masscan|hydra|metasploit|cobalt\s*strike|empire)\b", re.I),
     "attack_tool", "T1595"),
    (re.compile(r"\b(audit log|security log|event log)\b.{0,20}\b(cleared|deleted|wiped)\b", re.I),
     "log_cleared", "T1070"),
]  # fmt: skip

_MASKS = [
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
    (re.compile(r"\b[0-9a-f]{16,}\b", re.I), "<hex>"),
    (re.compile(r"\d+"), "<n>"),
]


def detect_structure(text: str) -> str:
    """'json', 'kv', 'csv' or 'text' from the first lines."""
    lines = [ln for ln in text.splitlines()[:50] if ln.strip()]
    if not lines:
        return "text"
    if sum(ln.lstrip().startswith("{") for ln in lines) >= 0.8 * len(lines):
        return "json"
    if sum(len(re.findall(r"\b\w[\w.-]*=\S", ln)) >= 3 for ln in lines) >= 0.8 * len(
        lines
    ):
        return "kv"
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines[:20]), delimiters=",;\t|")
        widths = {len(r) for r in csv.reader(lines[:20], dialect)}
        if len(widths) == 1 and widths.pop() >= 3:
            return "csv"
    except csv.Error:
        pass
    return "text"


def records(text: str, structure: str) -> list[dict[str, Any]]:
    from src.soc.sigma import flatten

    out: list[dict[str, Any]] = []
    if structure == "json":
        for ln in text.splitlines():
            try:
                obj = json.loads(ln)
            except (json.JSONDecodeError, RecursionError):
                continue
            if isinstance(obj, dict):
                out.append(flatten(obj))
    elif structure == "kv":
        for ln in text.splitlines():
            kv = {
                k.lower(): v.strip("\"'")
                for k, v in re.findall(r"(\w[\w.-]*)=(\"[^\"]*\"|'[^']*'|\S+)", ln)
            }
            if kv:
                kv["_raw"] = ln
                out.append(kv)
    elif structure == "csv":
        reader = csv.DictReader(io.StringIO(text))
        out = [
            {str(k).strip().lower(): v for k, v in row.items() if k} for row in reader
        ]
    else:
        out = [{"message": ln} for ln in text.splitlines() if ln.strip()]
    return out


def shape(value: str) -> str:
    for pat, rep in _MASKS:
        value = pat.sub(rep, value)
    return value[:200]


def check(fields: dict[str, Any]) -> tuple[str, list[str]] | None:
    text = " ".join(str(v) for v in fields.values())
    for pat, kind, tech in INDICATORS:
        if pat.search(text):
            return kind, [tech]
    return None


_TIME_KEYS = re.compile(
    r"(^|[._])(time|timestamp|ts|date|@timestamp|eventtime|created)$", re.I
)
_IP_KEYS = re.compile(r"(src|source|client|remote|ip|addr|host)", re.I)
_USER_KEYS = re.compile(r"(user|account|principal|login|actor)", re.I)
_MSG_KEYS = re.compile(r"(msg|message|detail|description|event|action)", re.I)
_IPV4 = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def suggest_parser(
    recs: list[dict[str, Any]], structure: str, name: str
) -> dict[str, Any]:
    """Which field is the time / source IP / user / message (by name and content)."""
    keys = Counter(k for r in recs[:500] for k in r)
    pick: dict[str, str | None] = {
        "time": None,
        "src_ip": None,
        "user": None,
        "message": None,
    }
    for k, _ in keys.most_common():
        vals = [str(r.get(k, "")) for r in recs[:200]]
        if pick["time"] is None and _TIME_KEYS.search(k):
            pick["time"] = k
        elif (
            pick["src_ip"] is None
            and _IP_KEYS.search(k)
            and any(_IPV4.match(v) for v in vals)
        ):
            pick["src_ip"] = k
        elif pick["user"] is None and _USER_KEYS.search(k):
            pick["user"] = k
        elif pick["message"] is None and _MSG_KEYS.search(k):
            pick["message"] = k
    return {"file": name, "structure": structure, "fields": pick,
            "field_names": [k for k, _ in keys.most_common(40)], "records": len(recs),
            "status": "pending owner approval", "ts": time.time()}  # fmt: skip


def parse_unknown(
    text: str, name: str = "", save_suggestion: bool = True
) -> list[dict[str, Any]]:
    """Events for a log of an unknown format, plus a stored parser suggestion."""
    from src.soc.parsers import _event, _ip, _ts, with_entities

    structure = detect_structure(text)
    recs = records(text, structure)
    sug = suggest_parser(recs, structure, name)
    f = sug["fields"]
    shapes = Counter(shape(json.dumps(r, sort_keys=True, default=str)) for r in recs)
    rare_cut = max(1, int(0.001 * len(recs)))
    events = []
    for r in recs:
        hit = check(r)
        msg = str(
            r.get(f["message"] or "", "") or r.get("message", "") or r.get("_raw", "")
        )
        detail = msg[:160] or json.dumps(r, default=str)[:160]
        rare = (
            len(recs) >= 200
            and shapes[shape(json.dumps(r, sort_keys=True, default=str))] <= rare_cut
        )
        ev = _event(
            ts=_ts(r.get(f["time"])) if f["time"] else time.time(),
            src_ip=_ip(r.get(f["src_ip"])) if f["src_ip"] else None,
            source=f"unknown:{structure}",
            kind=hit[0] if hit else ("rare_event" if rare else "log"),
            detail=detail + (" | unusual for this log" if rare and not hit else ""),
            techniques=hit[1] if hit else [],
        )
        events.append(with_entities(ev, user=r.get(f["user"]) if f["user"] else None))
    if save_suggestion and recs:
        SUGGESTIONS.parent.mkdir(parents=True, exist_ok=True)
        with open(SUGGESTIONS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(sug) + "\n")
    return events
