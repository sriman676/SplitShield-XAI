"""Log parsers -> one normalized event shape the correlator understands.

Every parser yields plain dicts:

    {"ts": epoch seconds, "src_ip": str|None, "dst_ip": str|None, "dst_port": int|None,
     "source": "suricata"|"zeek"|"firewall"|"winsec"|"syslog"|"detector",
     "kind": short event type, "detail": human text, "techniques": [ATT&CK ids]}

Supported inputs (auto-detected by ``parse_file`` / ``parse_line``):
  * Suricata ``eve.json`` (JSON lines, ``event_type: alert``)
  * Zeek ``conn.log`` (TSV with a ``#fields`` header)
  * Windows Firewall ``pfirewall.log`` (W3C with a ``#Fields`` header)
  * Windows Security events exported as JSON / JSON lines (``Get-WinEvent | ConvertTo-Json``),
    or pulled live with ``read_windows_security`` (needs Administrator)
  * Generic syslog / auth.log lines (sshd, sudo, CEF passthrough)

Lines that carry no security meaning still become events (``techniques: []``) because
the correlator's scan heuristic counts plain connections and firewall drops.
"""

from __future__ import annotations

import ipaddress
import json
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

Event = dict[str, Any]

_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# Windows Security event id -> (kind, technique ids). 4624/4648 are only suspicious
# in context (after failures; 4648 fires constantly for runas, scheduled tasks and
# RDP), so the correlator decides; they carry no technique on their own.
WINSEC: dict[int, tuple[str, list[str]]] = {
    4625: ("failed_logon", ["T1110"]),
    4771: ("failed_logon", ["T1110"]),  # Kerberos pre-authentication failed
    4740: ("account_locked_out", ["T1110"]),
    4624: ("successful_logon", []),
    4648: ("explicit_credential_logon", []),
    4688: ("process_creation", []),
    4720: ("account_created", ["T1136"]),
    4728: ("added_to_privileged_group", ["T1098"]),
    4732: ("added_to_privileged_group", ["T1098"]),
    4756: ("added_to_privileged_group", ["T1098"]),
    4697: ("service_installed", ["T1543"]),
    7045: ("service_installed", ["T1543"]),
    1102: ("audit_log_cleared", ["T1070"]),
    4698: ("scheduled_task_created", ["T1053"]),
    4719: ("audit_policy_changed", ["T1562"]),
    4672: ("special_privileges_logon", []),  # every admin logon: context only
    4104: ("powershell_script", []),  # PowerShell/Operational; content decides
}

# Sysmon (Microsoft-Windows-Sysmon/Operational) event id -> (kind, techniques);
# content rules refine 1 (process), 10 (process access), 11-13 (autostart).
SYSMON: dict[int, tuple[str, list[str]]] = {
    1: ("process_creation", []),
    3: ("network_connection", []),
    8: ("remote_thread_created", ["T1055"]),
    10: ("process_access", []),
    11: ("file_created", []),
    12: ("registry_change", []),
    13: ("registry_change", []),
    17: ("pipe_created", []),
    18: ("pipe_connected", []),
    19: ("wmi_filter", ["T1546"]),
    20: ("wmi_consumer", ["T1546"]),
    21: ("wmi_binding", ["T1546"]),
    22: ("dns_query", []),
    25: ("process_tampering", ["T1055"]),
}

# PowerShell script blocks (4104) and command lines (4688 / Sysmon 1) -> technique,
# most specific first; generic admin use does not match.
_SCRIPT_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"mimikatz|sekurlsa|lsadump|procdump.*lsass|"
                r"comsvcs(\.dll)?.*minidump", re.I), "credential_dumping", "T1003"),
    (re.compile(r"amsiutils|amsiinitfailed|set-mppreference.*-disable|"
                r"disablerealtimemonitoring", re.I), "defense_tampering", "T1562"),
    (re.compile(r"psexec|paexec|wmic\s+/node:|invoke-command\s+-computername|"
                r"enter-pssession|winrs\s", re.I), "remote_execution", "T1021"),
    (re.compile(r"(iex|invoke-expression).*(downloadstring|net\.webclient|"
                r"invoke-webrequest)|downloadstring\(", re.I), "download_cradle", "T1105"),
    (re.compile(r"\s-e(nc(odedcommand)?)?\s+[a-z0-9+/=]{40,}|frombase64string", re.I),
     "obfuscated_script", "T1027"),
]  # fmt: skip
_RUN_KEY = re.compile(
    r"currentversion\\(run|runonce)\b|start menu\\programs\\startup", re.I
)

# 4688 command lines -> (kind, technique); one kind, the technique says which. Each pattern is specific enough not to
# match routine use (a bare "rundll32 x.dll,Entry" is normal; with a URL it is not).
_SUSPICIOUS_CMDS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"powershell.*\s-e(nc(odedcommand)?)?\s|frombase64string", re.I),
     "suspicious_process", "T1059"),
    (re.compile(r"certutil.*-(urlcache|verifyctl)|bitsadmin.*/transfer", re.I),
     "suspicious_process", "T1105"),
    (re.compile(r"mshta(\.exe)?\s+[\"']?(https?:|javascript:|vbscript:)|"
                r"regsvr32.*/i:\s*https?:|rundll32.*(javascript:|https?:)", re.I),
     "suspicious_process", "T1218"),
    (re.compile(r"vssadmin.*delete\s+shadows|wmic.*shadowcopy.*delete|"
                r"bcdedit.*recoveryenabled\s+no", re.I),
     "suspicious_process", "T1490"),
]  # fmt: skip


def _first(rules: list[tuple[str, str | None]], text: str) -> list[str] | None:
    for pat, tech in rules:
        if re.search(pat, text, re.I):
            return [tech] if tech else []
    return None


# IDS signature -> technique, most specific evidence first (see signature_techniques).
_SIG_PRECISE: list[tuple[str, str | None]] = [
    (r"brute.?forc|password guess|dictionary attack|excessive (login|auth)|"
     r"multiple (failed|unsuccessful)|failed (login|logon|auth)", "T1110"),
    (r"exfil|data.?leak|dns.?tunnel", "T1048"),
    (r"heartbleed|log4j|log4shell|shellshock|struts|sql.?injection|union select|"
     r"\bxss\b|cross.site.scripting|directory traversal|path traversal|"
     r"command injection|remote code execution|\brce\b", "T1190"),
]  # fmt: skip
# Emerging Threats / GPL rule-set prefixes; None = informational (no technique)
_SIG_PREFIX: list[tuple[str, str | None]] = [
    (r"^(et|gpl) scan\b", "T1046"),
    (r"^(et|gpl) (web_server|web_specific_apps|exploit|exploit_kit)\b", "T1190"),
    (r"^(et|gpl) dos\b", "T1498"),
    (r"^et (trojan|malware|mobile_malware|cnc|botcc|worm)\b", "T1071"),
    (
        r"^(et|gpl) (policy|info|hunting|games|chat|p2p|user_agents|inappropriate)\b",
        None,
    ),
    (r"^suricata\b", None),  # engine protocol-anomaly / decoder events
]
# Suricata classtype (alert.category)
_SIG_CLASS: list[tuple[str, str | None]] = [
    (r"network scan", "T1046"),
    (r"web application attack", "T1190"),
    (r"denial of service", "T1498"),
    (r"trojan|command and control|malware", "T1071"),
    (r"privilege gain", "T1190"),
    (
        r"misc activity|not suspicious|privacy violation|protocol command decode|"
        r"unknown traffic",
        None,
    ),
]
_SIG_KEYWORD: list[tuple[str, str | None]] = [
    (r"\bscan|nmap|masscan|zmap", "T1046"),
    (r"\bd?dos\b|flood|amplification", "T1498"),
    (r"trojan|\bc2\b|\bcnc\b|command.and.control|beacon|botnet", "T1071"),
    (r"exploit|cve-\d|overflow", "T1190"),
]


def signature_techniques(signature: str, category: str = "") -> list[str]:
    """ATT&CK technique(s) for an IDS signature: precise keywords, then the rule-set
    category prefix, then the classtype, then generic keywords. Informational
    rule sets (ET POLICY/INFO, engine anomalies) map to no technique, so a
    "Login Successful" policy notice is never reported as brute force."""
    sig = signature.strip()
    for rules, text in (
        (_SIG_PRECISE, sig), (_SIG_PREFIX, sig), (_SIG_CLASS, category),
        (_SIG_KEYWORD, sig),
    ):  # fmt: skip
        hit = _first(rules, text) if text else None
        if hit is not None:
            return hit
    return []


# Syslog / auth.log keywords -> (kind, techniques).
_SYSLOG_RULES: list[tuple[str, str, list[str]]] = [
    (
        r"^(?!.*(error parsing signature|SC_ERR_)).*(failed password|authentication failure|invalid user|failed login)",
        "failed_logon",
        ["T1110"],
    ),
    (r"accepted (password|publickey)|session opened", "successful_logon", []),
    (r"useradd|new user", "account_created", ["T1136"]),
    (r"sudo: .*command=", "privileged_command", []),
    (r"port ?scan|possible scan", "scan", ["T1046"]),
]


def _event(**kw: Any) -> Event:
    base: Event = {
        "ts": time.time(),
        "src_ip": None,
        "dst_ip": None,
        "dst_port": None,
        "source": "syslog",
        "kind": "log",
        "detail": "",
        "techniques": [],
    }
    base.update(kw)
    return base


def _ts(value: Any) -> float:
    """Parse ISO / epoch / Windows /Date(ms)/ timestamps; fall back to now."""
    if value is None:
        return time.time()
    s = str(value).strip()
    m = re.search(r"/Date\((\d+)", s)
    if m:
        return int(m.group(1)) / 1000.0
    try:
        f = float(s)
    except ValueError:
        pass
    else:  # epoch milliseconds (NetFlow FLOW_START_MILLISECONDS) past year 5138
        return f / 1000.0 if f > 1e11 else f
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
_RFC3164 = re.compile(r"^(?:<\d+>)?([A-Z][a-z]{2})\s+(\d{1,2}) (\d\d):(\d\d):(\d\d)\b")
_ISO_HEAD = re.compile(
    r"^(?:<\d+>\d? ?)?(\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?)"
)
_APACHE = re.compile(r"^\[\w{3} (\w{3}) (\d\d) (\d\d):(\d\d):(\d\d) (\d{4})\]")


# Syslog lines carry no year. Live logs: the current year is right. Replaying an old
# file (a dataset, an archive): the year it was written must be given, or its events
# land years away from the dated logs of the same attack (owner 2026-10-24).
SYSLOG_YEAR: list[int | None] = [None]


@contextmanager
def assume_year(year: int | None) -> Iterator[None]:
    """Parse year-less syslog timestamps as ``year`` inside this block."""
    saved = SYSLOG_YEAR[0]
    SYSLOG_YEAR[0] = year
    try:
        yield
    finally:
        SYSLOG_YEAR[0] = saved


def _line_ts(line: str) -> float | None:
    """Timestamp at the start of a log line (syslog 'Dec 10 06:55:46', ISO, or
    Apache '[Sun Dec 04 04:47:44 2005]'); None when the line carries none. Syslog
    has no year: the current one, or last year if that would be in the future."""
    if m := _ISO_HEAD.match(line):
        return _ts(m.group(1))
    if m := _APACHE.match(line):
        mon, day, hh, mm, ss, yr = m.groups()
        if mon in _MONTHS:
            return datetime(
                int(yr), _MONTHS.index(mon) + 1, int(day), int(hh), int(mm), int(ss)
            ).timestamp()
    if (m := _RFC3164.match(line)) and m.group(1) in _MONTHS:
        mon, day, hh, mm, ss = m.groups()
        fixed = SYSLOG_YEAR[0]
        year = fixed or datetime.now().year
        try:
            t = datetime(
                year, _MONTHS.index(mon) + 1, int(day), int(hh), int(mm), int(ss)
            )
        except ValueError:  # Feb 29 outside a leap year
            return None
        if fixed is None and t.timestamp() > time.time() + 86400:
            t = t.replace(year=year - 1)
        return t.timestamp()
    return None


def _ip(v: Any) -> str | None:
    """A valid IP address string, or None. Log fields are attacker-controllable."""
    try:
        return str(ipaddress.ip_address(str(v).strip()))
    except ValueError:
        return None


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --- per-format parsers ------------------------------------------------------


def parse_suricata(obj: dict[str, Any]) -> Event | None:
    if obj.get("event_type") == "http" and isinstance(obj.get("http"), dict):
        # Suricata's HTTP transaction log: the request goes through the web rules
        # (Sigma web / proxy, OWASP CRS) like a web server's access log (2026-10-24)
        from src.soc import logformats

        h = obj["http"]
        return logformats.web_request(
            _ts(obj.get("timestamp")), _ip(obj.get("src_ip")), None, str(h.get("http_method", "")),
            str(h.get("url", "")), str(h.get("status", "")), str(h.get("protocol", "")),
            str(h.get("http_user_agent", "")), str(h.get("http_refer", "")), source="suricata_http")  # fmt: skip
    if obj.get("event_type") not in ("alert", "flow", "anomaly"):
        return None
    alert = obj.get("alert") or {}
    sig = str(alert.get("signature", obj.get("event_type", "")))
    techs = (
        signature_techniques(sig, str(alert.get("category", "")))
        if obj.get("event_type") == "alert"
        else []
    )
    ev = _event(
        ts=_ts(obj.get("timestamp")),
        src_ip=_ip(obj.get("src_ip")),
        dst_ip=_ip(obj.get("dest_ip")),
        dst_port=_int(obj.get("dest_port")),
        source="suricata",
        kind="ids_alert" if obj.get("event_type") == "alert" else "conn",
        detail=sig,
        techniques=techs,
    )
    if techs:
        from src.soc import rulectl

        rulectl.hit(ev, f"ids:{alert.get('signature_id') or sig[:80]}", techs, sig)
    return ev


def _content(msg: str) -> tuple[str, list[str]] | None:
    """Technique from a command line / script block, most specific rule first."""
    for pat, k, t in (*_SCRIPT_RULES, *_SUSPICIOUS_CMDS):
        if pat.search(msg):
            return k, [t]
    return None


def _sysmon(eid: int, msg: str) -> tuple[str, list[str]]:
    from src.soc import winrules

    msg = winrules.strip_rulename(msg)  # the config author's labels, not evidence
    kind, techs = SYSMON.get(eid, (f"sysmon_{eid}", []))
    if eid == 1 and (hit := _content(msg) or winrules.process(msg)):
        return hit
    if eid == 10 and re.search(r"TargetImage:.*\\lsass\.exe", msg, re.I):
        return "lsass_access", ["T1003"]
    if eid in (11, 12, 13) and _RUN_KEY.search(msg):
        return "autostart_added", ["T1547"]
    if eid in (12, 13, 14) and (hit := winrules.registry(msg)):
        return hit
    if hit := winrules.sysmon_other(eid, msg):
        return hit
    return kind, list(techs)


_PROC_ALIASES = {"NewProcessName": "Image", "ParentProcessName": "ParentImage"}


def _fields(obj: dict[str, Any], msg: str, eid: int) -> dict[str, Any]:
    """Event data by field name: an EventData map (our .evtx reader), flat
    shipper JSON (NXLog / OTRF), else the 'Name: value' lines of the message."""
    out: dict[str, Any] = {}
    data = obj.get("EventData")
    if isinstance(data, dict):
        # Windows keeps padding in some values ("NtLmSsp " in 4624); Sigma compares
        # them exactly, so trim (2026-10-17: missed Pass-the-Hash detections)
        out.update({k: v.strip() if isinstance(v, str) else v for k, v in data.items()})
    for k, v in obj.items():
        if isinstance(v, str | int | float) and k not in out:
            out[k] = v
    # message lines fill only what is missing. Sigma compares names case-blind, so a
    # title line such as "Image loaded:" (-> "Imageloaded", empty) must never shadow
    # the event's own ImageLoaded (2026-10-16: lost DLL side-loading detections)
    have = {str(k).lower() for k in out}
    for m in re.finditer(r"^\s*([A-Za-z][\w ]{1,40}?):[ \t]*(.*?)\s*$", msg, re.M):
        key, val = m.group(1).replace(" ", ""), m.group(2)
        if val and key.lower() not in have:
            out[key] = val
            have.add(key.lower())
    out.pop("RuleName", None)  # Sysmon config author's labels, not evidence
    if eid == 4688:  # Security process creation -> the Sysmon field names rules use
        for a, b in _PROC_ALIASES.items():
            if a in out:
                out.setdefault(b, out[a])
    return out


def _with_sigma(
    obj: dict[str, Any],
    eid: int,
    msg: str,
    provider: str,
    is_sysmon: bool,
    techs: list[str],
) -> tuple[list[str], list[str]]:
    """Add the techniques of every Sigma rule that fires (src/soc/sigma.py)."""
    from src.soc import sigma

    channel = str(obj.get("Channel") or obj.get("LogName") or "")
    fields = _fields(obj, msg, eid)
    hits = sigma.match(fields, eid, provider, channel, is_sysmon)
    _HITS.clear()
    for r in hits:
        techs += [t for t in r.techniques if t not in techs]
        _HITS.append((f"sigma:{r.rule_id}", list(r.techniques), r.title))
    if (is_sysmon and eid == 1) or eid == 4688:  # documented misuse of system programs
        from src.soc import lolbins

        img = str(fields.get("Image") or fields.get("NewProcessName") or "")
        cmd = str(fields.get("CommandLine") or fields.get("ProcessCommandLine") or img)
        for p in lolbins.match(img, cmd, "windows"):
            techs += [p["technique"]] if p["technique"] not in techs else []
            _HITS.append((f"lolbas:{p['binary'].removesuffix('.exe')}", [p["technique"]], f"misuse of {p['binary']}"))
    order = {"critical": 0, "high": 1, "medium": 2}
    titles = [r.title for r in sorted(hits, key=lambda r: order.get(r.level, 3))]
    return techs, titles


_HITS: list[tuple[str, list[str], str]] = []  # rule hits of the event _with_sigma just checked


def _winsec_provider(eid: int, provider: str) -> bool:
    """Whether ``eid`` from ``provider`` is the WINSEC event of that number."""
    p = provider.lower()
    if not p:
        return True
    if eid in (7045,):
        return "service control manager" in p
    if eid == 4104:
        return "powershell" in p
    if eid == 1102:
        return "eventlog" in p or "security-auditing" in p
    return "security-auditing" in p or p == "security"


# network logons by user accounts (4624 type 3): logon id -> time seen. Actions inside such a
# session on this host (new process, scheduled task, service) are remote execution: T1021
# (owner 2026-10-23, written from the EVTX development half; public ATT&CK knowledge).
_SESSIONS: dict[str, float] = {}
SESSION_TTL_S = 24 * 3600.0
SESSION_MAX = 50_000
_REMOTE_ACTIONS = {4688: "process", 4698: "scheduled task", 4702: "scheduled task", 4697: "service"}


def _field(obj: dict[str, Any], msg: str, key: str, label: str, nth: int = 0) -> str:
    v = obj.get(key)
    if v is None:
        hits = re.findall(rf"{label}:\s*(\S+)", msg)
        v = hits[nth] if len(hits) > nth else None
    return str(v or "").strip().lower()


def remote_session(eid: int, obj: dict[str, Any], msg: str) -> tuple[str, list[str]] | None:
    """Remembers network logons and flags actions taken inside one."""
    import time as _t

    if eid == 4624:
        ltype = _field(obj, msg, "LogonType", "Logon Type")
        user = _field(obj, msg, "TargetUserName", "Account Name", 1)
        lid = _field(obj, msg, "TargetLogonId", "Logon ID", 1)
        if ltype == "3" and lid not in ("", "0x0", "0x3e7") and not user.endswith("$") and "anonymous" not in user:
            if len(_SESSIONS) >= SESSION_MAX:
                _SESSIONS.pop(next(iter(_SESSIONS)))
            _SESSIONS[lid] = _t.time()
        return None
    if eid not in _REMOTE_ACTIONS or not _SESSIONS:
        return None
    ids = {_field(obj, msg, "SubjectLogonId", "Logon ID"), _field(obj, msg, "TargetLogonId", "Logon ID", 1)}
    now = _t.time()
    for lid in ids - {"", "0x0", "0x3e7"}:
        t = _SESSIONS.get(lid)
        if t is not None and now - t <= SESSION_TTL_S:
            return f"remote_{_REMOTE_ACTIONS[eid].replace(' ', '_')}", ["T1021"]
    return None


def reset_sessions() -> None:
    """Forget remembered network logons (one log file is scored on its own in tests)."""
    _SESSIONS.clear()


def parse_winsec(obj: dict[str, Any]) -> Event | None:
    from src.soc import winrules

    eid = _int(obj.get("Id", obj.get("EventID", obj.get("EventId"))))
    if eid is None:
        return None
    msg = str(obj.get("Message", obj.get("message", "")))
    provider = str(
        obj.get("ProviderName", obj.get("Provider", obj.get("SourceName", "")))
    )
    is_sysmon = "sysmon" in provider.lower() or (
        eid < 30 and re.search(r"^(Image|UtcTime|RuleName):", msg, re.M) is not None
    )
    if is_sysmon:
        kind, techs = _sysmon(eid, msg)
    else:
        # event ids are only unique per provider (EventSystem also logs a 4625);
        # exports without a provider name are trusted as before
        if _winsec_provider(eid, provider):
            kind, techs = WINSEC.get(eid, (f"event_{eid}", []))
        else:
            kind, techs = f"event_{eid}", []
        if eid in (4688, 4104) and (hit := _content(msg)):
            kind, techs = hit
        elif eid == 4688 and (hit := winrules.process(msg)):
            kind, techs = hit
        elif hit := winrules.security(eid, msg, provider):
            kind, techs = hit
        if eid == 4624 and not techs and re.search(r"Logon Type:\s*10\b", msg):
            kind, techs = "remote_interactive_logon", ["T1021"]  # RDP
        if (hit := remote_session(eid, obj, msg)) and "T1021" not in techs:
            kind, techs = hit[0], [*techs, *hit[1]]
    # the logon source field only: an IP elsewhere in the message (a download URL
    # in a command line) is not the event's source (and crashed: no group 1)
    m = re.search(r"Source Network Address:\s*([\d.]+)", msg)
    src = _ip(obj.get("IpAddress") or (m.group(1) if m else None))
    if src in ("-", "::1", "127.0.0.1"):
        src = None
    raw_ts = next(
        (
            obj[k]
            for k in ("TimeCreated", "TimeGenerated", "timestamp", "@timestamp")
            if obj.get(k)
        ),
        None,
    )
    if raw_ts is None and obj.get("UtcTime"):  # Sysmon's own field is UTC
        raw_ts = str(obj["UtcTime"]).replace(" ", "T") + "Z"
    host = obj.get("Hostname") or obj.get("Computer") or obj.get("computer_name")
    detail = f"EventID {eid}: " + msg.splitlines()[0][:160] if msg else f"EventID {eid}"
    own_techs = list(techs)
    techs, rules = _with_sigma(obj, eid, msg, provider, is_sysmon, list(techs))
    sigma_hits = list(_HITS)
    if rules:
        detail += f" | Sigma: {rules[0]}"
    ev = _event(
        ts=_ts(raw_ts),
        src_ip=src,
        source="sysmon" if is_sysmon else "winsec",
        kind=kind,
        detail=detail + (f" [host {str(host)[:60]}]" if host else ""),
        techniques=techs,
    )
    ev = with_entities(ev, user=_win_user(obj, msg), host=host)
    from src.soc import rulectl

    rulectl.hit(ev, f"own:{kind}", own_techs, kind)
    for rid, ts_, title in sigma_hits:
        rulectl.hit(ev, rid, ts_, title)
    image, sha = _win_file(obj, msg)
    if image:
        ev["file"] = image  # the risk board checks it (hash intel / malware model)
    if sha:
        ev["sha256"] = sha
    return ev


_SHA256_FIELD = re.compile(r"SHA256=([0-9A-Fa-f]{64})")


def _win_file(obj: dict[str, Any], msg: str) -> tuple[str | None, str | None]:
    """(executable or written file path, its SHA256) from Sysmon-style fields."""
    data = obj.get("EventData") if isinstance(obj.get("EventData"), dict) else {}
    get = lambda k: (data or {}).get(k) or obj.get(k)  # noqa: E731
    image = get("Image") or get("TargetFilename") or get("ImageLoaded")
    hashes = str(get("Hashes") or "")
    m = _SHA256_FIELD.search(hashes) or _SHA256_FIELD.search(msg)
    return (str(image)[:260] if image else None), (m.group(1).lower() if m else None)


def _win_user(obj: dict[str, Any], msg: str) -> str | None:
    data = obj.get("EventData") if isinstance(obj.get("EventData"), dict) else {}
    for k in ("TargetUserName", "SubjectUserName", "User", "AccountName"):
        v = (data or {}).get(k) or obj.get(k)
        if v:
            return str(v)
    m = re.findall(r"Account Name:\s*(\S+)", msg)
    return m[-1] if m else None


def with_entities(ev: Event, user: Any = None, host: Any = None) -> Event:
    """Attach the acting user / host (correlation groups by IP, else user, else
    host). Machine accounts, SYSTEM and placeholders are not useful entities."""
    u = str(user or "").strip()
    if (
        u
        and u not in ("-", "SYSTEM", "LOCAL SERVICE", "NETWORK SERVICE")
        and not u.endswith("$")
    ):
        ev["user"] = u[:120]
    h = str(host or "").strip()
    if h and h != "-":
        ev["host"] = h[:120]
    return ev


_FAST_ALERT = re.compile(
    r"\[\*\*\]\s*\[\d+:\d+:\d+\]\s*(?P<sig>.+?)\s*\[\*\*\]\s*"
    r"(?:\[Classification:\s*(?P<cls>[^\]]*)\]\s*)?(?:\[Priority:\s*\d+\]\s*)?"
    r"\{\w+\}\s*(?P<src>[\d.:a-fA-F]+?)(?::\d+)?\s*->\s*(?P<dst>[\d.a-fA-F:]+?)(?::(?P<dport>\d+))?\s*$"
)


def _fast_ts(line: str) -> float | None:
    """fast.log time '03/04/2020-19:18:34.892288' (month first)."""
    m = re.match(r"(\d\d)/(\d\d)/(\d{4})-(\d\d):(\d\d):(\d\d)", line)
    if not m:
        return None
    mo, d, y, hh, mm, ss = (int(x) for x in m.groups())
    return datetime(y, mo, d, hh, mm, ss).timestamp()


# the sending host of a syslog line (2026-10-24, Linux host test: it was never read, so local
# events from many servers could not be told apart): RFC 3164 and RFC 5424 headers
_SYSLOG_HOST = re.compile(
    r"^(?:<\d{1,3}>)?(?:[A-Z][a-z]{2}\s+\d{1,2}\s+\d\d:\d\d:\d\d|\d{4}-\d\d-\d\dT\S+)\s+"
    r"(?P<host>[A-Za-z0-9][\w.-]{0,62})\s+[\w./-]+(?:\[\d+\])?:"
)
_SYSLOG5424_HOST = re.compile(r"^<\d{1,3}>1\s+\S+\s+(?P<host>[A-Za-z0-9][\w.-]{0,62})\s")


def syslog_host(line: str) -> str | None:
    m = _SYSLOG_HOST.match(line) or _SYSLOG5424_HOST.match(line)
    return m.group("host") if m else None


_DNSMASQ = re.compile(r"(dnsmasq|named|unbound)\[\d+\]: (query\[(?P<qt>\w+)\]|client \S+#\d+ \S+: query:)\s*"
                      r"(?P<name>\S+)(?: from (?P<ip>[\d.:a-fA-F]+))?")
_LONG_LABEL = 40


def _dns_query(line: str, m: re.Match[str]) -> Event:
    """One DNS query line (dnsmasq / BIND / Unbound): exfiltration-looking names (very
    long names or labels, data chunks) are flagged; scans are found over time
    (correlate._behaviour). 2026-10-24, from AIT-LDS v2 russellmitchell."""
    name = m.group("name").rstrip(".")
    qtype = (m.group("qt") or "").upper()
    labels = name.split(".")
    longest = max((len(x) for x in labels), default=0)
    exfil = len(name) >= 100 or longest >= _LONG_LABEL or sum(len(x) >= 20 for x in labels) >= 3
    ev = _event(ts=_line_ts(line) or time.time(), src_ip=_ip(m.group("ip")), source="dns",
                kind="dns_exfiltration" if exfil else ("dns_ptr_query" if qtype == "PTR" else "dns_query"),
                detail=f"DNS {qtype or 'query'} {name[:170]}",
                techniques=["T1048.003"] if exfil else [])  # fmt: skip
    if exfil:
        from src.soc import rulectl

        rulectl.hit(ev, "own:dns_exfiltration", ["T1048.003"], "very long DNS name (data in the query)")
    ev["dns_name"] = name[:255]
    if (h := syslog_host(line)) and h != "-":
        ev["host"] = h
    return ev


def parse_syslog(line: str) -> Event | None:
    line = line.strip()
    if not line:
        return None
    if (m := _FAST_ALERT.search(line)) is not None:  # Suricata / Snort fast.log
        sig, cls = m.group("sig").strip(), (m.group("cls") or "").strip()
        return _event(
            ts=_fast_ts(line) or _line_ts(line) or time.time(),
            src_ip=_ip(m.group("src")),
            dst_ip=_ip(m.group("dst")),
            dst_port=_int(m.group("dport")),
            source="suricata",
            kind="ids_alert",
            detail=sig[:200],
            techniques=signature_techniques(sig, cls),
        )
    if "CEF:" in line:  # passthrough from another sensor
        m = re.search(r"\bsrc=([\d.]+)", line)
        name = line.split("|")[5] if line.count("|") >= 6 else "CEF event"
        techs = signature_techniques(name)
        return _event(
            ts=_line_ts(line) or time.time(),
            src_ip=_ip(m.group(1)) if m else None,
            kind="cef",
            detail=name,
            techniques=techs,
        )
    if dq := _DNSMASQ.search(line):  # DNS server query log: the client is the actor
        return _dns_query(line, dq)
    ips = _IPV4.findall(line)
    ev = _event(
        ts=_line_ts(line) or time.time(),
        src_ip=_ip(ips[0]) if ips else None,
        kind="log",
        detail=line[:200],
    )
    if (h := syslog_host(line)) and h != "-":
        ev["host"] = h
    for pat, kind, techs in _SYSLOG_RULES:
        if re.search(pat, line, re.I):
            ev["kind"], ev["techniques"] = kind, list(techs)
            break
    return _syslog_rules(ev, line)


# syslog program -> Sigma Linux service
_PROGRAMS = {"sshd": "sshd", "sudo": "sudo", "cron": "cron", "crond": "cron",
             "vsftpd": "vsftpd", "su": "auth", "login": "auth", "clamd": "clamav"}  # fmt: skip
_PROG = re.compile(
    r"^(?:<\d+>)?(?:\w{3}\s+\d+ [\d:]+|\S+T\S+) \S+ ([\w./-]+?)(?:\[\d+\])?:"
)


_SYSLOG_USER = re.compile(
    r"\buser=<?([\w.@-]+)>?|for (?:invalid user )?([\w.@-]+) from|ruser=([\w.@-]+)"
)


def _syslog_rules(ev: Event, line: str) -> Event:
    """Sigma Linux rules + our own (src/soc/logrules.py) for a syslog line."""
    from src.soc import logformats

    um = _SYSLOG_USER.search(line)
    if um:
        with_entities(ev, user=next(g for g in um.groups() if g))
    m = _PROG.match(line)
    prog = (m.group(1) if m else "").lower()
    service = _PROGRAMS.get(prog, "syslog")
    fields = {"message": line, "program": prog}
    return logformats.finish(ev, fields, ("linux", None, service), ev["techniques"])


def _w3c(lines: list[str], source: str) -> Iterator[Event]:
    """Zeek conn.log (#fields, tab) and Windows Firewall log (#Fields, space)."""
    fields: list[str] = []
    sep = "\t" if source == "zeek" else " "
    for raw in lines:
        if raw.startswith("#fields") or raw.startswith("#Fields:"):
            fields = raw.split(sep if source == "zeek" else None)[1:]
            continue
        if not raw.strip() or raw.startswith("#") or not fields:
            continue
        row = dict(
            zip(
                fields,
                raw.rstrip("\n").split(sep if source == "zeek" else None),
                strict=False,
            )
        )
        if source == "zeek":
            yield _event(
                ts=_ts(row.get("ts")),
                src_ip=_ip(row.get("id.orig_h")),
                dst_ip=_ip(row.get("id.resp_h")),
                dst_port=_int(row.get("id.resp_p")),
                source="zeek",
                kind=(
                    "conn_rejected"
                    if row.get("conn_state") in ("REJ", "S0", "RSTOS0")
                    else "conn"
                ),
                detail=f"{row.get('proto', '')} {row.get('service', '-')} {row.get('conn_state', '')}",
            )
        else:
            ts = _ts(f"{row.get('date', '')}T{row.get('time', '')}")
            action = row.get("action", "")
            yield _event(
                ts=ts,
                src_ip=_ip(row.get("src-ip")),
                dst_ip=_ip(row.get("dst-ip")),
                dst_port=_int(row.get("dst-port")),
                source="firewall",
                kind="fw_drop" if action == "DROP" else "fw_allow",
                detail=f"{action} {row.get('protocol', '')} -> {row.get('dst-port', '')}",
            )


# --- self-evolving loop hooks (src/soc/evolve.py) ---------------------------------------
_EVOLVE: dict[str, Any] = {"mtime": None, "state": {}}
_LINE_CACHE: dict[str, Event | None] = {}


_EVOLVE_CHECK_S = 2.0  # another process's change is seen within 2 s (same process: at once)


def _evolve_state() -> dict[str, Any]:
    from src.soc import evolve

    now = time.monotonic()
    if _EVOLVE["mtime"] is not None and now - _EVOLVE.get("checked", 0.0) < _EVOLVE_CHECK_S \
            and _EVOLVE.get("path") == str(evolve.STATE):  # fmt: skip
        mt = _EVOLVE["mtime"]  # 2026-10-25: was one disk stat per log line
    else:
        try:
            mt = evolve.STATE.stat().st_mtime
        except OSError:
            mt = 0.0
        _EVOLVE["checked"], _EVOLVE["path"] = now, str(evolve.STATE)
    if mt != _EVOLVE["mtime"]:
        _EVOLVE["mtime"], _EVOLVE["state"] = mt, evolve.state()
        _LINE_CACHE.clear()
    # cached results are only valid for the rule set that produced them: when the
    # Sigma rules are swapped (update agent before/after check) the cache is dropped
    sg = sys.modules.get("src.soc.sigma")
    rules = None if sg is None else (id(getattr(sg, "load", None)), str(sg.RULES_ROOT))
    if rules != _EVOLVE.get("rules"):
        _EVOLVE["rules"] = rules
        _LINE_CACHE.clear()
    state: dict[str, Any] = _EVOLVE["state"]
    return state


def reset_cache() -> None:
    _EVOLVE["mtime"] = None
    _LINE_CACHE.clear()


def _suppress(ev: Event | None, st: dict[str, Any]) -> Event | None:
    """Drop the alert (not the event) of a rule the evolution loop retired."""
    if ev is None or not ev.get("techniques") or not st.get("suppressed"):
        return ev
    from src.soc.evolve import _rule_of

    if _rule_of(ev) in st["suppressed"]:
        ev["techniques"] = []
    return ev


def parse_line(line: str) -> Event | None:
    """One log line -> event, with the evolution loop's suppressions and line cache."""
    st = _evolve_state()
    size = int(st.get("cache") or 0)
    if size and line in _LINE_CACHE:
        hit = _LINE_CACHE[line]
        return None if hit is None else {**hit, "techniques": list(hit["techniques"])}
    ev = _suppress(_parse_line(line), st)
    if size:
        if len(_LINE_CACHE) >= size:
            _LINE_CACHE.clear()
        _LINE_CACHE[line] = (
            None if ev is None else {**ev, "techniques": list(ev["techniques"])}
        )
    return ev


def _parse_line(line: str) -> Event | None:
    """Best-effort single line: JSON (Suricata / Windows event / our own) else syslog."""
    from src.soc import logformats

    s = line.strip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, RecursionError):  # too-deep JSON: plain text
            return parse_syslog(s)
        if not isinstance(obj, dict):
            return parse_syslog(s)
        if (
            logformats.json_source(obj) is not None
        ):  # cloud / Kubernetes / SaaS / osquery
            return logformats.parse_json_record(obj)
        if "id.orig_h" in obj and "uid" in obj:  # Zeek JSON
            return logformats.zeek_event(obj, str(obj.get("_path") or logformats.zeek_path_of(obj)))
        if "event_type" in obj:
            return parse_suricata(obj)
        if any(k in obj for k in ("Id", "EventID", "EventId")):
            return parse_winsec(obj)
        if "techniques" in obj and "source" in obj:  # already normalized
            ev = _event(**{k: obj[k] for k in obj if k in _event()})
            return ev | {"ts": _ts(ev["ts"]), "src_ip": _ip(ev["src_ip"])}
        return parse_syslog(json.dumps(obj))
    if s[:1].isdigit() or s[:1] == ":" or s.startswith("-"):  # web access log?
        if web := logformats.parse_access(s):
            return web
    if 'http_user_agent="' in s or ' request="' in s or s.startswith('request="'):  # key="value" web log
        if web := logformats.parse_kv_access(s):
            return web
    if "msg=audit(" in s[:80]:
        if audit := logformats.parse_auditd(s):
            return audit
    return parse_syslog(s)


def parse_text(text: str, name: str = "") -> list[Event]:
    """Parse a whole log file's text, detecting the format from name + content."""
    from src.soc import logformats

    head = text[:4000]
    lines = text.splitlines()
    if (
        "<Event" in head and "<EventID" in head
    ):  # XML events (Splunk / Sysmon for Linux)
        return logformats.parse_xml_events(text)
    if ("#fields" in head and "id.orig_h" in head) or "#path\t" in head:
        return list(logformats.parse_zeek(lines))
    if "#Fields:" in head and "src-ip" in head:
        return list(_w3c(lines, "firewall"))
    stripped = text.lstrip()
    if (
        stripped[:1] in "{["
        and (cloud := logformats.parse_json_text(stripped)) is not None
    ):
        return cloud  # CloudTrail {"Records": [...]}, Kubernetes audit, ...
    if stripped.startswith("["):  # JSON array (Get-WinEvent | ConvertTo-Json)
        try:
            arr = json.loads(stripped)
        except (json.JSONDecodeError, RecursionError):  # too-deep JSON: plain text
            arr = None  # e.g. Apache error log lines "[Sun Dec 04 ...]": read by line
        if arr is None:
            return _parse_lines(lines)
        out = []
        for obj in arr if isinstance(arr, list) else [arr]:
            ev = parse_line(json.dumps(obj))
            if ev:
                out.append(ev)
        return out
    evs = _parse_lines(lines)
    return _maybe_unknown(text, name, lines, evs)


def _parse_lines(lines: list[str]) -> list[Event]:
    """Line-by-line formats: big files in parallel (src/soc/parallel.py), same events."""
    from src.soc import parallel

    return parallel.parse_lines(lines)


def _maybe_unknown(
    text: str, name: str, lines: list[str], evs: list[Event]
) -> list[Event]:
    """A file almost no line of which any parser recognised is an unknown log type:
    structure detection, generic indicators, rarity and a parser suggestion
    (src/soc/unknown.py). Known formats are untouched."""
    plain = sum(1 for e in evs if e["kind"] == "log" and not e["techniques"])
    if len(evs) < 5 or plain / len(evs) < 0.9:
        return evs
    from src.soc import unknown

    structure = unknown.detect_structure(text)
    if structure == "text" and sum(1 for ln in lines[:20] if _line_ts(ln)) >= 10:
        return evs  # ordinary timestamped syslog text: the syslog parser already ran
    return unknown.parse_unknown(text, name)


def parse_file(path: str | Path) -> list[Event]:
    p = Path(path)
    if p.suffix.lower() == ".evtx":
        return read_evtx(p)
    return parse_text(p.read_text("utf-8", errors="replace"), p.name)


WINDOWS_LOGS = {
    "Security": "Security",
    "Sysmon": "Microsoft-Windows-Sysmon/Operational",
    "PowerShell": "Microsoft-Windows-PowerShell/Operational",
}


def _shell() -> str:
    return (
        "pwsh"
        if subprocess.run(["where", "pwsh"], capture_output=True).returncode == 0
        else "powershell"
    )


def read_windows_logs(max_events: int = 500) -> tuple[list[Event], dict[str, str]]:
    """Pull Security + Sysmon + PowerShell events (Security needs Administrator).
    Returns (events, per-log status); a missing or unreadable log is reported,
    not fatal."""
    events: list[Event] = []
    status: dict[str, str] = {}
    for name, log in WINDOWS_LOGS.items():
        ps = (
            f"Get-WinEvent -LogName '{log}' -MaxEvents {int(max_events)} "
            "-ErrorAction Stop | Select-Object Id,TimeCreated,Message,ProviderName "
            "| ConvertTo-Json -Depth 3"
        )
        try:
            res = subprocess.run(
                [_shell(), "-NoProfile", "-Command", ps],
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            status[name] = f"error: {exc}"
            continue
        if res.returncode != 0:
            err = (res.stderr.strip().splitlines() or [""])[0][:160]
            status[name] = f"not readable: {err}"
            continue
        evs = parse_text(res.stdout or "[]", f"{name}.json")
        status[name] = f"{len(evs)} events"
        events += evs
    return events, status


# Exported .evtx files: a provider not installed here (e.g. Sysmon on an analyst
# PC) leaves Message empty, so it is rebuilt from the event's own data fields
# ("Image: ...", "CommandLine: ...": the layout Sysmon messages use anyway).
_EVTX_PS = r"""
$ErrorActionPreference = 'Stop'
Get-WinEvent -Path $env:SPLITSHIELD_EVTX -MaxEvents $env:SPLITSHIELD_MAX | ForEach-Object {
  $m = $_.Message
  $d = ([xml]$_.ToXml()).Event.EventData.Data
  $h = @{}
  foreach ($x in $d) { if ($x.Name) { $h[$x.Name] = $x.'#text' } }
  if ($_.ProviderName -eq 'Microsoft-Windows-Sysmon' -and $h.Count) {
    # rendered through THIS PC's Sysmon template, which shifts fields when the event
    # came from another Sysmon version: rebuild from the event's own named fields
    $title = if ($m) { ($m -split "`r?`n")[0] } else { '' }
    $m = (@($title) + ($d | Where-Object { $_.Name } | ForEach-Object { "$($_.Name): $($_.'#text')" })) -join "`n"
  }
  if (-not $m) { $m = ($d | ForEach-Object { "$($_.Name): $($_.'#text')" }) -join "`n" }
  [pscustomobject]@{ Id = $_.Id; TimeCreated = $_.TimeCreated.ToUniversalTime().ToString('o');
                     ProviderName = $_.ProviderName; Channel = $_.LogName; Message = $m;
                     EventData = $h }
} | ConvertTo-Json -Depth 3 -Compress
"""


def _evtx_attr(sysn: dict[str, Any], key: str, attr: str) -> Any:
    node = sysn.get(key)
    return node.get("#attributes", {}).get(attr) if isinstance(node, dict) else None


# EventData field -> the label Windows shows in the rendered message (the rules read
# these labels); added next to the raw "Field: value" lines on macOS / Linux
_EVTX_LABELS = {
    "LogonType": "Logon Type", "IpAddress": "Source Network Address",
    "TargetUserName": "Account Name", "TargetDomainName": "Account Domain",
    "WorkstationName": "Workstation Name", "ImagePath": "Service File Name",
    "ServiceFileName": "Service File Name", "ServiceName": "Service Name",
    "ProcessName": "Process Name", "NewProcessName": "New Process Name",
    "ObjectName": "Object Name", "TicketEncryptionType": "Ticket Encryption Type",
    "PreAuthType": "Pre-Authentication Type", "RelativeTargetName": "Relative Target Name",
    "ShareName": "Share Name", "CommandLine": "Process Command Line", "PrivilegeList": "Privileges",
}  # fmt: skip


_YES_NO = {"%%1842": "Yes", "%%1843": "No"}
_EVTX_BOOT = {"KernelDebug": "Kernel Debugging", "TestSigning": "Test Signing",
              "DisableIntegrityChecks": "Disable Integrity Checks", "FlightSigning": "Flight Signing"}  # fmt: skip


def _evtx_message(eid: int, channel: str, data: dict[str, str], provider: str = "") -> str:
    """Raw "Field: value" lines + the labels / sentences of Windows' rendered message
    that the rules read (only for the events the rules use)."""
    g = data.get
    lines = [f"{k}: {v}" for k, v in data.items()]
    lines += [f"{_EVTX_LABELS[k]}: {v}" for k, v in data.items() if k in _EVTX_LABELS]
    prov = provider.lower()
    if eid in (104, 1102) and "eventlog" in prov:
        lines.insert(0, f"The {channel or 'event'} log file was cleared.")
    elif eid == 4826:
        lines += [f"{lbl}: {_YES_NO.get(g(k, ''), g(k, ''))}" for k, lbl in _EVTX_BOOT.items() if k in data]
    elif eid == 3001 and "processexitmonitor" in prov:
        lines.insert(0, f"The process '{g('param1', '')}' was terminated by the process "
                        f"'{g('param2', '')}' with termination code {g('param3', '')}.")  # fmt: skip
    elif eid == 500 and "application-experience" in prov:
        lines.insert(0, f"Compatibility fix applied to {g('ExePath', '')}. Fix information: "
                        f"{g('FixName', '')}, {g('FixID', '')}, {g('Flags', '')}.")  # fmt: skip
    elif eid in (325, 326, 327) and "esent" in prov:  # "NTDS (...) ... C:\$SNAP_...\ntds.dit"
        lines.insert(0, " ".join(v for v in data.values() if v))
    elif eid in (1040, 1042) and "msiinstaller" in prov:
        verb = "Beginning" if eid == 1040 else "Ending"
        lines.insert(0, f"{verb} a Windows Installer transaction: {g('Data0', '')}. "
                        f"Client Process Id: {g('Data1', '')}.")  # fmt: skip
    return "\n".join(lines)


def _evtx_rows_portable(path: str | Path, max_events: int) -> list[dict[str, Any]]:
    """.evtx -> Get-WinEvent-like rows without Windows (macOS / Linux; owner
    2026-10-17), via the ``evtx`` library. Messages are rebuilt from the event's own
    named fields, exactly like an .evtx whose provider is not installed on Windows."""
    import json as _json

    from evtx import PyEvtxParser

    def text(v: Any) -> str:
        if isinstance(v, dict):
            return str(v.get("#text", "")) if "#text" in v else ""
        return "" if v is None else str(v)

    rows: list[dict[str, Any]] = []
    for rec in PyEvtxParser(str(path)).records_json():
        if len(rows) >= max_events:
            break
        try:
            ev = _json.loads(rec["data"])["Event"]
        except (KeyError, ValueError, TypeError):
            continue
        sysn = ev.get("System", {})
        eid = sysn.get("EventID")
        eid = eid.get("#text") if isinstance(eid, dict) else eid
        raw = ev.get("EventData") or ev.get("UserData") or {}
        data: dict[str, str] = {}
        if isinstance(raw, dict):
            if "Data" in raw:  # unnamed <Data> items
                items = raw["Data"] if isinstance(raw["Data"], list) else [raw["Data"]]
                if len(items) == 1 and isinstance(items[0], dict) and isinstance(items[0].get("#text"), list):
                    items = items[0]["#text"]
                for i, x in enumerate(items):
                    data[f"Data{i}"] = text(x)
            for k, v in raw.items():
                if k in ("Data", "#attributes"):
                    continue
                if isinstance(v, dict) and "#text" not in v:  # UserData: one nested level
                    data.update({kk: text(vv) if isinstance(vv, dict) else ("" if vv is None else str(vv))
                                 for kk, vv in v.items() if kk != "#attributes"})  # fmt: skip
                else:
                    data[k] = text(v) if isinstance(v, dict) else ("" if v is None else str(v))
        rows.append({
            "Id": int(eid or 0),
            "TimeCreated": _evtx_attr(sysn, "TimeCreated", "SystemTime"),
            "ProviderName": _evtx_attr(sysn, "Provider", "Name"),
            "Channel": sysn.get("Channel"), "Computer": sysn.get("Computer"),
            "Message": _evtx_message(int(eid or 0), str(sysn.get("Channel") or ""), data,
                                     str(_evtx_attr(sysn, "Provider", "Name") or "")),
            "EventData": data,
        })  # fmt: skip
    return rows


def read_evtx(path: str | Path, max_events: int = 200_000) -> list[Event]:  # was 5,000: big exports were cut (2026-10-25)
    """Parse an exported Windows event log (.evtx): via Get-WinEvent on Windows,
    via the ``evtx`` library on macOS / Linux. The path goes in through an
    environment variable, never into the command."""
    import os
    import sys

    if sys.platform != "win32":
        try:
            rows = _evtx_rows_portable(path, max_events)
        except (OSError, RuntimeError, ImportError) as exc:
            raise RuntimeError(f"{Path(path).name}: cannot read .evtx ({exc})") from exc
        return parse_text(json.dumps(rows), Path(path).name)
    env = {
        **os.environ,
        "SPLITSHIELD_EVTX": str(Path(path).resolve()),
        "SPLITSHIELD_MAX": str(int(max_events)),
    }
    res = subprocess.run(
        [_shell(), "-NoProfile", "-NonInteractive", "-Command", _EVTX_PS],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env=env,
    )
    if res.returncode != 0:
        err = (res.stderr.strip().splitlines() or ["unknown error"])[0][:200]
        if "No events were found" in res.stderr:
            return []
        raise RuntimeError(f"{Path(path).name}: cannot read .evtx ({err})")
    out = res.stdout.strip() or "[]"
    return parse_text(out if out.startswith("[") else f"[{out}]", Path(path).name)


def read_windows_security(max_events: int = 500) -> list[Event]:
    """Pull recent Windows Security events via PowerShell (requires Administrator).

    Raises ``RuntimeError`` with the reason if the log cannot be read.
    """
    ids = ",".join(str(i) for i in WINSEC)
    ps = (
        f"Get-WinEvent -FilterHashtable @{{LogName='Security';Id={ids}}} "
        f"-MaxEvents {int(max_events)} -ErrorAction Stop | "
        "Select-Object Id,TimeCreated,Message | ConvertTo-Json -Depth 3"
    )
    shell = (
        "pwsh"
        if subprocess.run(["where", "pwsh"], capture_output=True).returncode == 0
        else "powershell"
    )
    res = subprocess.run(
        [shell, "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if res.returncode != 0:
        raise RuntimeError(
            "Cannot read the Security log (run the dashboard as Administrator): "
            + res.stderr.strip()[:300]
        )
    return parse_text(res.stdout or "[]", "winsec.json")
