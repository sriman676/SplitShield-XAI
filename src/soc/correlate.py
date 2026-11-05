"""Correlate events from many log sources into attack incidents, mapped to ATT&CK.

Algorithm (deterministic, explainable — every incident says *why* it exists):

1. Group events by source IP; split a group into separate incidents when the gap
   between consecutive events exceeds ``window_s``.
2. Scan heuristic: a source touching >= ``scan_threshold`` distinct (host, port)
   pairs inside one incident (Zeek conns, firewall drops, flows) adds T1046.
3. Chain rules raise severity for known multi-stage patterns, e.g. failed logons
   followed by a successful logon from the same IP => likely compromised account
   (T1078), or discovery followed by exploitation.
4. Severity = highest tactic weight + 1 per extra tactic + 1 when >= 2 independent
   log sources agree (corroboration), capped at 10.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

from src.soc.parsers import Event

# technique -> (name, tactic)


def technique_info(tid: str) -> tuple[str, str]:
    """(name, tactic) for any technique id (curated table, then full ATT&CK)."""
    from src.knowledge.attack_details import technique_info as info

    return info(tid)


TECHNIQUES: dict[str, tuple[str, str]] = {
    "T1595": ("Active Scanning", "reconnaissance"),
    "T1046": ("Network Service Discovery", "discovery"),
    "T1110": ("Brute Force", "credential-access"),
    "T1078": ("Valid Accounts", "initial-access"),
    "T1190": ("Exploit Public-Facing Application", "initial-access"),
    "T1059": ("Command and Scripting Interpreter", "execution"),
    "T1136": ("Create Account", "persistence"),
    "T1543": ("Create or Modify System Process", "persistence"),
    "T1098": ("Account Manipulation", "persistence"),
    "T1070": ("Indicator Removal", "defense-evasion"),
    "T1218": ("System Binary Proxy Execution", "defense-evasion"),
    "T1557": ("Adversary-in-the-Middle", "credential-access"),
    "T1071": ("Application Layer Protocol (C2)", "command-and-control"),
    "T1105": ("Ingress Tool Transfer", "command-and-control"),
    "T1048": ("Exfiltration Over Alternative Protocol", "exfiltration"),
    "T1498": ("Network Denial of Service", "impact"),
    "T1499": ("Endpoint Denial of Service", "impact"),
    "T1490": ("Inhibit System Recovery", "impact"),
    "T1003": ("OS Credential Dumping", "credential-access"),
    "T1055": ("Process Injection", "defense-evasion"),
    "T1053": ("Scheduled Task/Job", "persistence"),
    "T1547": ("Boot or Logon Autostart Execution", "persistence"),
    "T1021": ("Remote Services", "lateral-movement"),
    "T1562": ("Impair Defenses", "defense-evasion"),
    "T1027": ("Obfuscated Files or Information", "defense-evasion"),
    # Windows endpoint techniques (src/soc/winrules.py, owner 2026-10-14)
    "T1548": ("Abuse Elevation Control Mechanism", "privilege-escalation"),
    "T1134": ("Access Token Manipulation", "privilege-escalation"),
    "T1546": ("Event Triggered Execution", "persistence"),
    "T1574": ("Hijack Execution Flow", "persistence"),
    "T1047": ("Windows Management Instrumentation", "execution"),
    "T1569": ("System Services", "execution"),
    "T1033": ("System Owner/User Discovery", "discovery"),
    "T1087": ("Account Discovery", "discovery"),
    "T1069": ("Permission Groups Discovery", "discovery"),
    "T1016": ("System Network Configuration Discovery", "discovery"),
    "T1082": ("System Information Discovery", "discovery"),
    "T1018": ("Remote System Discovery", "discovery"),
    "T1558": ("Steal or Forge Kerberos Tickets", "credential-access"),
    "T1555": ("Credentials from Password Stores", "credential-access"),
    "T1550": ("Use Alternate Authentication Material", "lateral-movement"),
    "T1564": ("Hide Artifacts", "defense-evasion"),
    "T1222": ("File and Directory Permissions Modification", "defense-evasion"),
    "T1112": ("Modify Registry", "defense-evasion"),
    "T1036": ("Masquerading", "defense-evasion"),
    "T1220": ("XSL Script Processing", "defense-evasion"),
    "T1197": ("BITS Jobs", "defense-evasion"),
    "T1505": ("Server Software Component", "persistence"),
    "T1572": ("Protocol Tunneling", "command-and-control"),
}

# Kill-chain order (for the timeline) and severity weight per tactic.
TACTIC_ORDER = [  # ATT&CK Enterprise order (v17 and v18 tactic names)
    "reconnaissance",
    "resource-development",  # 2026-10-23: was missing - its techniques crashed incident building
    "initial-access",
    "execution",
    "persistence",
    "privilege-escalation",
    "defense-evasion",  # ATT&CK up to v17; v18 splits it into the next two
    "stealth",
    "defense-impairment",
    "credential-access",
    "discovery",
    "lateral-movement",
    "collection",
    "command-and-control",
    "exfiltration",
    "impact",
]
def _kb_tactic_order() -> list[str]:
    """ATT&CK's tactic order from the knowledge base (refreshed by the update agent)."""
    from utils.paths import PROJECT_ROOT

    try:
        kb = json.loads((PROJECT_ROOT / "src" / "knowledge" / "attack_enterprise.json").read_text("utf-8"))
        return [t["shortname"] for t in kb.get("tactics", []) if t.get("shortname")]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def _merged_order(known: list[str], kb_order: list[str]) -> list[str]:
    """The knowledge base's order, with every known name (older ATT&CK versions) kept at
    its place; nothing is dropped, so older rule tags still sort correctly."""
    out = list(kb_order)
    for i, t in enumerate(known):
        if t not in out:
            prev = next((k for k in reversed(known[:i]) if k in out), None)
            out.insert(out.index(prev) + 1 if prev else 0, t)
    return out


TACTIC_ORDER = _merged_order(TACTIC_ORDER, _kb_tactic_order())
NEW_TACTIC_WEIGHT = 5  # a tactic newer than the weight table (owner 2026-10-23: neutral + notify)


def tactic_rank(tactic: str) -> int:
    """Position in the ATT&CK order; a tactic the table does not know sorts last instead of
    raising (a new ATT&CK release or rule tag must never break incident building)."""
    return TACTIC_ORDER.index(tactic) if tactic in TACTIC_ORDER else len(TACTIC_ORDER)


TACTIC_WEIGHT = {
    "reconnaissance": 2,
    "resource-development": 2,  # before any contact with the victim, like reconnaissance
    "discovery": 3,
    "credential-access": 5,
    "initial-access": 6,
    "execution": 6,
    "persistence": 6,
    "defense-evasion": 6,
    "stealth": 6,
    "defense-impairment": 6,
    "privilege-escalation": 7,
    "lateral-movement": 7,
    "collection": 7,
    "command-and-control": 7,
    "exfiltration": 8,
    "impact": 7,
}

# Detector class -> techniques (CIC family9 + CIC-IoT + binary schemes).
CLASS_TECHNIQUES: dict[str, list[str]] = {
    "DoS": ["T1499"],
    "DDoS": ["T1498"],
    "PortScan": ["T1046"],
    "Recon": ["T1595", "T1046"],
    "BruteForce": ["T1110"],
    "WebAttack": ["T1190"],
    "WebBased": ["T1190"],
    "Bot": ["T1071"],
    "Mirai": ["T1071", "T1498"],
    "Infiltration": ["T1105"],
    "Heartbleed": ["T1190"],
    "Spoofing": ["T1557"],
    "ATTACK": [],
}


LEVEL_ICON = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}


def level_of(sev: float) -> str:
    """Severity 0-10 -> incident level."""
    return (
        "critical"
        if sev >= 8
        else "high"
        if sev >= 6
        else "medium"
        if sev >= 4
        else "low"
    )


# a time-group also ends after this much nonstop activity (owner 2026-10-23): a source
# that never pauses for ``window_s`` would otherwise become one ever-growing incident
# (2.09M entity-less events in one group exhausted memory in the 10M benchmark)
MAX_GROUP_S = 86_400.0


def _split_by_gap(events: list[Event], window_s: float) -> list[list[Event]]:
    groups: list[list[Event]] = []
    for ev in sorted(events, key=lambda e: e["ts"]):
        if (groups and ev["ts"] - groups[-1][-1]["ts"] <= window_s
                and ev["ts"] - groups[-1][0]["ts"] <= MAX_GROUP_S):  # fmt: skip
            groups[-1].append(ev)
        else:
            groups.append([ev])
    return groups


def _chains(evs: list[Event], techs: set[str]) -> list[str]:
    """Multi-stage rules; adds techniques to ``techs`` and returns findings."""
    findings = []
    kinds = [e["kind"] for e in evs]
    fails = [
        i
        for i, k in enumerate(kinds)
        if k == "failed_logon" or "T1110" in evs[i]["techniques"]
    ]
    succ = [i for i, k in enumerate(kinds) if k == "successful_logon"]
    if fails and any(s > fails[0] for s in succ):
        techs.add("T1078")
        findings.append(
            f"{len(fails)} failed logon(s) then a successful logon from the same source: "
            "likely compromised credentials (T1110 -> T1078)."
        )
    tactics = {technique_info(t)[1] for t in techs} - {"?"}
    if tactics & {"discovery", "reconnaissance"} and tactics & {
        "initial-access",
        "credential-access",
    }:
        findings.append(
            "Reconnaissance followed by an access attempt: a staged intrusion, not noise."
        )
    if tactics & {"persistence", "defense-evasion"} and len(tactics) >= 3:
        findings.append(
            "Post-compromise activity (persistence / log tampering) observed."
        )
    return findings


def entity_of(ev: Event) -> str:
    """Who an event is about: source IP, else the acting user, else the host
    (cloud API calls via AWS services, Linux audit and Windows events often have
    a user or host but no useful IP), else ``unknown:<source>:<file>`` (owner
    2026-10-23: events with none of these were all one entity, so unrelated logs
    merged into one incident; now they are grouped per log source and file)."""
    if ev.get("src_ip"):
        return str(ev["src_ip"])
    if ev.get("user"):
        return f"user:{ev['user']}"
    if ev.get("host"):
        return f"host:{ev['host']}"
    src, file = ev.get("source") or "", ev.get("file") or ""
    return f"unknown:{src}:{file}" if src or file else "unknown"


def is_unknown(entity: str | None) -> bool:
    """An entity with no IP, user or host (``unknown`` or ``unknown:<source>:<file>``)."""
    return entity in (None, "") or entity == "unknown" or str(entity).startswith("unknown:")


def unknown_label(entity: str) -> str:
    """Plain text for an entity-less incident: "no IP / user / host (syslog, auth.log)"."""
    parts = [p for p in str(entity).split(":", 2)[1:] if p]
    return "no IP / user / host" + (f" ({', '.join(parts)})" if parts else "")


_RECON_API = ("List", "Describe", "Get")


DNS_ENUM_MIN = 100  # distinct names under one parent domain in one time-group
PTR_SWEEP_MIN = 50  # distinct reverse lookups in one time-group


def _behaviour(grp: list[Event], techs: set[str], first: dict[str, float]) -> list[str]:
    """Behaviour-over-time rules: patterns no single event shows (owner 2026-10-15)."""
    findings = []
    cloud = [e for e in grp if e["source"] == "cloudtrail"]
    recon = {e["kind"] for e in cloud if str(e["kind"]).startswith(_RECON_API)}
    if len(recon) >= 15:
        techs.add("T1580")
        first.setdefault("T1580", min(e["ts"] for e in cloud if e["kind"] in recon))
        findings.append(
            f"{len(recon)} different List/Describe/Get cloud API calls in one session: "
            "cloud reconnaissance (T1580)."
        )
    denied = [
        e
        for e in cloud
        if "AccessDenied" in e["detail"] or "Unauthorized" in e["detail"]
    ]
    if len(denied) >= 10:
        techs.add("T1580")
        first.setdefault("T1580", denied[0]["ts"])
        findings.append(
            f"{len(denied)} denied cloud API calls: someone is probing which permissions "
            "these credentials have."
        )
    failed_users = {
        e.get("user")
        for e in grp
        if "T1110" in e.get("techniques", []) and e.get("user")
    }
    if len(failed_users) >= 5:
        findings.append(
            f"Failed logons against {len(failed_users)} different accounts from one source: "
            "password spraying (T1110.003)."
        )
    # DNS (2026-10-24, AIT-LDS v2 russellmitchell): many names under one parent domain
    # = subdomain brute force; many reverse lookups = sweeping the network for hosts
    names = {str(e.get("dns_name", "")).lower() for e in grp if e["kind"] == "dns_query"}
    parents = Counter(".".join(n.split(".")[1:]) for n in names if n.count(".") >= 2)
    if parents and (top := parents.most_common(1)[0])[1] >= DNS_ENUM_MIN:
        techs.add("T1590.002")
        first.setdefault("T1590.002", next(e["ts"] for e in grp if e["kind"] == "dns_query"))
        findings.append(f"{top[1]} different names under {top[0][:60]} looked up: subdomain enumeration (T1590.002).")
    ptr = {str(e.get("dns_name", "")) for e in grp if e["kind"] == "dns_ptr_query"}
    if len(ptr) >= PTR_SWEEP_MIN:
        techs.add("T1018")
        first.setdefault("T1018", next(e["ts"] for e in grp if e["kind"] == "dns_ptr_query"))
        findings.append(f"{len(ptr)} different reverse (PTR) lookups: sweeping the network for hosts (T1018).")
    web_4xx = [
        e
        for e in grp
        if e["kind"] == "http_request" and re.search(r"-> 4\d\d\b", e["detail"])
    ]
    if len(web_4xx) >= 50:
        techs.add("T1595")
        first.setdefault("T1595", web_4xx[0]["ts"])
        findings.append(
            f"{len(web_4xx)} failed web requests (4xx) from one source: automated web "
            "scanning or content discovery (T1595)."
        )
    return findings


def correlate(
    events: list[Event],
    window_s: float = 900.0,
    scan_threshold: int = 20,
    board: Any = None,
) -> list[dict[str, Any]]:
    """Return incidents (most severe first), grouped per entity (IP, else user, else
    host). Events with none of these are grouped under ``unknown`` so host-local
    evidence (e.g. a cleared audit log) still shows."""
    by_ip: dict[str, list[Event]] = defaultdict(list)
    for ev in events:
        by_ip[entity_of(ev)].append(ev)
    pre = []
    for ip, evs in by_ip.items():
        for grp in _split_by_gap(evs, window_s):
            inc = group_incident(ip, grp, scan_threshold)
            if inc is not None:
                pre.append(inc)
    return finish(pre, board)


def finish(pre: list[dict[str, Any]], board: Any = None) -> list[dict[str, Any]]:
    """Board-independent incidents (``group_incident``) -> final incidents: the shared
    risk board (src/soc/riskboard.py) adds points when other model families have
    findings on this entity or one link away. Most severe first; ties: latest first,
    then by id (stable whatever order the entities were read in)."""
    return sorted((finish_one(p, board) for p in pre), key=incident_order)


def incident_order(i: dict[str, Any]) -> tuple[float, float, str]:
    """Most severe first; ties: latest first, then by id."""
    return (-i["severity"], -i["end"], i["id"])


def finish_one(p: dict[str, Any], board: Any = None) -> dict[str, Any]:
    """One board-independent incident -> the final incident (see ``finish``)."""
    inc = {k: v for k, v in p.items() if not k.startswith("_")}
    linked: list[str] = []
    boost = 0
    if board is not None:
        weak = set(p.get("_weak", ())) if not getattr(getattr(board, "b", board), "link_failed", True) else set()
        pts, linked = board.boost(
            [k for k in p["_keys"] if k not in weak], set(p["_own"]), inc["start"], inc["end"],
            primary=p["_primary"], victims=p.get("_victims", ()),
        )
        boost = int(round(pts))
    sev = min(10, p["_sev"] + boost)
    return inc | {
        "severity": sev,
        "level": level_of(sev),
        "corroborated": len(inc["sources"]) >= 2 or bool(linked),
        "linked_findings": linked,
        "board_boost": boost,
    }


def _local_ip(ip: str) -> bool:
    """Loopback / unspecified: the machine itself, never an outside attacker."""
    from src.soc.riskboard import _local

    return _local(ip)


def group_incident(
    ip: str, grp: list[Event], scan_threshold: int = 20
) -> dict[str, Any] | None:
    """One time-group of one entity's events (sorted by time) -> its incident before
    the risk board is applied (``finish``), or None when nothing in it is suspicious.
    Keys starting with ``_`` are what ``finish`` needs and are not shown."""
    from src.soc.riskboard import entities_of, entity_key, family_of, is_failed_logon

    techs = {t for e in grp for t in e.get("techniques", [])}
    targets = {
        (e.get("dst_ip"), e.get("dst_port"))
        for e in grp
        if e.get("dst_port") is not None
    }
    findings = []
    first: dict[str, float] = {}  # technique -> first time seen (attack path order)
    for e in grp:
        for t in e.get("techniques", []):
            first.setdefault(t, e["ts"])
    scan_sources: set[str] = set()
    if len(targets) >= scan_threshold:
        techs.add("T1046")
        port_evs = [e for e in grp if e.get("dst_port") is not None]
        first.setdefault("T1046", port_evs[0]["ts"])
        scan_sources = {e["source"] for e in port_evs}
        findings.append(
            f"Touched {len(targets)} distinct host:port pairs (scan pattern)."
        )
    findings += _chains(grp, techs)
    findings += _behaviour(grp, techs, first)
    if "T1078" in techs:
        # the compromising logon is the first success AFTER the failures
        # (an earlier, legitimate logon must not reorder the attack path)
        fail_ts = first.get("T1110", grp[0]["ts"])
        first.setdefault(
            "T1078",
            next(
                (
                    e["ts"]
                    for e in grp
                    if e["kind"] == "successful_logon" and e["ts"] >= fail_ts
                ),
                grp[-1]["ts"],
            ),
        )
    if not techs:
        return None  # plain traffic with nothing suspicious: not an incident
    tactics = sorted({technique_info(t)[1] for t in techs} - {"?"}, key=tactic_rank)
    sources = sorted(
        scan_sources
        | {e["source"] for e in grp if e.get("techniques") or e["source"] == "detector"}
    )
    sev = max((TACTIC_WEIGHT.get(t, NEW_TACTIC_WEIGHT) for t in tactics), default=3)
    sev += max(0, len(tactics) - 1) + (1 if len(sources) >= 2 else 0)
    sev = min(10, sev + (2 if any("T1078" in f for f in findings) else 0))
    # users' severity overrides of built-in rules (src/soc/rulectl.py): when every
    # suspicious event carries one, they decide; otherwise they can only raise it
    hints = [int(e["severity_hint"]) for e in grp if e.get("techniques") and e.get("severity_hint")]
    if hints:
        n_sus = sum(1 for e in grp if e.get("techniques"))
        sev = max(hints) if len(hints) == n_sus else max(sev, max(hints))
    return {
        "id": f"{ip}-{int(grp[0]['ts'])}",
        "src_ip": ip,
        "start": grp[0]["ts"],
        "end": grp[-1]["ts"],
        "n_events": len(grp),
        "sources": sources,
        "techniques": [
            {"id": t, "name": technique_info(t)[0], "tactic": technique_info(t)[1]}
            for t in sorted(techs, key=lambda t: first.get(t, grp[-1]["ts"]))
        ],
        "tactics": tactics,
        "findings": findings,
        "timeline": [
            {
                "ts": e["ts"],
                "source": e["source"],
                "kind": e["kind"],
                "detail": e["detail"],
            }
            for e in grp
            if e.get("techniques")
            or e["kind"] in ("successful_logon", "fw_drop", "conn_rejected")
        ][:200],
        # for finish(): severity before the board, the entity keys and families
        "_sev": sev,
        "_keys": sorted({k for e in grp for k in entities_of(e)}),
        # entities named only in failed logons: the actor tried them, it is not them
        "_weak": sorted({k for e in grp if is_failed_logon(e) for k in entities_of(e)}
                        - {k for e in grp if not is_failed_logon(e) for k in entities_of(e)} - {entity_key(ip)}),
        "_own": sorted({family_of(e) for e in grp if e.get("techniques")}),
        "_primary": entity_key(ip),
        # machines this actor attacked from outside (they logged its suspicious events)
        "_victims": sorted({str(e["observer"]).lower() for e in grp
                            if e.get("techniques") and e.get("observer") and e.get("src_ip")
                            and not _local_ip(str(e["src_ip"]))}),  # fmt: skip
    }
