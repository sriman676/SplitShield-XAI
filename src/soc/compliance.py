"""Compliance pack (owner 2026-10-16, enterprise): evidence for NIST CSF 2.0 and
ISO/IEC 27001:2022 controls, collected from what the SOC itself records.

This SUPPORTS an audit; it does not certify anything. Each control is shown with the
evidence found in the chosen period and one of three states:
  * "evidence"  - the SOC produced records that show the control operating
  * "partial"   - some evidence, or the control depends on steps outside the SOC
  * "gap"       - nothing found in the period (what to do is listed)

Used by the report builder as the "compliance" section (src/soc/reports.py) and by the
default "Weekly compliance" schedule (Mondays 08:00).
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

# id -> (framework, title, what the SOC can show)
CONTROLS: dict[str, tuple[str, str, str]] = {
    "DE.CM-01": ("NIST CSF 2.0", "Networks and network services are monitored", "network"),
    "DE.CM-03": ("NIST CSF 2.0", "Personnel activity and technology usage are monitored", "auth"),
    "DE.CM-09": ("NIST CSF 2.0", "Computing hardware, software and their data are monitored", "endpoint"),
    "DE.AE-02": ("NIST CSF 2.0", "Potentially adverse events are analysed", "incidents"),
    "DE.AE-03": ("NIST CSF 2.0", "Information is correlated from multiple sources", "correlation"),
    "DE.AE-06": ("NIST CSF 2.0", "Information on adverse events is provided to authorised staff and tools", "forwarding"),
    "RS.AN-03": ("NIST CSF 2.0", "Analysis is performed to establish what has taken place", "reviews"),
    "RS.MA-02": ("NIST CSF 2.0", "Incident reports are triaged and validated", "reviews"),
    "PR.AA-05": ("NIST CSF 2.0", "Access permissions are defined, managed and enforced", "access"),
    "PR.PS-04": ("NIST CSF 2.0", "Log records are generated and made available for monitoring", "logging"),
    "PR.DS-11": ("NIST CSF 2.0", "Backups of data are created, protected and tested", "backup"),
    "ID.IM-01": ("NIST CSF 2.0", "Improvements are identified from evaluations", "improvement"),
    "A.5.15": ("ISO/IEC 27001:2022", "Access control", "access"),
    "A.5.25": ("ISO/IEC 27001:2022", "Assessment and decision on information security events", "reviews"),
    "A.5.26": ("ISO/IEC 27001:2022", "Response to information security incidents", "incidents"),
    "A.5.28": ("ISO/IEC 27001:2022", "Collection of evidence", "evidence"),
    "A.8.13": ("ISO/IEC 27001:2022", "Information backup", "backup"),
    "A.8.15": ("ISO/IEC 27001:2022", "Logging", "logging"),
    "A.8.16": ("ISO/IEC 27001:2022", "Monitoring activities", "monitoring"),
    "A.8.2": ("ISO/IEC 27001:2022", "Privileged access rights", "access"),
}  # fmt: skip

NETWORK_SOURCES = {"suricata", "zeek", "netflow", "detector", "firewall", "flow"}
ENDPOINT_SOURCES = {"sysmon", "winsec", "powershell", "evtx", "auditd", "osquery", "malware"}


def evidence(start: float, end: float, events: list[dict[str, Any]], incs: list[dict[str, Any]]) -> dict[str, Any]:
    """Facts for the period, read from the SOC's own records."""
    from src.soc import audit, delivery, engine, rcl, retention, users

    src = Counter(str(e.get("source") or "") for e in events)
    fam_sources = {s for s in src if s}
    ad = [r for r in audit.read() if start <= float(r.get("ts") or 0) < end]
    reviews = [r for r in engine.read_jsonl(rcl.FEEDBACK) if start <= float(r.get("ts") or 0) < end]
    try:
        from scripts import backup

        dest = backup.destination()
        backups = [p for p in Path(dest).glob("*") if start <= p.stat().st_mtime < end] if Path(dest).is_dir() else []
    except Exception:  # noqa: BLE001 - evidence only
        backups = []
    u = users.load()
    try:
        from src.soc import controller

        decisions = [h for h in controller.history(2000) if h.get("ts", "") >= time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start))]
    except Exception:  # noqa: BLE001
        decisions = []
    deliv = delivery.status()
    return {
        "events": len(events), "sources": dict(src.most_common()),
        "network": sum(n for s, n in src.items() if s in NETWORK_SOURCES),
        "endpoint": sum(n for s, n in src.items() if s in ENDPOINT_SOURCES),
        "auth": sum(1 for e in events if "logon" in str(e.get("kind", "")) or e.get("source") in ("auth", "winsec")),
        "incidents": len(incs), "multi_source": sum(1 for i in incs if len(i.get("sources", [])) > 1),
        "reviews": len(reviews), "users": len(u["users"]),
        "roles": {r: len(p) for r, p in u["roles"].items()},
        "admins": sum(1 for x in u["users"].values() if x["role"] == "admin" and not x.get("disabled")),
        "sign_ins": sum(1 for r in ad if r.get("action") == "sign_in"),
        "failed_sign_ins": sum(1 for r in ad if r.get("action") == "sign_in" and not r.get("ok")),
        "changes": sum(1 for r in ad if r.get("action") != "sign_in"),
        "audit_ok": audit.verify()["ok"],
        "delivered": sum(int(v.get("delivered") or 0) for v in deliv.values()),
        "dead": sum(int(v.get("dead") or 0) for v in deliv.values()),
        "forwarding_on": bool(deliv) or _jsonl_on(),
        "retention": retention.settings(),
        "backups": len(backups), "improvements": len(decisions),
        "monitoring": bool(fam_sources),
    }


def _jsonl_on() -> bool:
    from src.soc import sinks

    return bool(sinks.load_settings()["jsonl"]["enabled"])


def assess(ev: dict[str, Any]) -> list[dict[str, str]]:
    """One row per control: state, evidence sentence, what to do when not 'evidence'."""
    def row(state: str, text: str, todo: str = "") -> tuple[str, str, str]:
        return state, text, todo

    by_kind: dict[str, tuple[str, str, str]] = {
        "network": row("evidence", f"{ev['network']:,} network events analysed") if ev["network"]
        else row("gap", "no network events in the period", "Connect flows / Zeek / Suricata or start Live monitor."),
        "auth": row("evidence", f"{ev['auth']:,} sign-in related events") if ev["auth"]
        else row("gap", "no sign-in events", "Read this computer's logs (live logs) or send agents' auth logs."),
        "endpoint": row("evidence", f"{ev['endpoint']:,} endpoint events (process, file, script)") if ev["endpoint"]
        else row("partial", "no endpoint telemetry", "Install Sysmon (Windows) / auditd (Linux) and agents."),
        "incidents": row("evidence", f"{ev['incidents']:,} incidents built and ranked by severity")
        if ev["incidents"] else row("partial", "no incidents in the period (nothing to analyse, or no data)"),
        "correlation": row("evidence", f"{ev['multi_source']:,} incidents combine several sensors")
        if ev["multi_source"] else row("partial", f"{len(ev['sources'])} log source(s); no multi-source incident"),
        "forwarding": row("evidence", f"alerts forwarded (delivered {ev['delivered']:,}, dead-lettered {ev['dead']:,})")
        if ev["forwarding_on"] else row("gap", "no SIEM connection", "Connect SOC: pick your SIEM."),
        "reviews": row("evidence", f"{ev['reviews']:,} alerts reviewed (real / false alarm)") if ev["reviews"]
        else row("gap", "no analyst reviews recorded", "Review detector alerts on Incidents & review."),
        "access": row("evidence" if ev["users"] > 1 else "partial",
                      f"{ev['users']} account(s), roles {ev['roles']}, {ev['admins']} admin(s); "
                      f"{ev['sign_ins']} sign-ins ({ev['failed_sign_ins']} failed)",
                      "" if ev["users"] > 1 else "Give each person their own account (Users & roles)."),
        "logging": row("evidence" if ev["retention"]["enabled"] else "partial",
                       f"{ev['events']:,} events stored; retention {ev['retention']['days']} days "
                       f"({ev['retention']['mode']})", "" if ev["retention"]["enabled"] else "Switch retention on."),
        "evidence": row("evidence" if ev["audit_ok"] else "gap",
                        f"tamper-evident audit log {'intact' if ev['audit_ok'] else 'BROKEN'}; {ev['changes']:,} changes recorded",
                        "" if ev["audit_ok"] else "Investigate the broken audit chain now."),
        "backup": row("evidence", f"{ev['backups']} backup(s) made in the period") if ev["backups"]
        else row("gap", "no backup in the period", "Check the daily backup job (Getting started)."),
        "improvement": row("evidence", f"{ev['improvements']} self-improvement decisions (kept only if held-out "
                           "precision and recall did not drop)") if ev["improvements"]
        else row("partial", "no model update cycles in the period"),
        "monitoring": row("evidence", f"sources monitored: {', '.join(list(ev['sources'])[:8])}") if ev["monitoring"]
        else row("gap", "nothing monitored", "Start Live monitor or connect log sources."),
    }  # fmt: skip
    out = []
    for cid, (fw, title, kind) in CONTROLS.items():
        state, text, todo = by_kind[kind]
        out.append({"id": cid, "framework": fw, "title": title, "state": state, "evidence": text, "todo": todo})
    return out


def section(cx: dict[str, Any]) -> list[str]:
    """Report-builder section (reports.SECTIONS['compliance'])."""
    rows = assess(evidence(cx["start"], cx["end"], cx["in_period"], cx["incs"]))
    n = Counter(r["state"] for r in rows)
    md = ["## Compliance evidence (NIST CSF 2.0, ISO/IEC 27001:2022)", "",
          f"{n['evidence']} controls with evidence, {n['partial']} partial, {n['gap']} gaps. "
          "This supports an audit; it is not a certification.", "",
          "| Control | Framework | What it requires | State | Evidence in this period | To do |",
          "|---|---|---|---|---|---|"]  # fmt: skip
    md += [f"| {r['id']} | {r['framework']} | {r['title']} | {r['state']} | {r['evidence']} | {r['todo']} |"
           for r in rows]  # fmt: skip
    return md + [""]
