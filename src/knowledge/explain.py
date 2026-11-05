"""Step-by-step explanations for any ATT&CK technique and any incident
(owner 2026-10-15: "explain any attack neatly and clearly with proper steps").

Content comes from MITRE ATT&CK (src/knowledge/attack_enterprise.json: official
description, detection analytics, mitigations) plus what this SOC observed. Every
explanation has five parts:

  1. 🎯 What it is / what happened
  2. 🪜 How the attacker does it (numbered steps)
  3. 🔍 Evidence / how to detect it
  4. 🚑 What to do now (numbered response steps)
  5. 🛡️ How to prevent it

Nothing is invented: steps are the description's own sentences, in order.
"""

from __future__ import annotations

import re
import time
from typing import Any

from src.knowledge.attack_details import _enterprise

_KB: dict[str, dict[str, Any]] = {}


def kb() -> dict[str, dict[str, Any]]:
    if not _KB:
        import json
        from pathlib import Path

        p = Path(__file__).with_name("attack_enterprise.json")
        if p.is_file():
            _KB.update(json.loads(p.read_text("utf-8"))["techniques"])
    return _KB


_ALIASES = {
    "uac bypass": "T1548.002", "pass the hash": "T1550.002", "pass the ticket": "T1550.003",
    "dcsync": "T1003.006", "mimikatz": "T1003.001", "lsass dump": "T1003.001",
    "kerberoast": "T1558.003", "golden ticket": "T1558.001", "silver ticket": "T1558.002",
    "phishing": "T1566", "spearphishing": "T1566", "ransomware": "T1486",
    "web shell": "T1505.003", "webshell": "T1505.003", "sql injection": "T1190",
    "port scan": "T1046", "password spraying": "T1110.003", "credential stuffing": "T1110.004",
    "brute force": "T1110", "keylogger": "T1056.001", "dll hijacking": "T1574.001",
    "process hollowing": "T1055.012", "living off the land": "T1218", "lolbin": "T1218",
    "reverse shell": "T1059", "c2": "T1071", "command and control": "T1071",
    "data exfiltration": "T1041", "exfiltration": "T1041", "privilege escalation": "T1068",
    "lateral movement": "T1021", "persistence": "T1547", "ddos": "T1498", "dos": "T1499",
    "cryptomining": "T1496", "supply chain": "T1195", "drive-by": "T1189",
}  # fmt: skip


def find(query: str) -> str | None:
    """ATT&CK technique id for a question: an explicit id, a known alias, else the
    technique whose name best matches the words (or acronym) in the question."""
    q = query.lower()
    m = re.search(r"\bt\d{4}(?:\.\d{3})?\b", q)
    if m and m.group(0).upper() in kb():
        return m.group(0).upper()
    for alias in sorted(_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return _ALIASES[alias]
    words = set(re.findall(r"[a-z0-9]+", q)) - {"what", "is", "a", "an", "the", "how",
        "does", "do", "explain", "attack", "technique", "work", "works", "of", "to", "in"}  # fmt: skip
    best, score = None, 0.0
    for tid, v in kb().items():
        name = v["name"].lower()
        toks = set(re.findall(r"[a-z0-9]+", name))
        acro = "".join(t[0] for t in name.split() if t[:1].isalpha())
        s = len(words & toks) / max(1, len(toks)) + (
            1.0 if acro in words and len(acro) >= 3 else 0
        )
        if re.search(rf"\b{re.escape(name)}\b", q):  # whole words: not "at" in "what"
            s += 2
        if s > score:
            best, score = tid, s
    return best if score >= 0.6 else None


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", re.sub(r"\s+", " ", text).strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def technique(tid: str) -> dict[str, Any] | None:
    """The five-part explanation of one technique as structured data."""
    v = kb().get(tid) or kb().get(tid.split(".")[0])
    if not v:
        return None
    sents = _sentences(v["description"])
    tactics = _enterprise().get(tid.split(".")[0], ("", v["tactics"]))[1]
    return {
        "id": tid,
        "name": v["name"],
        "tactics": tactics,
        "platforms": v["platforms"],
        "what": " ".join(sents[:2]),
        "steps": sents[2:8],
        "detect": v.get("detection_notes")
        or ([v["detection"]] if v.get("detection") else []),
        "prevent": v.get("mitigations", []),
        "url": v["url"],
    }


def technique_markdown(tid: str) -> str:
    t = technique(tid)
    if t is None:
        return f"I do not know technique {tid}."
    lines = [
        f"### 📖 {t['name']} ({t['id']})",
        f"🧭 **Tactic:** {', '.join(x.replace('-', ' ') for x in t['tactics'])} · "
        f"💻 **Platforms:** {', '.join(t['platforms'][:6]) or '—'}",
        "",
        f"**🎯 What it is:** {t['what']}",
    ]
    if t["steps"]:
        lines += ["", "**🪜 How attackers do it:**"]
        lines += [f"{i}. {s}" for i, s in enumerate(t["steps"], 1)]
    if t["detect"]:
        lines += ["", "**🔍 How to detect it:**"]
        lines += [f"- {d}" for d in t["detect"][:4]]
    if t["prevent"]:
        lines += ["", "**🛡️ How to prevent / stop it:**"]
        lines += [f"{i}. {p}" for i, p in enumerate(t["prevent"][:6], 1)]
    lines += ["", f"_Source: MITRE ATT&CK — {t['url']}_"]
    return "\n".join(lines)


# --- incidents -------------------------------------------------------------------------

# first response steps per tactic (SANS/NIST incident-handling order)
_RESPONSE = {
    "reconnaissance": "Check what was probed; make sure nothing exposed should not be.",
    "initial-access": "Find the entry point (exposed service, phishing mail, stolen login) and close it.",
    "execution": "Identify the process or script that ran; collect it for analysis before removing it.",
    "persistence": "Remove the persistence (service, task, run key, new account) after documenting it.",
    "privilege-escalation": "Assume the attacker had admin/root rights on that host; plan a rebuild.",
    "defense-evasion": "Re-enable any disabled logging or security tools; treat missing logs as suspicious.",
    "credential-access": "Reset the passwords / keys / tokens that were exposed; revoke sessions; enforce MFA.",
    "discovery": "Expect a next step; watch the same account / host closely.",
    "lateral-movement": "Check every host this source logged on to; isolate the ones it reached.",
    "collection": "Find what data was gathered and where it was staged.",
    "command-and-control": "Block the remote address / domain at the firewall and DNS; look for other hosts talking to it.",
    "exfiltration": "Estimate what left the network; involve legal / data protection if personal data is involved.",
    "impact": "Restore from clean backups; check backups were not touched.",
}


def incident_markdown(inc: dict[str, Any], clean: Any = None) -> str:
    """Five-part write-up of a correlated incident, from what the SOC observed. Every
    value that comes from log content (entity, sources, findings, event kind and detail,
    the incident id) is escaped: the text is rendered as Markdown (pentest re-check
    2026-10-23: a log line could carry a link or a remote image). ``clean`` replaces the
    Markdown escaping for writers that are not rendered as Markdown (Word / HTML
    reports use plain cleaned text instead of backslash escapes)."""
    from src.soc.safe import md as _md

    md = clean or _md

    when = time.strftime
    techs = inc.get("techniques", [])
    lines = [
        f"### 📋 Incident {md(inc['id'], 160)} — {inc['level'].upper()} (severity {inc['severity']}/10)",
        "",
        f"**🎯 What happened:** activity from **{md(inc['src_ip'], 160)}** between "
        f"{when('%Y-%m-%d %H:%M', time.localtime(inc['start']))} and "
        f"{when('%Y-%m-%d %H:%M', time.localtime(inc['end']))}, seen by "
        f"{', '.join(md(x) for x in inc['sources'])}. Attack path:",
    ]
    for i, t in enumerate(techs, 1):
        lines.append(
            f"{i}. **{t['name']}** ({t['id']}, {t['tactic'].replace('-', ' ')})"
        )
    if techs:
        lines += ["", "**🪜 How the attacker did it:**"]
        for t in techs[:5]:
            info = technique(t["id"])
            if info:
                lines.append(f"- **{t['name']}:** {info['what']}")
    lines += ["", "**🔍 Evidence:**"]
    lines += [f"- {md(f)}" for f in inc.get("findings", [])]
    for e in inc.get("timeline", [])[:8]:
        lines.append(
            f"- {when('%H:%M:%S', time.localtime(e['ts']))} · {md(e['source'])} · "
            f"{md(e['kind'])} — {md(e['detail'], 140)}"
        )
    lines += [
        "",
        "**🚑 What to do now:**",
        "1. Confirm it is not an authorised test or admin task.",
    ]
    n = 2
    for tac in dict.fromkeys(t["tactic"] for t in techs):
        if tac in _RESPONSE:
            lines.append(f"{n}. {_RESPONSE[tac]}")
            n += 1
    lines.append(f"{n}. Keep the evidence (export this incident) before cleaning up.")
    prevent: list[str] = []
    for t in techs[:4]:
        info = technique(t["id"])
        prevent += [p for p in (info or {}).get("prevent", [])[:2] if p not in prevent]
    if prevent:
        lines += ["", "**🛡️ How to prevent it next time:**"]
        lines += [f"{i}. {p}" for i, p in enumerate(prevent[:6], 1)]
    return "\n".join(lines)
