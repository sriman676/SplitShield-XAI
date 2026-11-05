"""How an answer is laid out (owner 2026-10-15): answer first, then evidence, then
details; numbered "What to do now" steps; style that adapts to the situation.

Styles (``answer["style"]``):
  * ``urgent``   - an attack is active right now (status ONGOING, or a high/critical
                   incident with activity in the last hour): the answer line, then
                   the steps, then the evidence;
  * ``learning`` - "what is / explain / how does ..." questions (or beginner mode):
                   full explanation, plus short plain-word definitions of jargon;
  * ``expert``   - expert phrasing (technique ids, event ids, log fields, queries,
                   Sigma) or expert mode: adds technique ids, Windows event ids, log
                   fields and an example search;
  * ``normal``   - everything else.

``mode`` is the user's override: ``auto`` (detect), ``beginner`` or ``expert``.
An active attack always puts the steps first, whatever the mode.

Every step and suggestion is built from facts already in the answer (IPs,
techniques, sources from the logs; mitigations from MITRE ATT&CK) - nothing is
invented here.
"""

from __future__ import annotations

import re
import time
from typing import Any

MODES = ("auto", "beginner", "expert")
ONGOING_S = 3600

_EXPERT = re.compile(
    r"\bT\d{4}\b|\b(event ?id|eventcode|event code|sysmon|sigma|spl|kql|query|queries|"
    r"field|fields|ioc|iocs|ttp|ttps|yara|regex|pcap|netflow|zeek|siem|4625|4624|4688|"
    r"4720|1102|7045)\b",
    re.I,
)
_LEARNING = re.compile(
    r"^\s*(what\s+(is|are|does)|explain|describe|tell me about|how\s+(does|do|is)|"
    r"why\s+(is|do|does)|teach me|what's)\b",
    re.I,
)

# plain-word definitions shown in beginner / learning style (only for words present)
GLOSSARY = {
    "ATT&CK": "MITRE ATT&CK is a public catalogue of how attackers work.",
    "tactic": "A tactic is the attacker's goal at that step (for example, stealing passwords).",
    "technique": "A technique is how the attacker reaches that goal.",
    "incident": "An incident is a group of related alerts from the same source.",
    "SIEM": "A SIEM is a central system that collects security logs.",
    "Sysmon": "Sysmon is a free Microsoft tool that records what programs do on Windows.",
    "firewall": "A firewall decides which network connections are allowed.",
    "MFA": "MFA means a second login step, such as a code on your phone.",
}


def wants_expert(q: str, mode: str = "auto") -> bool:
    """Expert detail: chosen by the reader, or expert phrasing in the question."""
    return mode == "expert" or (mode == "auto" and bool(_EXPERT.search(q)))


def detect(q: str, intent: str, mode: str = "auto", urgent: bool = False) -> str:
    """The style for this answer (see module doc)."""
    if urgent:
        return "urgent"
    if mode == "expert":
        return "expert"
    if mode == "beginner":
        return "learning"
    if _EXPERT.search(q):
        return "expert"
    if intent in ("explain", "explain_technique", "defend", "help") or _LEARNING.search(
        q
    ):
        return "learning"
    return "normal"


def is_urgent(intent: str, status: dict[str, Any] | None, incs: list[dict[str, Any]],
              now: float | None = None) -> bool:  # fmt: skip
    """An attack is active now: status ONGOING, or a high/critical incident that saw
    activity within the last hour."""
    now = time.time() if now is None else now
    if intent == "status" and status and status.get("status") == "ongoing":
        return True
    if intent not in ("status", "incidents", "ip", "mitre", "top"):
        return False
    if intent in ("incidents", "ip", "mitre", "top"):
        return any(
            i.get("level") in ("high", "critical")
            and now - float(i.get("end", 0)) <= ONGOING_S
            for i in incs
        )
    return False


# --- steps ---------------------------------------------------------------------------


def _first_sentence(text: str, limit: int = 160) -> str:
    s = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0].strip()
    s = s if len(s) <= limit else s[: limit - 1].rsplit(" ", 1)[0] + "…"
    return s if s.endswith((".", "!", "?", "…")) else s + "."


def _mitigation(tids: list[str]) -> str | None:
    """First ATT&CK mitigation for the first technique that has one."""
    from src.knowledge.explain import technique

    for tid in tids:
        t = technique(tid)
        if t and t["prevent"]:
            return _first_sentence(str(t["prevent"][0]))
    return None


def _response(tids: list[str]) -> str | None:
    """Incident-handling first step for the technique's tactic (SANS/NIST order)."""
    from src.knowledge.explain import _RESPONSE
    from src.soc.correlate import technique_info

    for tid in tids:
        tactic = technique_info(tid.split(".")[0])[1]
        if tactic in _RESPONSE:
            return _RESPONSE[tactic]
    return None


def steps(subject: str, ips: list[str], tids: list[str], sources: list[str],
          status: str, plain: str) -> list[str]:  # fmt: skip
    """Numbered 'What to do now' steps: contain, investigate, fix, verify."""
    from src.soc.correlate import is_unknown

    ips = list(dict.fromkeys(ip for ip in ips if not is_unknown(ip)))[:3]
    who = ", ".join(ips[:-1]) + " and " + ips[-1] if len(ips) > 1 else "".join(ips)
    out: list[str] = []
    if status == "possible":
        out.append(
            f"**Confirm first:** ask *is there any {plain}, including low-confidence "
            "alerts?* and mark those alerts *Real* or *False alarm* on 🧩 Incidents & review."
        )
    elif ips:
        many = len(ips) > 1
        verb = (
            ("are still" if many else "is still")
            if status == "ongoing"
            else ("were" if many else "was")
        )
        it, its = ("them", "they") if many else ("it", "it")
        out.append(
            f"**Contain:** {who} {verb} the {'sources' if many else 'source'}. Consider "
            f"blocking {it} at the firewall (blocking stays your decision) and isolate any "
            f"host {its} reached."
        )
    else:
        out.append(
            "**Contain:** isolate the affected host or account until it has been checked."
        )
    where = f" for {who}" if who else ""
    seen = f" in {', '.join(sources[:4])}" if sources else ""
    out.append(
        f"**Investigate:** open 🧩 Incidents & review and read the timeline{where}; "
        f"check which accounts and hosts were touched{seen}."
    )
    fix = [x for x in (_response(tids), _mitigation(tids)) if x]
    if fix:
        out.append("**Fix:** " + " ".join(fix))
    else:
        out.append(
            f"**Fix:** apply the defences for {subject} (ask me *how do I stop {plain}?*)."
        )
    out.append(
        f"**Verify:** in an hour, ask me *is {plain} still going on?* It should say "
        "`HAPPENED EARLIER` or `NOT SEEN`."
    )
    return [f"{i}. {s}" for i, s in enumerate(out, 1)]


# --- expert detail ---------------------------------------------------------------------


def expert_lines(tids: list[str], ips: list[str]) -> list[str]:
    """Technique ids, Windows event ids, log fields and an example search."""
    from src.soc.correlate import technique_info
    from src.soc.parsers import WINSEC

    tids = list(dict.fromkeys(t for t in tids if t))[:5]
    if not tids:
        return []
    out = ["🧑‍💻 **Expert detail:**"]
    out.append("- Techniques: " + "; ".join(
        f"{t} {technique_info(t.split('.')[0])[0]} ({technique_info(t.split('.')[0])[1]})"
        for t in tids))  # fmt: skip
    base = {t.split(".")[0] for t in tids}
    ids = sorted(eid for eid, (_k, ts) in WINSEC.items() if base & set(ts))
    if ids:
        out.append("- Windows Security event ids: " + ", ".join(map(str, ids)) + ".")
    out.append("- Event fields in this SOC: `ts`, `src_ip`, `user`, `host`, `source`, "
               "`kind`, `techniques`, `detail`.")  # fmt: skip
    where = " OR ".join(f"EventCode={i}" for i in ids[:4])
    ip = f' src_ip="{ips[0]}"' if ips else ""
    if where:
        out.append(
            f"- Example Splunk search: `index=* ({where}){ip} | stats count by src_ip, user, host`"
        )
    out.append(
        f"- Sigma rules: search `third_party/sigma` for the tag `attack.{tids[0].lower()}`."
    )
    return out


def glossary_lines(text: str) -> list[str]:
    hits = [
        d for k, d in GLOSSARY.items() if re.search(rf"\b{re.escape(k)}\b", text, re.I)
    ]
    return ["💡 **In plain words:** " + " ".join(hits[:3])] if hits else []


# --- next questions --------------------------------------------------------------------


def suggestions(intent: str, plain: str | None, ips: list[str], tids: list[str],
                status: str | None, asked: str = "") -> list[str]:  # fmt: skip
    """2-3 follow-up questions the assistant can answer (never the one just asked)."""
    from src.soc.correlate import is_unknown

    ip = next((i for i in ips if not is_unknown(i)), None)
    tid = tids[0] if tids else None
    s: list[str] = []
    if intent == "status":
        if status in ("ongoing", "happened"):
            s += [
                f"how do I stop {plain}?",
                f"what did {ip} do?" if ip else "who is attacking us?",
            ]
        s += [f"explain {plain}", "what attacks are happening today?"]
    elif intent in ("explain", "explain_technique"):
        s += [f"is there any {plain} happening?", f"how do I stop {plain}?"]
        if tid:
            s.append(f"which incidents used {tid}?")
    elif intent == "defend":
        s += [f"is there any {plain} happening?", f"explain {plain}"]
    elif intent in ("incidents", "top"):
        s += [f"what did {ip} do?" if ip else "is there any brute force happening?",
              "how many alerts in the last hour?", "who is attacking us?"]  # fmt: skip
    elif intent == "ip":
        if tid:
            s += [f"explain {tid}", f"which incidents used {tid}?"]
        s.append("who is attacking us?")
    elif intent == "mitre":
        s += [f"explain {tid}" if tid else "what is lateral movement?",
              f"is there any {tid} happening?" if tid else "who is attacking us?"]  # fmt: skip
    elif intent == "count":
        s += ["what attacks are happening today?", "who is attacking us?"]
    else:
        s += ["what attacks are happening today?", "what is a brute force attack?",
              "is there any defense evasion happening?"]  # fmt: skip
    done = asked.strip().lower().rstrip("?")
    out = list(dict.fromkeys(x for x in s if x and "None" not in x
                             and x.lower().rstrip("?") != done))  # fmt: skip
    if intent != "incidents" and len(out) < 2:
        out.append("what attacks are happening today?")
    return out[:3]


# --- layout ------------------------------------------------------------------------------


def compose(head: str, body: list[str], style: str, step_lines: list[str],
            extra: list[str], note: str | None = None) -> str:  # fmt: skip
    """Answer line first; urgent: steps before the evidence; otherwise after it."""
    parts = [head]
    if note:
        parts.append(note)
    todo = ["🧭 **What to do now:**", *step_lines] if step_lines else []
    if style == "urgent":
        parts += todo + body + extra
    else:
        parts += body + extra + todo
    return "\n".join(parts)


def tidy(text: str) -> str:
    """Grammar/format clean-up applied to every answer line."""
    text = re.sub(
        r"[ \t]{2,}(?!\n)", " ", text
    )  # doubled spaces (markdown "  \n" kept)
    text = re.sub(r"\b(\w+) \1\b", r"\1", text)  # doubled words ("the the")
    text = re.sub(r"[ \t]+([.,;:!?])(?=\s|$)", r"\1", text)  # space before punctuation
    return re.sub(r"(?<![.\d])\.\.(?![.\d])", ".", text)  # ".." but not "..." or IPs
