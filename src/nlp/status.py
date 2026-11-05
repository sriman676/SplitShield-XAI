"""'Is <attack / ATT&CK tactic> happening?' - answered from the stored logs.

Combines every log source the SOC has ingested (detector flows, Suricata, Zeek,
firewall, Windows Security, syslog) and classifies the evidence:

  * ongoing      - strong evidence and the latest matching event is within
                   ``ONGOING_S`` (owner 2026-10-14: 1 hour) of now;
  * happened     - strong evidence, but the activity stopped earlier;
  * possible     - only low-confidence evidence (low-tier detector alerts whose
                   techniques are candidates, not confirmed);
  * not_seen     - nothing in the logs checked (and which logs those were).

Strong evidence = a confirmed detector alert, or a log event (Windows, Suricata,
...) that carries a matching ATT&CK technique. When the evidence is thin or mixed
(only possible, or one source with a handful of events), the answer ends with
follow-up questions so the analyst decides how to dig further.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

ONGOING_S = 3600
THIN = 3  # fewer strong events than this from a single source -> ask a follow-up


def _named(ev: dict[str, Any], family: str | None) -> bool:
    """Named attacks sharing a generic technique (Heartbleed = T1190 like any web
    exploit) need their name in the event; everything else matches on technique."""
    import re

    from src.knowledge.attack_details import FAMILY_KEYWORDS

    pat = FAMILY_KEYWORDS.get(family or "")
    if pat is None:
        return True
    text = f"{ev.get('detail', '')} {ev.get('kind', '')} {ev.get('title', '')}"
    return re.search(pat, text, re.I) is not None


def _match(ev: dict[str, Any], techs: set[str], family: str | None) -> str | None:
    """'strong', 'weak' or None for one stored event."""
    if ev.get("source") == "detector":
        label = ev.get("label")
        tier = ev.get("tier", "confirmed")
        hit_label = family is not None and label == family
        hit_tech = bool(techs & set(ev.get("techniques") or []))
        cand = bool(techs & set(ev.get("candidate_techniques") or []))
        if tier == "confirmed" and (hit_label or hit_tech):
            return "strong"
        if tier == "low" and (hit_label or cand):
            return "weak"
        return None
    hit = techs & set(ev.get("techniques") or [])
    return "strong" if hit and _named(ev, family) else None


def check(
    events: list[dict[str, Any]],
    techniques: list[str],
    family: str | None = None,
    since: float | None = None,
    now: float | None = None,
    include_low: bool = False,
) -> dict[str, Any]:
    """Evidence summary for a family and/or technique set over ``events``."""
    now = time.time() if now is None else now
    window = [e for e in events if since is None or e.get("ts", 0) >= since]
    span = (
        (min(e.get("ts", now) for e in window), max(e.get("ts", now) for e in window))
        if window
        else None
    )
    checked = sorted({str(e.get("source", "?")) for e in window})
    return _summary(
        window, techniques, family, now, include_low, checked, len(window), span
    )


def check_store(
    techniques: list[str],
    family: str | None = None,
    since: float | None = None,
    now: float | None = None,
    include_low: bool = False,
) -> dict[str, Any]:
    """``check`` over the SOC's whole store, through its index (src/soc/eventindex.py):
    the evidence summary is kept in the index and updated with new events only, so
    the answer takes milliseconds at millions of events."""
    from src.soc import eventindex

    now = time.time() if now is None else now
    st = eventindex.stats(since)
    acc = eventindex.status_state(techniques, family, since)
    span = (st["first"], st["last"]) if st["events"] else None
    return _render(
        acc, set(techniques), now, include_low, st["sources"], st["events"], span
    )


def new_state() -> dict[str, Any]:
    """Running evidence summary: per class, count / first / last and the counters in
    first-seen order (what ``_render`` needs; JSON-safe, so the index can keep it)."""
    return {
        c: {"n": 0, "first": None, "last": None, "src": {}, "ips": {}, "techs": {}}
        for c in ("strong", "weak")
    }


def accumulate(
    state: dict[str, Any],
    events: list[dict[str, Any]],
    techniques: list[str],
    family: str | None,
) -> dict[str, Any]:
    """Add ``events`` (in store order) to a running summary."""
    techs = set(techniques)
    for e in events:
        m = _match(e, techs, family)
        if m is None:
            continue
        a = state[m]
        a["n"] += 1
        ts = e["ts"]
        a["first"] = ts if a["first"] is None else min(a["first"], ts)
        a["last"] = ts if a["last"] is None else max(a["last"], ts)
        src = str(e.get("source"))
        a["src"][src] = a["src"].get(src, 0) + 1
        if ip := e.get("src_ip"):
            a["ips"][ip] = a["ips"].get(ip, 0) + 1
        for t in e.get("techniques") or e.get("candidate_techniques") or []:
            if not techs or t in techs:
                a["techs"][t] = a["techs"].get(t, 0) + 1
    return state


def _summary(
    events: list[dict[str, Any]],
    techniques: list[str],
    family: str | None,
    now: float,
    include_low: bool,
    sources_checked: list[str],
    events_checked: int,
    span: tuple[float, float] | None,
) -> dict[str, Any]:
    acc = accumulate(new_state(), events, techniques, family)
    return _render(
        acc, set(techniques), now, include_low, sources_checked, events_checked, span
    )


def _render(
    acc: dict[str, Any],
    techs: set[str],
    now: float,
    include_low: bool,
    sources_checked: list[str],
    events_checked: int,
    span: tuple[float, float] | None,
) -> dict[str, Any]:
    strong, weak = acc["strong"], acc["weak"]
    used = [strong] + ([weak] if include_low else [])
    used = [u for u in used if u["n"]]
    out: dict[str, Any] = {
        "strong": strong["n"],
        "weak": weak["n"],
        "sources_checked": sources_checked,
        "events_checked": events_checked,
        "span": span,
    }
    if used:
        src: Counter[str] = Counter()
        ips: Counter[str] = Counter()
        tcount: Counter[str] = Counter()
        for u in used:  # strong first, then weak: the first-seen order of the list
            src.update(u["src"])
            ips.update(u["ips"])
            tcount.update(u["techs"])
        out |= {
            "first": min(u["first"] for u in used),
            "last": max(u["last"] for u in used),
            "by_source": dict(src),
            "top_ips": [ip for ip, _ in ips.most_common(5)],
            "techniques": dict(tcount),
        }
    if strong["n"]:
        out["status"] = "ongoing" if now - out["last"] <= ONGOING_S else "happened"
    elif weak["n"]:
        out["status"] = "possible"
    else:
        out["status"] = "not_seen"
    thin = out["status"] == "possible" or (
        bool(strong["n"]) and strong["n"] < THIN and len(out.get("by_source", {})) == 1
    )
    out["needs_follow_up"] = thin
    return out


def _ago(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 90:
        return f"{s} s ago"
    if s < 5400:
        return f"{s // 60} min ago"
    if s < 172800:
        return f"{s / 3600:.1f} h ago"
    return f"{s / 86400:.1f} days ago"


def _is(subject: str) -> str:
    """Verb agreement: 'web application attacks are', 'brute force is'."""
    head = subject.split(" (")[0].strip().lower()
    return "are" if head.endswith(("attacks", "scans")) else "is"


def describe(subject: str, res: dict[str, Any], now: float | None = None) -> list[str]:
    """Answer lines (markdown) for ``check`` results about ``subject``."""
    from src.soc.safe import md  # sources / IPs come from logs

    now = time.time() if now is None else now
    st = res["status"]
    when = time.strftime
    lines: list[str] = []
    if st == "ongoing":
        lines.append(
            f"🚨 `ONGOING` **Yes - {subject} {_is(subject)} happening now.** Last matching activity "
            f"{_ago(now - res['last'])} ({when('%Y-%m-%d %H:%M', time.localtime(res['last']))}); "
            f"started {when('%Y-%m-%d %H:%M', time.localtime(res['first']))}."
        )
    elif st == "happened":
        lines.append(
            f"🕒 `HAPPENED EARLIER` **Yes - {subject} happened, but it is not active now.** Seen from "
            f"{when('%Y-%m-%d %H:%M', time.localtime(res['first']))} to "
            f"{when('%Y-%m-%d %H:%M', time.localtime(res['last']))} "
            f"(last activity {_ago(now - res['last'])}; 'ongoing' means within the last hour)."
        )
    elif st == "possible":
        lines.append(
            f"⚠️ `POSSIBLE` **Possibly - only low-confidence signs of {subject}.** "
            f"{res['weak']} low-tier detector alert(s) point to it, but nothing confirmed it."
        )
    else:
        lines.append(
            f"✅ `NOT SEEN` **No - I found no evidence of {subject}** in the logs I have."
        )
    if res.get("by_source"):
        src = ", ".join(f"{md(k)}: {v}" for k, v in sorted(res["by_source"].items()))
        lines.append(
            f"- 🔎 Evidence: {res['strong']} confirmed + {res['weak']} low-confidence event(s) ({src})."
        )
    if res.get("techniques"):
        lines.append(
            "- 🧬 Techniques seen: "
            + ", ".join(f"{t} ({n})" for t, n in res["techniques"].items())
        )
    if res.get("top_ips"):
        lines.append("- 🌐 Sources: " + ", ".join(md(x) for x in res["top_ips"]))
    checked = (
        ", ".join(md(x) for x in res["sources_checked"])
        or "none - no logs ingested yet"
    )
    span = res.get("span")
    rng = (
        f" from {when('%Y-%m-%d %H:%M', time.localtime(span[0]))} to {when('%Y-%m-%d %H:%M', time.localtime(span[1]))}"
        if span
        else ""
    )
    lines.append(f"- 📁 Logs checked: {checked} ({res['events_checked']} events{rng}).")
    return lines


def follow_ups(subject: str, res: dict[str, Any], had_window: bool) -> list[str]:
    """Questions to offer when the evidence is thin or mixed (analyst decides)."""
    if not res.get("needs_follow_up") and res["status"] != "not_seen":
        return []
    qs = []
    if res["weak"]:
        qs.append(f"is there any {subject}, including low-confidence alerts?")
    if had_window and res["status"] in ("not_seen", "possible"):
        qs.append(f"is there any {subject} at any time in the logs?")
    if res.get("top_ips"):
        qs.append(f"what did {res['top_ips'][0]} do?")
    return qs
