"""SOC reports (owner 2026-10-15; report builder 2026-10-16): on request or on a
schedule, as Word (.docx), Markdown and HTML, built from ALL models through the shared
risk board.

    python -m src.soc.reports --period today|yesterday|this_week|week|this_month|last<N>d
    python -m src.soc.reports --from "2026-10-01 00:00" --to "2026-10-08 00:00"
    python -m src.soc.reports --period incident --id <incident id>

What goes in a report comes from the GLOBAL TEMPLATE (``report_template.json`` next to
the event store; edited on the dashboard's Reports page by users with the ``reports``
permission): title, intro text, which sections in which order, how many top entities,
which file formats. Every single report can override any of it, and a finished report
can still be edited and saved again (``save_markdown``).

Sections (see SECTIONS; the default template keeps the original seven):
  summary, incidents, findings (by model family), entities (top IPs / users / hosts),
  attack (ATT&CK coverage), todo (what to do), health (automatic parts and
  self-improvement), allowlisted; optional: timeline, sources, reviews, models, audit.
Allow-listed alerts are listed separately and not counted in the totals (owner rule).

Schedules (``report_schedules.json``): name, period, weekdays, time, optional section
list and formats, optional folder. The autopilot calls ``run_due`` every cycle; a
schedule runs once per day on its first cycle after its time. Reports are saved to
results/reports/ unless a schedule names another folder.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from utils.paths import RESULTS_DIR

OUT = RESULTS_DIR / "reports"
Event = dict[str, Any]
DAY = 86400.0
FORMATS = {"docx": "Word (.docx)", "md": "Markdown (.md)", "html": "HTML page (.html)"}
PERIODS = {"today": "Today", "yesterday": "Yesterday", "this_week": "This week (since Monday)",
           "week": "Last 7 days", "this_month": "This month", "last30d": "Last 30 days"}  # fmt: skip


# --- periods ---------------------------------------------------------------------------


def _midnight(ts: float) -> float:
    lt = time.localtime(ts)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))


def period_bounds(period: str, now: float | None = None) -> tuple[float, float, str]:
    """(start, end, label) of a named period."""
    now = time.time() if now is None else now
    lt = time.localtime(now)
    today = _midnight(now)
    day = time.strftime("%Y-%m-%d", lt)
    if period == "today":
        return today, now, day
    if period == "yesterday":
        return today - DAY, today, time.strftime("%Y-%m-%d", time.localtime(today - DAY))
    if period == "this_week":
        start = _midnight(today - lt.tm_wday * DAY + 3 * 3600)  # DST-safe Monday 00:00
        return start, now, f"week of {time.strftime('%Y-%m-%d', time.localtime(start))}"
    if period == "this_month":
        start = time.mktime((lt.tm_year, lt.tm_mon, 1, 0, 0, 0, 0, 0, -1))
        return start, now, time.strftime("%B %Y", lt)
    if period == "week":
        return now - 7 * DAY, now, f"7 days to {day}"
    if m := re.fullmatch(r"last(\d+)d", period):
        n = int(m.group(1))
        return now - n * DAY, now, f"{n} days to {day}"
    raise ValueError(f"unknown period {period!r}")


def parse_when(text: str) -> float:
    """'2026-10-01', '2026-10-01 14:30' or '2026-10-01T14:30:00' -> local timestamp."""
    text = text.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(text, fmt))
        except ValueError:
            continue
    raise ValueError(f"not a date/time: {text!r} (use YYYY-MM-DD or YYYY-MM-DD HH:MM)")


def range_label(start: float, end: float) -> str:
    whole = _midnight(start) == start and _midnight(end) == end
    fmt = "%Y-%m-%d" if whole else "%Y-%m-%d %H:%M"
    if whole and end - start <= DAY + 3600:
        return time.strftime("%Y-%m-%d", time.localtime(start))
    return f"{time.strftime(fmt, time.localtime(start))} to {time.strftime(fmt, time.localtime(end))}"


# --- template --------------------------------------------------------------------------


def _state_dir() -> Path:
    from src.soc import engine

    return Path(engine.EVENTS).parent


def template_path() -> Path:
    return _state_dir() / "report_template.json"


def schedules_path() -> Path:
    return _state_dir() / "report_schedules.json"


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), "utf-8")
    tmp.replace(p)


DEFAULT_TEMPLATE: dict[str, Any] = {
    "title": "SplitShield-XAI SOC report",
    "intro": "",
    "sections": ["summary", "incidents", "findings", "entities", "attack", "todo",
                 "health", "allowlisted"],  # fmt: skip
    "top_n": 5,
    "max_details": 10,
    "formats": ["docx", "md", "html"],
}


def load_template() -> dict[str, Any]:
    """The global template (defaults for anything missing or unknown)."""
    t = dict(DEFAULT_TEMPLATE)
    try:
        saved = json.loads(template_path().read_text("utf-8"))
        t.update({k: v for k, v in saved.items() if k in DEFAULT_TEMPLATE or k == "changed"})
    except (OSError, ValueError):
        pass
    return check_template(t)


def check_template(t: dict[str, Any]) -> dict[str, Any]:
    """Validate a template or a per-report override (raises ValueError)."""
    t = dict(DEFAULT_TEMPLATE) | dict(t)
    if bad := [s for s in t["sections"] if s not in SECTIONS]:
        raise ValueError(f"unknown section(s): {', '.join(map(str, bad))}")
    if bad := [f for f in t["formats"] if f not in FORMATS]:
        raise ValueError(f"unknown format(s): {', '.join(map(str, bad))}")
    if not t["formats"]:
        raise ValueError("pick at least one file format")
    t["sections"] = list(dict.fromkeys(t["sections"]))
    t["top_n"] = max(1, min(int(t["top_n"]), 50))
    t["max_details"] = max(1, min(int(t["max_details"]), 100))
    t["title"] = str(t["title"]).strip()[:120] or DEFAULT_TEMPLATE["title"]
    t["intro"] = str(t["intro"])[:4000]
    return t


def save_template(t: dict[str, Any], who: str = "") -> dict[str, Any]:
    from src.soc import audit

    t = check_template({k: v for k, v in t.items() if k in DEFAULT_TEMPLATE})
    _write_json(template_path(), t | {"changed": time.strftime("%Y-%m-%d %H:%M")})
    audit.record(who or "system", "report_template", {"sections": t["sections"],
                                                      "formats": t["formats"]})  # fmt: skip
    return t


# --- sections --------------------------------------------------------------------------


def _muted(obj: dict[str, Any]) -> str | None:
    """Allow-list reason when the owner allow-listed this entity (else None)."""
    if obj.get("allowlisted"):
        return str(obj["allowlisted"])
    try:
        from src.soc import allowlist  # built alongside (Fork D); optional
    except ImportError:
        return None
    try:
        muted, reason = allowlist.is_muted(obj)
    except Exception:  # noqa: BLE001 - a broken allow-list must not stop a report
        return None
    return str(reason) if muted else None


def _v(x: Any, limit: int = 200) -> str:
    """A value from log content, safe inside the report's Markdown, Word and HTML (pentest
    re-check 2026-10-23): one line, no table pipes, no link / image syntax, cut to
    ``limit``. Plain text in every format - no escape characters to show."""
    s = " ".join(str("" if x is None else x).split())[:limit]
    return s.replace("|", "/").replace("![", "! [").replace("](", "] (").replace("<", "‹").replace(">", "›")


def _fmt_t(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _s_summary(cx: dict[str, Any]) -> list[str]:
    return [f"**{cx['head']}**", ""]


def _s_incidents(cx: dict[str, Any]) -> list[str]:
    incs = cx["incs"]
    md = ["## Incidents", ""]
    if not incs:
        return md + ["No incidents in this period.", ""]
    from src.soc import cases

    case_of = cases.all_cases()
    md += ["| Level | Severity | Entity | Attack path | Sensors | Other models | Allow-listed | Case |",
           "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for i in incs:
        path = " → ".join(f"{t['tactic']}: {t['name']}" for t in i["techniques"])
        md.append(f"| {i['level']} | {i['severity']}/10 | {_v(i['src_ip'])} | {_v(path, 400)} | "
                  f"{_v(', '.join(i['sources']))} | {len(i.get('linked_findings', []))} | "
                  f"{_muted(i) or ''} | "
                  f"{cases.STATUSES[case_of.get(i['id'], {}).get('status', 'new')]} |")  # fmt: skip
    md.append("")
    for i in incs:
        if i.get("linked_findings"):
            md += [f"**{_v(i['src_ip'])}**, other models agree (severity +{i.get('board_boost', 0)}):"]
            md += [f"- {_v(r, 400)}" for r in i["linked_findings"]] + [""]
    return md


def _s_findings(cx: dict[str, Any]) -> list[str]:
    from src.soc import riskboard

    counted = cx["counted"]
    fam = Counter(riskboard.family_of(e) for e in counted)
    md = ["## Findings by model family", "", "| Family | Alerts |", "|---|---|"]
    md += [f"| {_v(k)} | {v:,} |" for k, v in fam.most_common()] or ["| none | 0 |"]
    cross = Counter(str(e.get("kind")) for e in counted
                    if e.get("source") in ("phishing", "intel", "malware"))  # fmt: skip
    if cross:
        md += ["", "Cross-checks: "
               + ", ".join(f"{_v(k.replace('_', ' '))} {v}" for k, v in cross.items()) + "."]  # fmt: skip
    return md + [""]


def _s_entities(cx: dict[str, Any]) -> list[str]:
    n = cx["t"]["top_n"]
    tops = [(name, k, v) for key, name in (("src_ip", "IP"), ("user", "User"), ("host", "Host"))
            for k, v in Counter(str(e[key]) for e in cx["counted"] if e.get(key)).most_common(n)]  # fmt: skip
    md = ["## Top entities", ""]
    if not tops:
        return md + ["No entities with alerts.", ""]
    md += ["| Type | Entity | Alerts |", "|---|---|---|"]
    return md + [f"| {t} | {_v(k)} | {v:,} |" for t, k, v in tops] + [""]


def _s_attack(cx: dict[str, Any]) -> list[str]:
    from src.soc.correlate import technique_info

    techs = Counter(t.split(".")[0] for e in cx["counted"] for t in e.get("techniques", []))
    md = ["## ATT&CK coverage", ""]
    if not techs:
        return md + ["No ATT&CK techniques observed.", ""]
    md += ["| Technique | Name | Tactic | Alerts |", "|---|---|---|---|"]
    for t, n in techs.most_common(15):
        name, tactic = technique_info(t)
        md.append(f"| {t} | {name} | {tactic} | {n:,} |")
    return md + [""]


def _s_todo(cx: dict[str, Any]) -> list[str]:
    serious = [i for i in cx["live_incs"] if i["level"] in ("critical", "high")]
    md = ["## What to do", ""]
    if not serious:
        return md + ["Nothing high or critical: keep monitoring; review the alerts above "
                     "when convenient.", ""]  # fmt: skip
    from src.knowledge.explain import incident_markdown

    for i in serious[: cx["t"]["max_details"]]:
        md += [incident_markdown(i, clean=_v), ""]
    return md


def _s_health(cx: dict[str, Any]) -> list[str]:
    md = ["## System health", ""]
    try:
        from src.soc import controller

        for k, v in controller.components().items():
            md.append(f"- {'✅' if v['ok'] else '❌'} {k.replace('_', ' ')}: {v['detail']}")
        dec = [h for h in controller.history(200) if cx["start"] <= _hts(h) < cx["end"]]
        if dec:
            md += ["", "Self-improvement decisions:"] + [
                f"- {h['ts']} {h['model']}: {h.get('decision', '')}" for h in dec[-10:]
            ]
    except Exception as exc:  # noqa: BLE001 - health is informative only
        md.append(f"- health unavailable ({type(exc).__name__})")
    return md + [""]


def _hts(h: dict[str, Any]) -> float:
    try:
        return time.mktime(time.strptime(str(h["ts"])[:19], "%Y-%m-%dT%H:%M:%S"))
    except (KeyError, ValueError):
        return 0.0


def _s_allowlisted(cx: dict[str, Any]) -> list[str]:
    muted = cx["muted_alerts"]
    if not muted:
        return []
    md = ["## Allow-listed (not counted above)", "",
          f"{len(muted):,} alerts involved entities you allow-listed:"]  # fmt: skip
    for reason, n in Counter(_muted(e) for e in muted).most_common(cx["t"]["top_n"]):
        md.append(f"- {_v(reason)}: {n:,}")
    return md + [""]


def _s_timeline(cx: dict[str, Any]) -> list[str]:
    """Alerts per hour (periods up to 2 days) or per day."""
    hourly = cx["end"] - cx["start"] <= 2 * DAY
    fmt = "%Y-%m-%d %H:00" if hourly else "%Y-%m-%d"
    per = Counter(time.strftime(fmt, time.localtime(float(e.get("ts") or 0))) for e in cx["counted"])
    md = ["## Alerts over time", ""]
    if not per:
        return md + ["No alerts in this period.", ""]
    md += [f"| {'Hour' if hourly else 'Day'} | Alerts |", "|---|---|"]
    return md + [f"| {k} | {per[k]:,} |" for k in sorted(per)] + [""]


def _s_sources(cx: dict[str, Any]) -> list[str]:
    """What was collected: events and alerts per log source."""
    ev = Counter(str(e.get("source") or "unknown") for e in cx["in_period"])
    al = Counter(str(e.get("source") or "unknown") for e in cx["counted"])
    md = ["## Data collected", "", f"{sum(ev.values()):,} events stored in this period.", ""]
    if ev:
        md += ["| Source | Events | Alerts |", "|---|---|---|"]
        md += [f"| {_v(k)} | {v:,} | {al.get(k, 0):,} |" for k, v in ev.most_common()]
    return md + [""]


def _s_reviews(cx: dict[str, Any]) -> list[str]:
    """Analyst verdicts on detector alerts (they teach the models)."""
    from src.soc import engine, rcl

    rows = [r for r in engine.read_jsonl(rcl.FEEDBACK)
            if cx["start"] <= float(r.get("ts") or 0) < cx["end"]]  # fmt: skip
    md = ["## Analyst reviews", ""]
    if not rows:
        return md + ["No alerts were reviewed in this period.", ""]
    real = sum(bool(r.get("correct")) for r in rows)
    md += [f"{len(rows):,} alerts reviewed: {real:,} real, {len(rows) - real:,} false alarms "
           f"(precision of reviewed alerts {real / len(rows):.1%}).", "",
           "| Detector class | Real | False alarm |", "|---|---|---|"]  # fmt: skip
    by: dict[str, list[int]] = {}
    for r in rows:
        by.setdefault(str(r.get("pred")), [0, 0])[0 if r.get("correct") else 1] += 1
    return md + [f"| {k} | {a:,} | {b:,} |" for k, (a, b) in sorted(by.items())] + [""]


def _s_models(cx: dict[str, Any]) -> list[str]:
    """Which models ran, evolving or frozen, and what changed in the period."""
    md = ["## Models", ""]
    try:
        from src.soc import controller, evolution

        ev = evolution.load()
        md += [f"Evolution: **{ev['mode']}**"
               + (f"; frozen: {', '.join(ev['frozen'])}" if ev.get("frozen") else "") + ".", ""]  # fmt: skip
        dec = [h for h in controller.history(2000) if cx["start"] <= _hts(h) < cx["end"]]
        kept = Counter(h["model"] for h in dec if "promot" in str(h.get("decision", "")).lower())
        tried = Counter(h["model"] for h in dec)
        md += ["| Model | Update cycles | New version kept |", "|---|---|---|"]
        md += [f"| {m} | {tried.get(m, 0)} | {kept.get(m, 0)} |" for m in evolution.components()]
    except Exception as exc:  # noqa: BLE001 - informative only
        md.append(f"Model status unavailable ({type(exc).__name__}).")
    return md + [""]


def _s_audit(cx: dict[str, Any]) -> list[str]:
    """Who changed what (sign-ins counted, changes listed)."""
    from src.soc import audit

    rows = [r for r in audit.read() if cx["start"] <= float(r.get("ts") or 0) < cx["end"]]
    signs = [r for r in rows if r.get("action") == "sign_in"]
    changes = [r for r in rows if r.get("action") != "sign_in"]
    md = ["## User activity (audit log)", "",
          f"{sum(bool(r.get('ok')) for r in signs):,} sign-ins, "
          f"{sum(not r.get('ok') for r in signs):,} failed sign-ins, {len(changes):,} changes. "
          f"Audit chain: {'intact' if audit.verify()['ok'] else 'BROKEN - investigate'}.", ""]  # fmt: skip
    if changes:
        md += ["| When | Who | Action |", "|---|---|---|"]
        md += [f"| {_v(r.get('when'))} | {_v(r.get('who'))} | {_v(str(r.get('action')).replace('_', ' '))} |"
               for r in changes[-cx["t"]["max_details"] * 5:]]  # fmt: skip
    return md + [""]


# key -> (title shown when picking, builder, short description)
SECTIONS: dict[str, tuple[str, Callable[[dict[str, Any]], list[str]], str]] = {
    "summary": ("Summary", _s_summary, "one line: incidents, critical/high, what is ongoing"),
    "incidents": ("Incidents", _s_incidents, "level, entity, ATT&CK path, sensors, other models"),
    "findings": ("Findings by model family", _s_findings, "alerts per detector family + cross-checks"),
    "entities": ("Top entities", _s_entities, "IPs, users and hosts behind the most alerts"),
    "attack": ("ATT&CK coverage", _s_attack, "techniques and tactics seen"),
    "todo": ("What to do", _s_todo, "the write-up of every high or critical incident"),
    "timeline": ("Alerts over time", _s_timeline, "alerts per hour or per day"),
    "sources": ("Data collected", _s_sources, "events and alerts per log source"),
    "reviews": ("Analyst reviews", _s_reviews, "real vs false alarm verdicts"),
    "models": ("Models", _s_models, "evolving / frozen, update cycles, versions kept"),
    "health": ("System health", _s_health, "automatic parts, self-improvement decisions"),
    "audit": ("User activity", _s_audit, "sign-ins and changes from the audit log"),
    "allowlisted": ("Allow-listed", _s_allowlisted, "muted alerts, not counted above"),
    "compliance": ("Compliance evidence", lambda cx: _s_compliance(cx),
                   "NIST CSF 2.0 / ISO 27001:2022 controls with evidence, partial or gap"),
}


def _s_compliance(cx: dict[str, Any]) -> list[str]:
    from src.soc import compliance

    return compliance.section(cx)


# --- build -----------------------------------------------------------------------------


def build(period: str = "today", incident_id: str | None = None,
          events: list[Event] | None = None, now: float | None = None,
          start: float | None = None, end: float | None = None,
          template: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    """Report content (Markdown text + facts) for a period, a start/end range or one
    incident. ``template`` overrides the global template for this report only."""
    from src.soc import engine, places, riskboard
    from src.soc.correlate import correlate

    t = check_template(load_template() | (template or {}))
    now = time.time() if now is None else now
    if events is None:  # the SOC's own store: through its index (any size)
        all_incs = [{k: v for k, v in i.items() if k != "allowlisted"} for i in engine.incidents()]
    else:
        board = riskboard.board_for(events, scale=places.board_scale())
        all_incs = correlate(events, board=board)
    if incident_id:
        incs = [i for i in all_incs if i["id"] == incident_id]
        if not incs:
            raise ValueError(f"no incident {incident_id!r}")
        start, end, label = incs[0]["start"], incs[0]["end"], f"incident {incident_id}"
    elif start is not None or end is not None:
        start = float(start if start is not None else 0.0)
        end = float(end if end is not None else now)
        if end <= start:
            raise ValueError("the end must be after the start")
        label = range_label(start, end)
    else:
        start, end, label = period_bounds(period, now)
    assert start is not None and end is not None
    if not incident_id:
        incs = [i for i in all_incs if i["end"] >= start and i["start"] < end]
    if events is None:
        in_period = engine.events_between(start, end)
    else:
        in_period = [e for e in events if start <= float(e.get("ts") or 0) < end]
    alerts = [e for e in in_period if e.get("techniques")]
    muted_alerts = [e for e in alerts if _muted(e)]
    counted = [e for e in alerts if not _muted(e)]
    live_incs = [i for i in incs if not _muted(i)]
    lv = Counter(i["level"] for i in live_incs)
    ongoing = [i for i in live_incs if now - i["end"] < 3600]
    if live_incs:
        head = (f"{len(live_incs)} incident{'s' if len(live_incs) != 1 else ''} "
                f"({lv['critical']} critical, {lv['high']} high) and {len(counted):,} alerts.")  # fmt: skip
    else:
        head = f"No incidents; {len(counted):,} alerts."
    if ongoing:
        head += " Ongoing now: " + "; ".join(
            f"{', '.join(x['name'] for x in i['techniques'][:2])} from {_v(i['src_ip'])}"
            for i in ongoing[:3]
        ) + "."
    cx = {"t": t, "start": start, "end": end, "now": now, "incs": incs, "live_incs": live_incs,
          "in_period": in_period, "counted": counted, "muted_alerts": muted_alerts, "head": head}  # fmt: skip
    md = [f"# {t['title']}: {label}", "",
          f"_Generated {_fmt_t(time.time())} · place profile: {places.current()['name'].replace('_', ' ')}_", ""]  # fmt: skip
    if t["intro"].strip():
        md += [t["intro"].strip(), ""]
    for key in t["sections"]:
        md += SECTIONS[key][1](cx)
    return {"label": label, "headline": head, "markdown": "\n".join(md), "start": start,
            "end": end, "sections": t["sections"], "formats": t["formats"],
            "incidents": len(live_incs), "alerts": len(counted), "muted": len(muted_alerts)}  # fmt: skip


# --- Word and HTML ---------------------------------------------------------------------

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_TABLE_RULE = re.compile(r"^\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")


def _runs(par: Any, text: str) -> None:
    """Add text with **bold** spans (other Markdown marks are dropped)."""
    text = text.replace("_", "") if text.startswith("_") and text.endswith("_") else text
    pos = 0
    for m in _BOLD.finditer(text):
        par.add_run(text[pos : m.start()])
        par.add_run(m.group(1)).bold = True
        pos = m.end()
    par.add_run(text[pos:])


def _blocks(markdown: str) -> list[tuple[str, Any]]:
    """The report's Markdown subset as blocks: (h1..h4 | li | p | table, content)."""
    out: list[tuple[str, Any]] = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|") and i + 1 < len(lines) and _TABLE_RULE.match(lines[i + 1]):
            rows = [ln]
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            width = len(cells[0])
            out.append(("table", [(r + [""] * width)[:width] for r in cells]))
            continue
        if m := re.match(r"^(#{1,4})\s+(.*)", ln):
            out.append((f"h{len(m.group(1))}", m.group(2)))
        elif re.match(r"^\s*([-*]|\d+\.)\s+", ln):
            out.append(("li", re.sub(r"^\s*([-*]|\d+\.)\s+", "", ln)))
        elif ln.strip():
            out.append(("p", ln))
        i += 1
    return out


def to_docx(markdown: str, path: Path) -> Path:
    """Write the report's Markdown subset (headings, bullets, tables, bold) as .docx."""
    from docx import Document

    doc = Document()
    for kind, val in _blocks(markdown):
        if kind == "table":
            table = doc.add_table(rows=len(val), cols=len(val[0]))
            table.style = "Light Grid Accent 1"
            for r, row in enumerate(val):
                for c, cell in enumerate(row):
                    table.cell(r, c).text = cell.replace("**", "")
        elif kind.startswith("h"):
            doc.add_heading(val.replace("**", ""), level=min(int(kind[1]), 3))
        elif kind == "li":
            _runs(doc.add_paragraph(style="List Bullet"), val)
        else:
            _runs(doc.add_paragraph(), val)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    return path


def _inline_html(text: str) -> str:
    text = text[1:-1] if len(text) > 1 and text.startswith("_") and text.endswith("_") else text
    s = html.escape(text)
    s = _BOLD.sub(r"<strong>\1</strong>", s)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", s)


_CSS = """body{font-family:Segoe UI,system-ui,sans-serif;max-width:1100px;margin:24px auto;
padding:0 16px;color:#1b2430;background:#fff;line-height:1.45}h1{color:#1f4e8c}
h2{color:#1f4e8c;border-bottom:1px solid #d6dfeb;padding-bottom:4px;margin-top:28px}
table{border-collapse:collapse;margin:8px 0;width:100%;font-size:14px}
th,td{border:1px solid #d6dfeb;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#eef3fa}code{background:#f1f3f6;padding:0 3px}
@media (prefers-color-scheme:dark){body{background:#12161c;color:#e3e8ef}
th{background:#1d2633}th,td{border-color:#2c3644}h1,h2{color:#8fb7ef}code{background:#1d2633}}"""


def to_html(markdown: str, path: Path) -> Path:
    """A self-contained HTML page (no external files; opens in any browser)."""
    blocks = _blocks(markdown)
    title = next((v for k, v in blocks if k == "h1"), "SOC report")
    body: list[str] = []
    in_list = False
    for kind, val in blocks:
        if kind != "li" and in_list:
            body.append("</ul>")
            in_list = False
        if kind == "table":
            head, *rows = val
            body.append("<table><thead><tr>" + "".join(f"<th>{_inline_html(c)}</th>" for c in head)
                        + "</tr></thead><tbody>"
                        + "".join("<tr>" + "".join(f"<td>{_inline_html(c)}</td>" for c in r) + "</tr>"
                                  for r in rows) + "</tbody></table>")  # fmt: skip
        elif kind.startswith("h"):
            body.append(f"<{kind}>{_inline_html(val)}</{kind}>")
        elif kind == "li":
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{_inline_html(val)}</li>")
        else:
            body.append(f"<p>{_inline_html(val)}</p>")
    if in_list:
        body.append("</ul>")
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(title.replace('**', ''))}</title><style>{_CSS}</style></head>"
            f"<body>{''.join(body)}</body></html>")  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, "utf-8")
    return path


def save_markdown(markdown: str, stem: str, formats: list[str] | None = None,
                  folder: Path | None = None) -> dict[str, Path]:  # fmt: skip
    """Write a (possibly hand-edited) report in the chosen formats."""
    out = Path(folder) if folder else OUT
    out.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^\w.-]+", "_", stem)[:100] or "report"
    files: dict[str, Path] = {}
    for f in formats or ["docx", "md", "html"]:
        if f == "md":
            (out / f"{stem}.md").write_text(markdown, "utf-8")
            files["md"] = out / f"{stem}.md"
        elif f == "docx":
            files["docx"] = to_docx(markdown, out / f"{stem}.docx")
        elif f == "html":
            files["html"] = to_html(markdown, out / f"{stem}.html")
        else:
            raise ValueError(f"unknown format {f!r}")
    return files


def save(rep: dict[str, Any], stem: str) -> dict[str, Path]:
    return save_markdown(rep["markdown"], stem, rep.get("formats") or ["docx", "md"])


def generate(period: str = "today", incident_id: str | None = None,
             start: float | None = None, end: float | None = None,
             template: dict[str, Any] | None = None, folder: Path | None = None,
             who: str = "") -> dict[str, Any]:  # fmt: skip
    """Build + save; returns the report dict with ``files`` and ``stem``."""
    from src.soc import audit

    rep = build(period, incident_id, start=start, end=end, template=template)
    safe = re.sub(r"[^\w.-]+", "_", rep["label"])[:60]
    rep["stem"] = f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}"
    rep["files"] = save_markdown(rep["markdown"], rep["stem"], rep["formats"], folder)
    if who:
        audit.record(who, "report_made", {"period": rep["label"], "sections": rep["sections"]})
    return rep


# --- schedules -------------------------------------------------------------------------

DEFAULT_SCHEDULES: list[dict[str, Any]] = [
    {"name": "Daily", "period": "yesterday", "days": [0, 1, 2, 3, 4, 5, 6], "time": "07:00",
     "enabled": True},
    {"name": "Weekly", "period": "week", "days": [0], "time": "07:00", "enabled": True},
    {"name": "Weekly compliance", "period": "week", "days": [0], "time": "08:00", "enabled": True,
     "sections": ["summary", "compliance", "audit", "reviews", "models"], "formats": ["docx", "md"]},
]  # fmt: skip


def check_schedule(s: dict[str, Any]) -> dict[str, Any]:
    name = str(s.get("name", "")).strip()
    if not re.fullmatch(r"[\w .-]{1,40}", name):
        raise ValueError("schedule name: 1-40 letters, digits, space, dot, dash")
    period = str(s.get("period", ""))
    period_bounds(period)  # raises on an unknown period
    days = sorted({int(d) for d in s.get("days", [])})
    if not days or not all(0 <= d <= 6 for d in days):
        raise ValueError("pick at least one weekday")
    if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", str(s.get("time", ""))):
        raise ValueError("time as HH:MM, 24-hour")
    out = {"name": name, "period": period, "days": days, "time": s["time"],
           "enabled": bool(s.get("enabled", True)), "last_run": s.get("last_run", "")}  # fmt: skip
    if s.get("sections"):
        out["sections"] = check_template({"sections": s["sections"]})["sections"]
    if s.get("formats"):
        out["formats"] = check_template({"formats": s["formats"]})["formats"]
    if s.get("folder"):
        folder = Path(str(s["folder"]).strip())
        if not folder.is_absolute() or not folder.is_dir():
            raise ValueError("report folder must be an existing folder, given as a full path")
        out["folder"] = str(folder)
    return out


def load_schedules() -> list[dict[str, Any]]:
    try:
        rows = json.loads(schedules_path().read_text("utf-8"))
    except (OSError, ValueError):
        return [dict(s, last_run="") for s in DEFAULT_SCHEDULES]
    # 2026-10-17: the weekly compliance report is added once to existing installs
    flag = schedules_path().with_name("report_schedules.compliance_added")
    if not flag.exists():
        if not any(r.get("name") == "Weekly compliance" for r in rows):
            rows.append(dict(DEFAULT_SCHEDULES[2], last_run=""))
            _write_json(schedules_path(), rows)
        flag.write_text("1", "utf-8")
    out = []
    for s in rows:
        try:
            out.append(check_schedule(s))
        except (ValueError, TypeError):
            continue  # a broken entry never stops the others
    return out


def save_schedules(rows: list[dict[str, Any]], who: str = "") -> list[dict[str, Any]]:
    from src.soc import audit

    rows = [check_schedule(s) for s in rows]
    if len({s["name"] for s in rows}) != len(rows):
        raise ValueError("two schedules have the same name")
    _write_json(schedules_path(), rows)
    audit.record(who or "system", "report_schedules", {"schedules": [s["name"] for s in rows]})
    return rows


def due(s: dict[str, Any], now: float) -> bool:
    lt = time.localtime(now)
    hh, mm = map(int, s["time"].split(":"))
    return (s.get("enabled", True) and lt.tm_wday in s["days"]
            and (lt.tm_hour, lt.tm_min) >= (hh, mm)
            and s.get("last_run", "")[:10] != time.strftime("%Y-%m-%d", lt))  # fmt: skip


def run_due(now: float | None = None) -> list[dict[str, Any]]:
    """Make every schedule that is due (called by the autopilot each cycle)."""
    from src.soc.controller import notify

    now = time.time() if now is None else now
    rows = load_schedules()
    made = []
    for s in rows:
        if not due(s, now):
            continue
        override = {k: s[k] for k in ("sections", "formats") if s.get(k)}
        rep = generate(s["period"], template=override, folder=Path(s["folder"]) if s.get("folder") else None)
        s["last_run"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(now))
        rep["schedule"] = s["name"]
        made.append(rep)
        notify(f"{s['name']} report ready ({rep['label']}): {rep['headline'][:150]}")
    if made:
        _write_json(schedules_path(), rows)
    return made


def daily(now: float | None = None) -> list[dict[str, Any]]:
    """Kept for older callers: the scheduled reports that are due now."""
    return run_due(now)


# --- requests in plain English ---------------------------------------------------------

_ASK = re.compile(r"\b(make|generate|create|write|give me|prepare|produce|build|send me)\b.*\breport\b|\breport (for|of|on|from)\b|^\s*report\b", re.I)  # fmt: skip
_DATE = r"(\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2})?)"


def is_request(q: str) -> bool:
    return bool(_ASK.search(q))


def period_of(q: str) -> tuple[str, str | None]:
    """(period, incident id) named in a request; default today. A date range comes back
    as ``"range:<start>|<end>"`` (see ``range_of``)."""
    low = q.lower()
    if m := re.search(r"incident\s+([\w.:-]+-\d{6,})", q):
        return "incident", m.group(1)
    if m := re.search(rf"\bfrom\s+{_DATE}\s+(?:to|until|till|-)\s+{_DATE}", q, re.I):
        return f"range:{m.group(1)}|{m.group(2)}", None
    if m := re.search(rf"\b(?:on|for)\s+{_DATE}", q, re.I):
        return f"range:{m.group(1)}|", None
    if "yesterday" in low:
        return "yesterday", None
    if re.search(r"\bthis week\b", low):
        return "this_week", None
    if re.search(r"\bthis month\b", low):
        return "this_month", None
    if re.search(r"\b(week|weekly|7 days|seven days)\b", low):
        return "week", None
    if m := re.search(r"\blast (\d{1,3}) days\b", low):
        return f"last{m.group(1)}d", None
    return "today", None


def range_of(period: str) -> tuple[float, float]:
    """``range:<start>|<end>`` -> timestamps; a lone date covers that whole day."""
    a, _, b = period[len("range:"):].partition("|")
    start = parse_when(a)
    end = parse_when(b) if b else (start + DAY if len(a.strip()) == 10 else time.time())
    if len(b.strip()) == 10:
        end += DAY  # "to 2026-10-08" includes that day
    return start, end


def generate_request(q: str, who: str = "") -> dict[str, Any]:
    """Make the report a plain-English request asks for."""
    period, inc_id = period_of(q)
    if period.startswith("range:"):
        start, end = range_of(period)
        return generate(start=start, end=end, who=who)
    return generate(period, inc_id, who=who)


def main() -> int:
    from utils.logging_config import configure_logging

    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default="today")
    ap.add_argument("--id", default=None)
    ap.add_argument("--from", dest="start", default=None, help="YYYY-MM-DD[ HH:MM]")
    ap.add_argument("--to", dest="end", default=None, help="YYYY-MM-DD[ HH:MM]")
    a = ap.parse_args()
    if a.start or a.end:
        rep = generate(start=parse_when(a.start) if a.start else None,
                       end=parse_when(a.end) if a.end else None)  # fmt: skip
    else:
        rep = generate(a.period, a.id if a.period == "incident" else None)
    print(rep["headline"])
    print(*rep["files"].values(), sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
