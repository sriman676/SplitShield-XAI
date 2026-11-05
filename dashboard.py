"""SplitShield-XAI SOC dashboard.

    python -m streamlit run dashboard.py        (LAUNCH.cmd opens it automatically)

Pages: Live monitor - Incidents & review - Analyze logs - Explain attacks -
Add dataset - Learning (RCL) - Connect SOC - Project results.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.knowledge.attack_kb import explain, known_names
from src.soc import engine, sinks
from src.soc.correlate import (
    CLASS_TECHNIQUES,
    LEVEL_ICON,
    is_unknown,
    level_of,
    unknown_label,
)
from src.soc.rcl import RCL
from src.soc.safe import md
from src.soc.samples import SAMPLES, make_samples
from utils.paths import PROJECT_ROOT, RESULTS_DIR

st.set_page_config(page_title="SplitShield-XAI SOC", page_icon="🛡️", layout="wide")
DRIFT_LABEL = {
    "stable": "🟢 STABLE",
    "moderate": "🟡 MODERATE",
    "drifted": "🔴 DRIFTED",
    "not_enough_data": "⏳ NOT ENOUGH DATA",
    "no_reference": "⚪ NO REFERENCE",
}
ACTION_ICON = {
    "retrain_detector": "🔁",
    "drift_notice": "📈",
    "improve_assistant": "🤖",
    "assistant_skip": "⏭️",
    "skip": "⏭️",
}


@st.cache_resource
def detector():  # noqa: ANN201 - Detector | None
    return engine.load_detector()


def _t(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _mitre(tid: str) -> str:
    from src.soc.correlate import technique_info

    name = technique_info(tid)[0]
    return f"[{tid} {name}](https://attack.mitre.org/techniques/{tid}/)"


def _howto(text: str) -> None:
    """Plain-English help for the page (owner 2026-10-14), as one quiet caption under
    the title instead of a box on every page (2026-10-16 redesign, no emoji)."""
    st.caption(_NO_EMOJI.sub("", text).replace("( ", "(").replace("  ", " "))


_NO_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\u2600-\u27bf\u2b00-\u2bff\u2300-\u23ff\ufe0f\u200d]\\s?"
)


HELP = {
    "macro_f1": "Macro-F1: the average of the per-attack-type F1 scores (0-1). Every "
    "attack type counts equally, so rare attacks are not hidden by common ones.",
    "psi": "PSI (Population Stability Index): how much live traffic differs from the "
    "training data. Below 0.1 = same, 0.1-0.25 = some change, above 0.25 = large shift.",
    "temperature": "Temperature: a calibration knob. Above 1 makes confidence scores "
    "more modest, so 90% confidence really means right about 9 times in 10.",
    "rcl": "Reinforcement Calibrated Learning: each 'Real' / 'False alarm' click is a "
    "reward that moves the alert threshold for that attack type.",
    "precision": "Of the alerts you reviewed recently, the share that were real attacks.",
}


def _reviewed() -> set[str]:
    return {a for r in RCL().rows() if (a := r.get("alert_id"))}


# --- pages -----------------------------------------------------------------------


def page_live() -> None:
    st.header("Live monitor")
    _howto(
        "1. Open **⚙️ Start / stop monitoring** and pick a source: replay a flow CSV "
        "(the demo works out of the box), live capture, or logs only.\n"
        "2. Press **▶ Start monitoring**. New alerts pop up in the corner and appear in "
        "the table below (it refreshes every 3 seconds).\n"
        "3. Open **🧩 Incidents** to see related alerts grouped into attacks."
    )
    det = detector()
    if det is None:
        st.warning(
            "The detector is not trained yet (training is still running or failed). "
            "Log analysis still works; flow detection starts once training finishes."
        )
    run = engine.monitor_running()
    c1, c2, c3 = st.columns(3)
    c1.metric("Monitor", "🟢 RUNNING" if run else "⚪ STOPPED")
    c2.metric("Flows checked", run.get("flows", 0) if run else "-")
    c3.metric("Alerts raised", len(engine.read_jsonl(engine.ALERTS)))

    with st.expander("Start / stop monitoring", expanded=not run):
        mode = st.radio(
            "Source",
            [
                "Replay a flow CSV (demo / exported traffic)",
                "Live network capture",
                "Logs only (inbox folder + syslog listener)",
            ],
        )
        args: list[str] = []
        if mode.startswith("Replay"):
            up = st.file_uploader("Flow CSV (CICFlowMeter columns)", type="csv")
            use_demo = st.checkbox("Use the demo flows", value=up is None)
            rate = st.slider("Flows per second", 1, 200, 20)
            if up is not None and not use_demo:
                path = sinks.SOC_DIR / f"replay_{Path(up.name).name}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(up.getvalue())
            else:
                path = SAMPLES / "flows_sample.csv"
            args = ["--replay", str(path), "--rate", str(rate), "--keep-running"]
        elif mode.startswith("Live"):
            st.caption(
                "Needs Npcap + the `cicflowmeter` package and the dashboard running as Administrator."
            )
            iface = st.text_input(
                "Network interface name (e.g. Ethernet, Wi-Fi)", "Wi-Fi"
            )
            args = ["--live", "--iface", iface]
        block = st.checkbox(
            "Auto-block severe attackers in Windows Firewall (Administrator)"
        )
        b1, b2 = st.columns(2)
        if b1.button("▶ Start monitoring", disabled=bool(run), type="primary"):
            if mode.startswith("Replay") and not Path(args[1]).exists():
                make_samples()
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "src.soc.engine",
                    "monitor",
                    *args,
                    *(["--block"] if block else []),
                ],
                cwd=PROJECT_ROOT,
                creationflags=flags,
            )
            time.sleep(2)
            st.rerun()
        if b2.button("■ Stop", disabled=not run) and run:
            try:
                os.kill(int(run["pid"]), signal.SIGTERM)
            except (OSError, KeyError, ValueError):
                pass
            engine._status(running=False)
            st.rerun()

    feed()


@st.fragment(run_every=3)
def feed() -> None:
    """Auto-refreshing alert feed; pops a toast for every new alert."""
    every = engine.read_jsonl(engine.ALERTS)
    # two-tier alerting: low-confidence detector alerts are recorded, hidden by default
    show_low = st.session_state.get("show_low", False)
    alerts = [a for a in every if show_low or a.get("tier") != "low"]
    seen = st.session_state.get("seen_alerts", len(alerts))
    for a in alerts[seen:][-5:]:
        st.toast(f"🚨 {md(a['title'])} from {md(a.get('src_ip') or 'unknown')}", icon="🚨")
    st.session_state["seen_alerts"] = len(alerts)

    incs = engine.incidents()
    crit = [i for i in incs if i["level"] == "critical"]
    if crit:
        st.error(
            f"🔴 ATTACK IN PROGRESS / DETECTED: {len(crit)} critical incident(s) — "
            + "; ".join(
                f"{i['src_ip']}: {', '.join(t['name'] for t in i['techniques'][:3])}"
                for i in crit[:3]
            )
        )
    st.subheader(f"Latest alerts ({len(alerts)} total)")
    st.checkbox(
        f"Show low-confidence detector alerts ({len(every) - len([a for a in every if a.get('tier') != 'low'])})",
        key="show_low",
        help="Flows with attack probability >= 0.5 but below the ~1%-false-alarm "
        "threshold: recorded so nothing is lost, low severity, not pushed to the SIEM.",
    )
    if alerts:
        rows = [
            {
                "time": _t(a["ts"]),
                "attack": a["title"],
                "source IP": a.get("src_ip") or "-",
                "confidence": round(float(a.get("confidence", 1)), 2),
                "severity": (
                    f"{LEVEL_ICON[level_of(a['severity'])]} {a['severity']}"
                    if isinstance(a.get("severity"), int | float)
                    else a.get("severity")
                ),
                "tier": "🔵 LOW" if a.get("tier") == "low" else "🔴 CONFIRMED",
                "MITRE": ", ".join(
                    a.get("techniques")
                    or [f"({t}?)" for t in a.get("candidate_techniques", [])]
                ),
                "sensor": a.get("source"),
            }
            for a in alerts[-50:][::-1]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info(
            "✅ No alerts yet. Start monitoring, or load the demo on 'Analyze logs'."
        )


def _case_box(incident_id: str) -> None:
    """Status, assignee and notes of one incident's case (src/soc/cases.py)."""
    from src.soc import cases, users

    c = cases.get(incident_id)
    st.caption(f"Case: **{cases.STATUSES[c['status']]}**"
               + (f" · assigned to {md(c['assignee'])}" if c.get("assignee") else "")
               + (f" · updated {c['updated']}" if c.get("updated") else ""))  # fmt: skip
    for n in c.get("notes", [])[-5:]:
        st.caption(f"📝 {n['when']} {md(n['who'])}: {md(n['text'])}")
    if not _can("review"):
        return
    k = f"case_{incident_id}"
    c1, c2, c3 = st.columns([1, 1, 2])
    status = c1.selectbox("Status", list(cases.STATUSES), index=list(cases.STATUSES).index(c["status"]),
                          format_func=cases.STATUSES.__getitem__, key=f"{k}_s")  # fmt: skip
    people = ["", *users.load()["users"]]
    who = c2.selectbox("Assignee", people, index=people.index(c["assignee"]) if c.get("assignee") in people else 0,
                       key=f"{k}_a")  # fmt: skip
    note = c3.text_input("Add a note", key=f"{k}_n")
    if st.button("Save case", key=f"{k}_b"):
        try:
            if status != c["status"]:
                cases.set_status(_me(), incident_id, status)
            if who != c.get("assignee", ""):
                cases.assign(_me(), incident_id, who)
            if note.strip():
                cases.add_note(_me(), incident_id, note)
            st.success("Case saved.")
        except (ValueError, PermissionError) as exc:
            st.error(str(exc))


def page_incidents() -> None:
    st.header("Incidents & review")
    _howto(
        "Alerts from the same source close in time are grouped into one **incident** "
        "with its MITRE ATT&CK attack path and a severity (🔴 critical ≥ 8, 🟠 high "
        "≥ 6, 🟡 medium ≥ 4, 🔵 low). Below, review detector alerts with **✔ Real** "
        "or **✖ False alarm**. Each answer teaches the detector (RCL)."
    )
    window = st.slider("Correlation window (minutes)", 1, 120, 15)
    incs = engine.incidents(window_s=window * 60)
    if not incs:
        st.info("✅ No incidents yet.")
    else:
        import json as _json

        from src.soc.navigator import layer

        st.download_button(
            "Download ATT&CK Navigator layer",
            _json.dumps(layer(incs), indent=2),
            file_name="splitshield_attack_layer.json",
            mime="application/json",
            help="Open in MITRE ATT&CK Navigator: Open Existing Layer -> Upload from local.",
        )
    icons = {
        "critical": "emergency",
        "high": "warning",
        "medium": "report",
        "low": "info",
    }
    from src.soc import cases

    case_of = cases.all_cases()
    show = st.multiselect("Show cases", list(cases.STATUSES), default=list(cases.OPEN),
                          format_func=cases.STATUSES.__getitem__)  # fmt: skip
    incs = [i for i in incs if case_of.get(i["id"], {}).get("status", "new") in show]
    for inc in incs:
        who = inc["src_ip"] if not is_unknown(inc["src_ip"]) else unknown_label(inc["src_ip"])
        techs = ", ".join(t["name"] for t in inc["techniques"][:3])
        more = len(inc["techniques"]) - 3
        # labels render Markdown: entities come from log content (user / file names)
        title = (f"{inc['level'].capitalize()} {inc['severity']}/10 · {md(who, 120)} · {md(techs)}"
                 + (f" +{more}" if more > 0 else "") + f" · {_t(inc['start'])}")  # fmt: skip
        icon = f":material/{icons.get(inc['level'], 'info')}:"
        with st.expander(title, expanded=inc["level"] == "critical", icon=icon):
            st.write(
                f"🕒 **When:** {_t(inc['start'])} → {_t(inc['end'])} · **events:** {inc['n_events']} · "
                f"**sensors:** {', '.join(md(s) for s in inc['sources'])}"
                + (" · ✅ corroborated by 2+ sensors" if inc["corroborated"] else "")
            )
            _case_box(inc["id"])
            st.write(
                "🧭 **Attack path (MITRE ATT&CK):** "
                + " → ".join(
                    f"{t['tactic']}: {_mitre(t['id'])}" for t in inc["techniques"]
                )
            )
            for f in inc["findings"]:
                st.warning(f"⚠️ {md(f)}")
            if inc.get("linked_findings"):
                st.info(
                    f"🔗 **Other models agree** (severity +{inc.get('board_boost', 0)}, "
                    "shared risk board):\n\n"
                    + "\n".join(f"- {md(r)}" for r in inc["linked_findings"])
                )
            if inc["techniques"]:
                from src.knowledge.explain import incident_markdown

                with st.expander(
                    "Full step-by-step explanation (what happened, how, what to do)"
                ):
                    st.markdown(incident_markdown(inc))
                n = shield_assistant().narrate(inc)
                st.info(f"📝 **Narrative** ({md(n['by'])}): {md(n['text'])}")
                if n["delex"]:
                    c = st.columns([1, 1, 6])
                    for col, good, icon in ((c[0], True, "👍"), (c[1], False, "👎")):
                        if col.button(icon, key=f"nf{int(good)}{inc['id']}"):
                            from src.nlp.selfimprove import record_feedback

                            record_feedback(n["facts"], n["delex"], good)
                            st.toast(
                                "Thanks - used by the next 'Improve assistant' run"
                            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "time": _t(e["ts"]),
                            **{k: e[k] for k in ("source", "kind", "detail")},
                        }
                        for e in inc["timeline"]
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

    st.subheader("Review detector alerts")
    if not _can("review"):
        st.caption("Your role may look at alerts but not review them.")
        return
    det = detector()
    done = _reviewed()
    todo = [
        a
        for a in engine.read_jsonl(engine.ALERTS)
        if a.get("source") == "detector" and a["id"] not in done
    ]
    st.caption(
        f"{len(todo)} alert(s) waiting for review. Each verdict is a reward that tunes the alert "
        "threshold and calibration."
    )
    rcl = RCL()
    for a in todo[-10:][::-1]:
        c = st.columns([4, 1, 1, 2])
        c[0].write(
            f"**{a['label']}** ({a['confidence']:.2f}) from {a.get('src_ip') or '-'} at {_t(a['ts'])}"
        )
        if c[1].button("✔ Real", key=f"ok{a['id']}"):
            st.success(f"Threshold {rcl.feedback(a, True)}")
            _audit("review", {"alert": a["id"], "verdict": "real"})
            st.rerun()
        if c[2].button("✖ False alarm", key=f"fp{a['id']}"):
            st.info(f"Threshold {rcl.feedback(a, False)}")
            _audit("review", {"alert": a["id"], "verdict": "false_alarm"})
            st.rerun()
        if det is not None:
            other = c[3].selectbox(
                "Actually…",
                ["-", *det.label_names],
                key=f"rl{a['id']}",
                label_visibility="collapsed",
            )
            if other != "-":
                rcl.feedback(a, False, true_label=other)
                _audit("review", {"alert": a["id"], "verdict": "relabel", "label": other})
                st.rerun()


def page_analyze() -> None:
    st.header("Analyze logs")
    _howto(
        "Drag any mix of files onto the box (several at once is fine) and press "
        "**🔍 Analyze**. The format of each file is detected automatically. No files "
        "yet? **🎬 Load demo attack data** fills the dashboard with a realistic attack."
    )
    st.write(
        "Upload any mix of: **flow CSVs** (CICFlowMeter / NetFlow), **Suricata eve.json**, **Zeek conn.log**, "
        "**Windows Firewall pfirewall.log**, **Windows Security events** (JSON), "
        "**Windows event log exports (.evtx: Security, Sysmon, PowerShell)**, **syslog / auth.log**, "
        "or a **.zip** holding any of these. The format is detected automatically, original "
        "timestamps are kept, and everything is correlated into incidents."
    )
    ups = st.file_uploader("Log / flow files", accept_multiple_files=True)
    send = st.checkbox("Also forward alerts to the connected SOC", value=True)
    if ups and st.button("Analyze", type="primary"):
        det = detector()
        for up in ups:
            if up.name.lower().endswith(".zip"):  # e.g. OTRF Security-Datasets
                try:
                    st.write(
                        f"**{up.name}:** ",
                        engine.ingest_zip(up.getvalue(), up.name, det, send),
                    )
                except (ValueError, RuntimeError) as exc:
                    st.warning(f"⚠️ {exc}")
                continue
            if up.name.lower().endswith((".html", ".htm", ".eml")):  # page / e-mail: phishing page model
                try:
                    res = engine.ingest_page(up.getvalue(), up.name, send)
                    st.write(f"**{md(up.name)}:** {res['pages']} page(s), {res['phishing']} phishing")
                except (FileNotFoundError, ValueError) as exc:
                    st.warning(f"⚠️ {exc}")
                continue
            if up.name.lower().endswith(".evtx"):  # binary Windows event log export
                try:
                    st.write(
                        f"**{up.name}:** ",
                        engine.ingest_evtx(up.getvalue(), up.name, send),
                    )
                except RuntimeError as exc:
                    st.warning(f"⚠️ {exc}")
                continue
            if det is not None:
                det.last_input_check = None
            st.write(
                f"**{up.name}:** ",
                engine.ingest_text(
                    up.getvalue().decode("utf-8", "replace"), up.name, det, send
                ),
            )
            chk = getattr(det, "last_input_check", None)
            for w in (chk or {}).get("warnings", []):  # live-input preprocessing check
                st.warning(f"{up.name}: {w}. Alerts from this file may be unreliable.")
        st.success("✅ Done — see 'Incidents'.")
    real = _real_world_samples()
    if real:
        with st.expander("Real-world logs (downloaded, nothing removed)"):
            st.caption(
                "Unedited logs with their original timestamps, hostnames and IPs: OTRF "
                "Security-Datasets (Windows attack recordings) and Loghub (a real internet-"
                "facing SSH server). Old dates are expected: they show as 'happened', not "
                "'ongoing'."
            )
            pick = st.selectbox("Sample", list(real))
            if st.button("Load this sample"):
                path = real[pick]
                with st.spinner("Parsing…"):
                    if path.suffix == ".zip":
                        res = engine.ingest_zip(
                            path.read_bytes(), path.name, detector(), send
                        )
                    elif path.suffix == ".evtx":
                        res = engine.ingest_evtx(path.read_bytes(), path.name, send)
                    else:
                        text = path.read_text("utf-8", errors="replace")
                        res = engine.ingest_text(text, path.name, detector(), send)
                st.success(f"✅ {pick}: {res}")
    c1, c2, c3 = st.columns(3)
    if c1.button("Load demo attack data"):
        for p in make_samples():
            st.write(
                f"{p.name}: ",
                engine.ingest_text(p.read_text("utf-8"), p.name, detector(), send),
            )
        st.success(
            "✅ Demo loaded — open 'Incidents' to see the correlated attack chains."
        )
    from src.soc import osops

    if osops.PLATFORM != "windows" and c2.button("Read this computer's logs now"):
        from src.soc import oslive

        n = oslive.make_local_agent().run_once()
        st.success(f"{n} new records read.")
        for src_name, s_txt in oslive.status().items():
            if s_txt == "ok":
                st.write(f"{src_name}: {s_txt}")
            else:
                st.warning(f"{src_name}: {s_txt}")
    if osops.PLATFORM == "windows" and c2.button("Pull Windows logs: Security, Sysmon, PowerShell"):
        from src.soc.parsers import read_windows_logs

        evs, status = read_windows_logs()
        for name, s in status.items():
            if "events" in s:
                st.write(f"{name}: {s}")
            else:
                st.warning(f"{name}: {s}")
        if evs:
            st.success(
                f"{len(evs)} events read, {len(engine.record(evs, send))} alerts."
            )
        st.caption(
            "Security needs the dashboard to run as Administrator; Sysmon must be "
            "installed (scripts/install_sysmon.ps1)."
        )
    if c3.button("Clear all stored events / alerts"):
        engine.clear_store()
        _audit("store_cleared")
        st.rerun()


def _real_world_samples() -> dict[str, Path]:
    """Downloaded raw log samples (data/raw), by display name."""
    from utils.paths import RAW_DIR

    out: dict[str, Path] = {}
    for f in sorted((RAW_DIR / "otrf-security-datasets" / "windows").rglob("*.zip")):
        out[f"OTRF · {f.parts[-3].replace('_', ' ')} · {f.stem}"] = f
    for name in ("SSH", "Linux", "Apache"):
        f = RAW_DIR / "loghub" / name / f"{name}.log"
        if f.is_file():
            out[f"Loghub · real {name} server log"] = f
    for f in sorted((RAW_DIR / "evtx-attack-samples").rglob("*.evtx"))[:400]:
        out[f"EVTX · {f.parent.name} · {f.stem}"] = f
    return out


def _technique_explorer() -> None:
    """Any of the 706 MITRE ATT&CK techniques, explained in five parts."""
    from src.knowledge.explain import find, kb, technique_markdown

    st.subheader("Any ATT&CK technique")
    q = st.text_input(
        "Name, ID or everyday term",
        placeholder="e.g. kerberoasting, T1548, golden ticket, web shell",
    )
    if q:
        tid = find(q)
        if tid:
            st.markdown(technique_markdown(tid))
        else:
            st.info("No match. Try another word or an ID like T1003.")
    else:
        names = sorted(f"{t} · {v['name']}" for t, v in kb().items())
        pick = st.selectbox("…or browse all techniques", ["—", *names])
        if pick != "—":
            st.markdown(technique_markdown(pick.split(" · ")[0]))
    st.divider()


def page_explain() -> None:
    from src.eval.final_report import ATTACKS

    st.header("Explain attacks")
    _technique_explorer()
    _howto(
        "Pick an attack to see what it is, how it is done, how the model spots it and how to defend."
    )
    det = detector()
    name = st.selectbox("Attack", known_names())
    exp = explain(name)
    if exp:
        st.subheader(
            f"{exp['name']} ({'network attack class' if exp['kind'] == 'attack_class' else 'attack on the ML model'})"
        )
        st.write(f"❓ **What it is:** {exp['what']}")
        st.write(f"🛠️ **How the attacker does it:** {exp['attacker']}")
        st.write(f"🔬 **How it is studied in this project:** {exp['howto']}")
        if exp["name"] in CLASS_TECHNIQUES:
            st.write(
                "🧬 **MITRE ATT&CK:** "
                + ", ".join(_mitre(t) for t in CLASS_TECHNIQUES[exp["name"]])
                or "-"
            )
        if det is not None and exp["name"] in det.top_features:
            st.write(
                "🧠 **Why our model flags it (top SHAP features):** "
                + ", ".join(det.top_features[exp["name"]])
            )
        st.write("🛡️ **How to defend:**")
        for d in exp["defenses"]:
            st.write(f"- {d}")
    st.divider()
    st.subheader("Attacks against the detector itself (and how we measured them)")
    for a in ATTACKS.values():
        with st.expander(a["name"]):
            st.write(
                f"**What:** {a['what']}\n\n**Adversary:** {a['adversary']}\n\n**Measured by:** {a['measure']}"
            )


BIG = 300 * 2**20  # files above this are sampled while reading


def _read_table(src, size: int, name: str, max_rows: int = 300_000) -> pd.DataFrame:  # noqa: ANN001
    """Read a CSV / Parquet (upload or path). A big file is sampled evenly while
    reading (every chunk contributes), so it never sits in memory whole."""
    if name.lower().endswith(".parquet"):
        df = pd.read_parquet(src)
        return df.sample(n=max_rows, random_state=0) if len(df) > max_rows else df
    if size <= BIG:
        return pd.read_csv(src, low_memory=False)
    frac = min(1.0, max_rows * 400 / size)  # ~400 bytes per CICFlowMeter row
    parts = [
        c.sample(frac=frac, random_state=i)
        for i, c in enumerate(pd.read_csv(src, chunksize=200_000, low_memory=False))
    ]
    return pd.concat(parts, ignore_index=True)


def _read_path(text: str) -> tuple[pd.DataFrame | None, str]:
    """A file or a whole folder of CSV / Parquet files on this PC (no size limit);
    each file contributes in proportion to its size."""
    path = Path(text.strip().strip('"'))
    if not path.exists():
        return None, f"Not found: {path}"
    files = (
        [path]
        if path.is_file()
        else sorted(
            q for q in path.rglob("*") if q.suffix.lower() in (".csv", ".parquet")
        )
    )
    if not files:
        return None, "No .csv or .parquet files in that folder."
    total = sum(f.stat().st_size for f in files)
    parts = [
        _read_table(
            f,
            f.stat().st_size,
            f.name,
            max(1000, int(300_000 * f.stat().st_size / total)),
        )
        for f in files
    ]
    return pd.concat(
        parts, ignore_index=True
    ), f"{len(files)} file(s), {total / 2**30:.2f} GB on disk"


def page_dataset() -> None:
    from src.soc.columns import align, explain

    st.header("Add dataset")
    _howto(
        "Drag a **labelled** flow CSV / Parquet onto the box (up to 100 GB), or give a "
        "**path** to a file or a whole folder on this PC (no limit). Column names "
        "from CIC-IDS2017, CIC-IDS2018 and CICFlowMeter v4 are recognised "
        "automatically. You see which columns matched and how the labels map before "
        "anything changes. The new model is kept only if it is not worse on a "
        "held-out 20% of your rows."
    )
    det = detector()
    if det is None:
        st.warning("⏳ Train the detector first.")
        return
    how = st.radio(
        "Source",
        [
            "📥 Drag and drop (up to 100 GB)",
            "📁 Path on this PC (any size, file or folder)",
        ],
        horizontal=True,
    )
    if how.startswith("📥"):
        up = st.file_uploader(
            "Labelled flow CSV / Parquet (drag and drop)",
            type=["csv", "parquet"],
            key="ds",
        )
        if up is None:
            return
        with st.spinner("Reading…"):
            df = _read_table(up, up.size, up.name)
        src_name, big = up.name, up.size > BIG
    else:
        text = st.text_input("File or folder path", placeholder=r"D:\datasets\my_flows")
        if not text:
            return
        with st.spinner("Reading (large files are sampled while reading)…"):
            df, msg = _read_path(text)
        if df is None:
            st.error(f"❌ {msg}")
            return
        st.caption(f"📁 {msg}")
        src_name, big = Path(text.strip().strip('"')).name, True
    if big:
        st.caption(f"Large source: an even sample of {len(df):,} rows is used.")
    st.dataframe(df.head(), use_container_width=True)
    aligned, cols = align(df, det.feature_names)
    verdict = explain(cols)
    c1, c2, c3 = st.columns(3)
    c1.metric("Format", cols["format"].split(" (")[0])
    c2.metric("Features matched", f"{cols['matched']}/{cols['needed']}")
    c3.metric("Rows", f"{len(df):,}")
    (st.success if cols["share"] >= 0.8 else st.error)(
        ("✅ " if cols["share"] >= 0.8 else "❌ ") + verdict
    )
    with st.expander("Column details"):
        if cols["rename"]:
            st.write("**Renamed to the detector's names:**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "your column": list(cols["rename"]),
                        "detector feature": list(cols["rename"].values()),
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        st.write(
            "**Missing:** "
            + (", ".join(cols["missing"]) if cols["missing"] else "none ✅")
        )
    if cols["share"] < 0.8:
        return
    names = list(df.columns)
    guess = next(
        (c for c in names if str(c).strip().lower() in ("label", "attack", "class")),
        names[-1],
    )
    label = st.selectbox(
        "Which column holds the attack label?", names, index=names.index(guess)
    )
    mapped, unknown = engine.map_labels(df[label], det.label_names)
    counts = (
        pd.DataFrame(
            {
                "your label": df[label].astype(str),
                "detector class": mapped.fillna("❌ ignored"),
            }
        )
        .value_counts()
        .rename("rows")
        .reset_index()
    )
    st.write("**How your labels map to the detector's classes:**")
    st.dataframe(counts, hide_index=True, use_container_width=True)
    if unknown:
        st.info(
            f"ℹ️ {len(unknown)} label(s) do not match a detector class and are skipped: "
            + ", ".join(unknown[:10])
        )
    name = st.text_input("Dataset name", Path(src_name).stem)
    if st.button("Validate and fine-tune", type="primary"):
        try:
            with st.spinner("Fine-tuning…"):
                res = engine.add_dataset(det, aligned, str(label).strip(), name)
            st.cache_resource.clear()
            (st.success if res["kept"] else st.warning)(
                f"macro-F1 {res['macro_f1_before']:.3f} → {res['macro_f1_after']:.3f} — "
                + (
                    "✅ new model deployed."
                    if res["kept"]
                    else "↩️ kept the previous model (the new one was worse)."
                )
            )
            st.json(res)
        except ValueError as exc:
            st.error(f"❌ {exc}")


def page_learning() -> None:
    st.header("Learning from your reviews")
    _howto(
        "This page shows what the detector learned from your reviews. **🔁 Retrain from "
        "feedback** folds reviewed alerts into the model; it is kept only if it does "
        "not get worse. **🚀 Improve assistant** retrains the chat assistant on your "
        "incidents (several minutes)."
    )
    st.write(
        "Analyst verdicts are rewards: false alarms raise that attack class's alert threshold, confirmed "
        "attacks lower it; the confidence temperature is refit so confidence matches reality; confirmed "
        "alerts become training rows."
    )
    rcl = RCL()
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviewed alerts", len(rcl.history), help=HELP["rcl"])
    c2.metric("Temperature T", f"{rcl.temperature:.2f}", help=HELP["temperature"])
    c3.metric(
        "Precision (last 20)",
        f"{rcl.precision_curve()[-1]['precision']:.0%}" if rcl.history else "-",
        help=HELP["precision"],
    )
    if rcl.thresholds:
        st.bar_chart(pd.Series(rcl.thresholds, name="alert threshold"))
    if rcl.history:
        st.line_chart(pd.DataFrame(rcl.precision_curve()).set_index("review"))
    det = detector()
    if det is not None and st.button("Retrain from feedback", type="primary"):
        with st.spinner("Fine-tuning on reviewed alerts…"):
            res = engine.retrain_from_feedback(det, rcl)
            _audit("retrain_from_feedback", {"result": str(res)[:300]})
        st.cache_resource.clear()
        st.json(res)

    st.subheader("Improve the assistant (ShieldLM self-improvement)")
    from src.nlp import selfimprove

    rd = selfimprove.ready()
    st.write(
        "One gated cycle on your real incidents: self-training on the model's own "
        "faithful narratives (unsupervised), your 👍/👎 on narratives (supervised), then "
        "RLCD. The new model replaces the current one only if it passes the quality "
        "gate; the previous one is kept for rollback."
    )
    if rd["missing"]:
        st.info(f"Train the NLP stack first (missing {', '.join(rd['missing'])}).")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Incidents with techniques", rd["incidents"])
    c2.metric("narratives", rd["feedback_up"])
    c3.metric("narratives", rd["feedback_down"])
    c = st.columns(2)
    if c[0].button("🚀 Improve assistant", type="primary"):
        with st.spinner("Self-improving ShieldLM (several minutes on CPU)…"):
            res = selfimprove.improve()
        st.cache_resource.clear()
        if "error" in res:
            st.warning(res["error"])
        else:
            (st.success if res["promoted"] else st.info)(
                "✅ New model deployed." if res["promoted"] else
                "↩️ Kept the current model: the candidate did not pass the gate."
            )  # fmt: skip
            st.json(
                {k: v for k, v in res.items() if k != "gate"}
                | {"gate": res["gate"]["checks"]}
            )
    if c[1].button("↩️ Roll back last improvement"):
        ok = selfimprove.rollback()
        st.cache_resource.clear()
        (st.success if ok else st.info)(
            "✅ Restored." if ok else "Nothing to roll back."
        )


def page_check() -> None:
    """The extra ML detectors (owner 2026-10-15): phishing URLs, malware files, ICS."""
    st.header("Check URL / file / sensors")
    _howto(
        "Each tab uses a model trained and tested on public data (see 📊 Project "
        "results → 🧪 tab for how well it does). Every answer comes with numbered "
        "reasons and what to do."
    )
    t1, t2, t3, t4 = st.tabs(
        [
            "Phishing URL",
            "Malware file (EMBER features)",
            "ICS sensors",
            "Insider risk",
        ]
    )
    with t1:
        url = st.text_input(
            "URL or domain", placeholder="e.g. secure-paypa1-login.xyz/verify"
        )
        if url:
            try:
                from src.mlareas.phishing import check_url

                r = check_url(url)
                (st.error if r.get("verdict") == "phishing" else st.success)(
                    f"{'🎣' if r.get('verdict') == 'phishing' else '✅'} {r.get('verdict', '?').upper()} "
                    f"— phishing probability {float(r.get('phishing_probability', 0)):.0%}"
                )
                why = r.get("why")
                st.markdown("\n".join(why) if isinstance(why, list) else str(why))
            except Exception as exc:  # noqa: BLE001 - model missing / bad input
                st.warning(f"⚠️ Phishing model not available: {exc}")
    with t2:
        up = st.file_uploader(
            "One line of EMBER-format JSON features", type=["json", "jsonl"], key="mw"
        )
        if up is not None:
            import json as _json

            try:
                from src.mlareas.malware import check_file_features

                raw = _json.loads(up.getvalue().decode("utf-8").splitlines()[0])
                r = check_file_features(raw)
                bad = r.get("verdict") != "benign"
                (st.error if bad else st.success)(
                    f"{'🦠' if bad else '✅'} {str(r.get('verdict', '?')).upper()} — malware "
                    f"probability {float(r.get('malware_probability', 0)):.1%}"
                )
                why = r.get("why")
                st.markdown("\n".join(why) if isinstance(why, list) else str(why))
            except Exception as exc:  # noqa: BLE001
                st.warning(f"⚠️ Could not score this file: {exc}")
    with t3:
        up = st.file_uploader(
            "Sensor CSV (1 row per second, HAI column names)", type="csv", key="ics"
        )
        if up is not None:
            try:
                from src.mlareas.ics import score_window

                r = score_window(pd.read_csv(up))
                (st.error if r.get("alarm") else st.success)(
                    f"{'🚨 ALARM' if r.get('alarm') else '✅ normal'} — anomaly score "
                    f"{float(r.get('score', r.get('anomaly_score', 0))):.3f}"
                )
                why = r.get("why")
                st.markdown("\n".join(why) if isinstance(why, list) else str(why))
            except Exception as exc:  # noqa: BLE001
                st.warning(f"⚠️ Could not score this window: {exc}")
    with t4:
        _insider_tab()


def _insider_tab() -> None:
    up = st.file_uploader(
        "User-day features (CSV, one row per user-day)", type="csv", key="ins"
    )
    if up is None:
        st.caption(
            "Columns as produced by scripts/train_insider (e.g. usb_days7, http_n_z30, after_hours_logons)."
        )
        return
    try:
        from src.mlareas.insider import score_user_day

        df = pd.read_csv(up)
        for _, row in df.head(20).iterrows():
            r = score_user_day(row.to_dict())
            risky = str(r.get("verdict", "")).lower() not in (
                "low",
                "normal",
                "low risk",
            )
            with st.expander(f"{'🚨' if risky else '✅'} {md(row.get('user', ''))} {md(row.get('day', ''))} — "
                             f"risk {float(r.get('risk', 0)):.2f} ({r.get('verdict')})"):  # fmt: skip
                why = r.get("why")
                st.markdown("\n".join(why) if isinstance(why, list) else str(why))
    except Exception as exc:  # noqa: BLE001
        st.warning(f"⚠️ Could not score: {exc}")


def page_test_dataset() -> None:
    """Report card for ANY dataset (owner 2026-10-15)."""
    import tempfile

    from scripts.test_dataset import test_path

    st.header("Test any dataset")
    _howto(
        "Drop in a file, or give a file / folder path. The type is detected "
        "automatically (flows, NetFlow, URLs, EMBER features, any logs) and every "
        "matching detector runs. With a label column you also get caught / missed / "
        "false alarms. Nothing is trained on the data you test."
    )
    up = st.file_uploader("File", key="anyds")
    path_text = st.text_input("…or a file / folder path on this PC")
    label = st.text_input("Label column (optional)", "")
    if st.button("Run report card", type="primary") and (up or path_text):
        with st.spinner("Testing…"):
            if up is not None:
                tmp = Path(tempfile.mkdtemp()) / Path(up.name).name
                tmp.write_bytes(up.getvalue())
                cards = test_path(tmp, label or None)
            else:
                cards = test_path(Path(path_text.strip().strip('"')), label or None)
        for c in cards:
            with st.container(border=True):
                st.markdown(f"**{c.get('file')}** · {c.get('type', '?')}")
                if "error" in c:
                    st.warning(c["error"])
                    continue
                cols = st.columns(4)
                cols[0].metric("Rows / events", c.get("rows", c.get("events", "-")))
                cols[1].metric("🚨 Alerts", c.get("alerts", "-"))
                sc = c.get("score")
                if sc:
                    cols[2].metric("✅ Attacks caught", f"{sc['detection_rate']:.1%}")
                    cols[3].metric("🔕 False alarms", f"{sc['false_alarm_rate']:.2%}")
                st.json(
                    {k: v for k, v in c.items() if k not in ("file", "type")},
                    expanded=False,
                )


def page_new_formats() -> None:
    """Parser suggestions for log formats the SOC had never seen (owner approves)."""
    import json as _json

    from src.soc.unknown import SUGGESTIONS

    st.header("New log formats")
    _howto(
        "When you load a log type the SOC does not know, it still checks every value "
        "for attack indicators and flags unusual lines, and it proposes which fields "
        "hold the time, source IP, user and message. Approve or reject each proposal."
    )
    rows = []
    if SUGGESTIONS.is_file():
        for ln in SUGGESTIONS.read_text("utf-8").splitlines():
            try:
                rows.append(_json.loads(ln))
            except ValueError:
                continue
    if not rows:
        st.info("✅ No new formats seen yet.")
        return
    changed = False
    for i, r in enumerate(reversed(rows[-30:])):
        with st.expander(
            f"{r.get('status', '?')} · {r.get('file')} · {r.get('structure')} · {r.get('records')} records"
        ):
            st.json(r.get("fields"))
            st.caption("Fields seen: " + ", ".join(r.get("field_names", [])[:25]))
            c1, c2 = st.columns(2)
            if c1.button("Approve", key=f"ap{i}"):
                r["status"], changed = "approved", True
                _audit("log_format_approved", {"format": str(r.get("name", i))})
            if c2.button("Reject", key=f"rj{i}"):
                r["status"], changed = "rejected", True
                _audit("log_format_rejected", {"format": str(r.get("name", i))})
    if changed:
        SUGGESTIONS.write_text("\n".join(_json.dumps(r) for r in rows) + "\n", "utf-8")
        st.rerun()


def page_soc() -> None:
    st.header("Connect SOC")
    _howto(
        "1. Choose your SIEM: the steps for it appear below.\n"
        "2. Tick where alerts should go (file, syslog/CEF, webhook) and fill in host / "
        "URL / token.\n"
        "3. Press **💾 Save**, then **🔌 Test connection**: every output shows ✅ or ❌ "
        "with the reason."
    )
    s = sinks.load_settings()
    preset = st.selectbox(
        "Your SIEM",
        list(sinks.PRESETS),
        index=list(sinks.PRESETS).index(s.get("preset", "Generic")),
    )
    if preset != s.get("preset"):
        s = sinks.apply_preset(s, preset)
    st.info("📘 **How to connect " + preset + ":**")
    st.code(sinks.preset_help(preset), language="text")

    st.subheader("Send alerts out")
    s["jsonl"]["enabled"] = st.checkbox(
        "Write alerts to a file (SIEM agent reads it)", s["jsonl"]["enabled"]
    )
    if s["jsonl"]["enabled"]:
        c = st.columns([3, 2])
        c[0].caption(f"File: `{s['jsonl']['path']}`")
        jf = list(sinks.JSON_FORMATS)
        s["jsonl"]["format"] = c[1].selectbox("File format", jf, jf.index(s["jsonl"].get("format", "raw")),
                                              format_func=sinks.JSON_FORMATS.__getitem__)  # fmt: skip
    s["syslog"]["enabled"] = st.checkbox("Send Syslog", s["syslog"]["enabled"])
    if s["syslog"]["enabled"]:
        c = st.columns(4)
        s["syslog"]["host"] = c[0].text_input("Host / IP", s["syslog"]["host"])
        protos = ["udp", "tcp", "tls"]
        s["syslog"]["protocol"] = c[2].selectbox("Protocol", protos, protos.index(s["syslog"]["protocol"]),
                                                 help="tls = encrypted (RFC 5425), usually port 6514")  # fmt: skip
        default_port = 6514 if s["syslog"]["protocol"] == "tls" and int(s["syslog"]["port"]) == 514 else int(s["syslog"]["port"])
        s["syslog"]["port"] = c[1].number_input("Port", 1, 65535, default_port)
        sf = list(sinks.SYSLOG_FORMATS)
        s["syslog"]["format"] = c[3].selectbox("Format", sf, sf.index(s["syslog"].get("format", "cef")),
                                               format_func=sinks.SYSLOG_FORMATS.__getitem__)  # fmt: skip
        if s["syslog"]["protocol"] == "tls":
            c = st.columns(3)
            s["syslog"]["ca_file"] = c[0].text_input("CA certificate file (empty = Windows store)",
                                                     s["syslog"].get("ca_file", ""))  # fmt: skip
            s["syslog"]["client_cert"] = c[1].text_input("Client certificate (optional)",
                                                         s["syslog"].get("client_cert", ""))  # fmt: skip
            s["syslog"]["client_key"] = c[2].text_input("Client key (optional)",
                                                        s["syslog"].get("client_key", ""))  # fmt: skip
            s["syslog"]["verify_tls"] = not st.checkbox(
                "Do not check the receiver's certificate", not s["syslog"].get("verify_tls", True),
                help="Only for testing: traffic could then be intercepted.")  # fmt: skip
    s["webhook"]["enabled"] = st.checkbox(
        "Send to a webhook / HTTP collector", s["webhook"]["enabled"]
    )
    if s["webhook"]["enabled"]:
        s["webhook"]["url"] = st.text_input("URL", s["webhook"]["url"])
        s["webhook"]["token"] = st.text_input(
            "Token (optional)", s["webhook"]["token"], type="password"
        )
        s["webhook"]["header_name"] = st.text_input(
            "Token header", s["webhook"]["header_name"] or "Authorization"
        )
        wf = {"json": "SplitShield alert JSON", "splunk_hec": "Splunk HEC envelope",
              "ecs": "ECS (Elastic)", "ocsf": "OCSF Detection Finding"}  # fmt: skip
        cur = s["webhook"].get("format", "json")
        s["webhook"]["format"] = st.selectbox("Body format", list(wf), list(wf).index(cur if cur in wf else "json"),
                                              format_func=wf.__getitem__)  # fmt: skip
        s["webhook"]["verify_tls"] = not st.checkbox(
            "Trust a self-signed certificate (Splunk's default HEC certificate)",
            not s["webhook"].get("verify_tls", True),
            help="Only for your own SIEM on a trusted network: the certificate is "
            "then not checked, so traffic could be intercepted.",
        )

    with st.expander("Delivery guarantee"):
        st.caption("Alerts that cannot reach the SIEM are queued here and retried in order "
                   "(30 s, doubling up to 1 h). After the time below they move to a dead-letter "
                   "file - never deleted - and can be put back in the queue.")  # fmt: skip
        c = st.columns(3)
        s["delivery"]["max_age_days"] = c[0].number_input("Keep retrying for (days)", 1, 365,
                                                          int(s["delivery"].get("max_age_days", 7)))  # fmt: skip
        s["delivery"]["max_queue"] = c[1].number_input("Largest queue (alerts)", 1000, 50_000_000,
                                                       int(s["delivery"].get("max_queue", 1_000_000)), 1000)  # fmt: skip
        s["forward_audit"] = c[2].checkbox("Also send the audit log (who changed what)",
                                           bool(s.get("forward_audit")))  # fmt: skip
    _delivery_card()
    with st.expander("Privacy"):
        s["pseudonymize"]["enabled"] = st.checkbox(
            "Replace user and host names in everything sent out by a stable code",
            bool(s.get("pseudonymize", {}).get("enabled")),
            help="The SIEM still sees which alerts belong to the same user or host, but not "
                 "the real name. The dashboard here keeps showing the real names.")  # fmt: skip

    st.subheader("Receive logs in")
    s["inbox"]["enabled"] = st.checkbox(
        "Watch a folder for exported logs", s["inbox"]["enabled"]
    )
    s["inbox"]["path"] = st.text_input(
        "Folder (point your SIEM export or log shipper here)", s["inbox"]["path"]
    )
    s["listener"]["enabled"] = st.checkbox(
        "Listen for syslog from the SIEM / devices (UDP)", s["listener"]["enabled"]
    )
    s["listener"]["port"] = st.number_input(
        "Listen port", 1024, 65535, int(s["listener"]["port"])
    )
    s["auto_block"] = st.checkbox(
        "Auto-block severe attackers in Windows Firewall", s["auto_block"]
    )

    c1, c2 = st.columns(2)
    if c1.button("Save", type="primary"):
        sinks.save_settings(s)
        _audit("siem_settings_saved")
        st.success("✅ Saved. Restart monitoring for listener/inbox changes to apply.")
    if c2.button("Test connection"):
        sinks.save_settings(s)
        for k, v in sinks.test_connection(s).items():
            (st.success if v == "ok" else st.error)(
                f"{'✅' if v == 'ok' else '❌'} {k}: {v}"
            )


def _delivery_card() -> None:
    """Per SIEM connection: delivered, waiting, dead-lettered, last error."""
    from src.soc import delivery

    stat = delivery.status()
    if not stat:
        return
    rows = []
    for sink, v in stat.items():
        state = ("retrying later" if v["breaker_open"] else "waiting" if v["queued"] else "ok")
        rows.append({"to": sink, "state": state, "delivered": v["delivered"] or 0,
                     "waiting": v["queued"], "of which retried": v["retried"] or 0,
                     "dead-lettered": v["dead"] or 0,
                     "oldest waiting": _t(v["oldest"]) if v["oldest"] else "",
                     "last error": (v["last_error"] or "")[:80]})  # fmt: skip
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    if any(r["waiting"] for r in rows) and c1.button("Retry now"):
        st.write(delivery.flush(force=True))
        _audit("delivery_retry")
    if any(r["dead-lettered"] for r in rows) and delivery.dead_letters(1) and c2.button(
        "Put dead-lettered alerts back in the queue"
    ):
        n = delivery.requeue_dead()
        _audit("delivery_requeue", {"alerts": n})
        st.success(f"{n} alerts queued again.")


def page_results() -> None:
    st.header("Project results")
    _howto(
        "**Research report**: the registered experiments and hypotheses. **Extra "
        "datasets**: the detector recipe tested on new datasets plus the log-mapping "
        "test. " + HELP["macro_f1"]
    )
    import re

    figs = sorted((RESULTS_DIR / "figures").glob("*.png"))
    research, extra = st.tabs(["Research report", "Extra datasets & log mapping"])
    with research:
        rep = RESULTS_DIR / "FINAL_REPORT.md"
        if rep.is_file():
            st.markdown(rep.read_text("utf-8"))
        else:
            st.info("⏳ The final report appears here when the one-click run finishes.")
        for fig in (f for f in figs if not f.name.startswith("extra_")):
            st.image(str(fig), caption=fig.name)
    with extra:
        rep = RESULTS_DIR / "extra" / "EXTRA_DATASETS.md"
        if not rep.is_file():
            st.info(
                "⏳ Run `python -m scripts.extra_datasets` to test the extra datasets."
            )
            return
        # figures are shown below as images; drop the report's relative image links
        st.markdown(re.sub(r"!\[[^\]]*\]\([^)]*\)\s*", "", rep.read_text("utf-8")))
        for fig in (f for f in figs if f.name.startswith("extra_")):
            st.image(str(fig), caption=fig.name)
        for name, title in (("phishing", "🎣 Phishing URLs"), ("malware", "🦠 Malware files"),
                            ("ics", "🏭 Industrial control"), ("insider", "👤 Insider threat"),
                            ("followup_T1_present_classes", "🔁 T1 follow-up")):  # fmt: skip
            md = RESULTS_DIR / "extra" / f"{name}.md"
            if md.is_file():
                with st.expander(title):
                    st.markdown(
                        re.sub(r"!\[[^\]]*\]\([^)]*\)\s*", "", md.read_text("utf-8"))
                    )
                    for fig in (f for f in figs if f.name.startswith(f"mlarea_{name}")):
                        st.image(str(fig), caption=fig.name)


@st.cache_resource
def _shield_assistant(model_mtime: float):  # noqa: ANN202, ARG001 - cache key
    from src.nlp.assistant import ShieldAssistant

    bot = ShieldAssistant()
    bot.start_prewarm()  # top incidents' narratives ready before anyone asks
    return bot


def shield_assistant():  # noqa: ANN201 - ShieldAssistant
    """The assistant, reloaded whenever feedback promoted a new intent model."""
    from src.nlp import feedback

    try:
        mt = feedback.MODEL.stat().st_mtime
    except OSError:
        mt = 0.0
    return _shield_assistant(mt)


def _learn_in_background() -> None:
    import threading

    from src.nlp import feedback

    def _run() -> None:
        from src.soc import evolution

        if not evolution.allowed("intent"):  # frozen: the rating waits for later
            return
        try:
            feedback.quick_update()
        except Exception:  # noqa: BLE001 - learning must never break the chat
            pass

    threading.Thread(target=_run, daemon=True).start()


def _rate_last_answer() -> None:
    """👍 / 👎 under the latest reply; every rating teaches the assistant (gated)."""
    from src.nlp import feedback

    last = st.session_state.get("last_qa")
    if not last:
        return
    q, used = last
    st.caption("Was this answer useful? Your rating teaches the assistant "
               "(kept only if it does not get worse on held-out questions).")  # fmt: skip
    c1, c2, _ = st.columns([1, 1, 6])
    if c1.button("", key="rate_up"):
        feedback.record(q, used, True)
        _learn_in_background()
        st.session_state.pop("last_qa", None)
        st.toast("Thanks! Learning from it.")
        st.rerun()
    if c2.button("", key="rate_down"):
        st.session_state["rating_down"] = True
    if st.session_state.get("rating_down"):
        opts = ["(not sure)"] + feedback.intents()
        pick = st.selectbox("What were you asking for?", opts, key="rate_pick")
        if st.button("Save feedback", key="rate_save"):
            feedback.record(q, used, False, None if pick == opts[0] else pick)
            if pick != opts[0]:
                _learn_in_background()
            st.session_state.pop("last_qa", None)
            st.session_state.pop("rating_down", None)
            st.toast("Thanks! Saved.")
            st.rerun()
    s = feedback.state()
    if s:
        st.caption(f"Last learning step {s.get('ts', '')}: {s.get('status', '')} · "
                   f"{s.get('rated', 0)} rated questions · held-out accuracy "
                   f"{s.get('heldout_challenger', 0):.1%}")  # fmt: skip


def page_ask() -> None:
    st.header("Ask SplitShield")
    _howto(
        "Type a question in plain English; spelling mistakes are fine. Every answer "
        "starts with a one-line reply, then the evidence, then the details; when an "
        "attack is active you get numbered *What to do now* steps first. The assistant "
        "remembers the topic, so *is it still going on?* works after asking about an "
        "attack. Click a suggested question below an answer to ask it."
    )
    labels = {"auto": "🔄 Auto (adapts to the question)", "beginner": "🌱 Beginner",
              "expert": "🧑‍💻 Expert"}  # fmt: skip
    mode = st.radio("Answer style", list(labels), format_func=labels.__getitem__,
                    horizontal=True, key="ask_mode")  # fmt: skip
    bot = shield_assistant()
    st.caption(
        f"Answers come from your alerts and incidents; narratives are written by {bot.lm_name}. "
        "Try: *what attacks are happening today?* · *what did 203.0.113.45 do?* · "
        "*how do I stop brute force?* · *who is attacking us?* · "
        "*what is a brute force attack?* · *is there any defense evasion happening?*"
    )
    hist = st.session_state.setdefault("chat", [])
    for role, text in hist:
        st.chat_message(role).markdown(text)
    # a follow-up the assistant offered (thin / mixed evidence) that the analyst
    # clicked is asked like a typed question
    q = st.session_state.pop("ask_next", None) or st.chat_input(
        "🔎 Ask about attacks, IPs, techniques, results…"
    )
    if q:
        st.chat_message("user").markdown(q)
        follow: list[str] = []
        try:
            # previous turn's subject: "is it still going on?" knows what "it" is
            ans = bot.answer(q, st.session_state.get("chat_ctx"), mode=mode)
            st.session_state["chat_ctx"] = ans.get("context")
            text = (
                ans["text"]
                + f"\n\n_intent: {ans['intent']} · style: {ans.get('style', 'normal')}_"
            )
            follow = ans.get("follow_up", [])
            # the rating teaches the corrected wording (typos fixed) of the question
            st.session_state["last_qa"] = (ans.get("corrected", q), ans["intent"])
            st.session_state.pop("rating_down", None)
        except Exception as exc:  # noqa: BLE001 - never crash the chat on a bad question
            text = f"Sorry, I could not answer that ({type(exc).__name__}). Try rephrasing."
        st.chat_message("assistant").markdown(
            text
        )  # never raw HTML: log data is untrusted
        if "ans" in locals() and ans.get("search", {}).get("rows"):
            st.dataframe(pd.DataFrame(ans["search"]["rows"]), use_container_width=True)
        # big searches show "10,000+": offer the exact count (owner 2026-10-16)
        capped = "ans" in locals() and ans.get("search", {}).get("capped")
        st.session_state["count_q"] = q if capped else None
        st.session_state.pop("count_exact", None)
        for kind, path in ("ans" in locals() and ans.get("report_files") or {}).items():
            p = Path(path)
            if p.is_file():
                st.download_button(f"Download report ({kind})", p.read_bytes(),
                                   file_name=p.name, key=f"dl_{p.name}")  # fmt: skip
        hist += [("user", q), ("assistant", text)]
        st.session_state["follow_ups"] = follow
    if cq := st.session_state.get("count_q"):
        if st.button("Count exactly", key="count_exact_btn"):
            from src.soc import eventindex

            st.session_state["count_exact"] = eventindex.search_count(cq)
        if (n := st.session_state.get("count_exact")) is not None:
            st.info(f"Exactly {n:,} stored events match: {cq}")
    _rate_last_answer()
    for i, f in enumerate(st.session_state.get("follow_ups", [])):
        if st.button(f"{f}", key=f"follow_{i}"):
            st.session_state["ask_next"] = f
            st.session_state["follow_ups"] = []
            st.rerun()


def page_drift() -> None:
    """Live drift vs training data + what the autopilot decided (owner 2026-10-14)."""
    import json as _json

    from scripts import autopilot

    st.header("Drift & autopilot")
    _howto(
        HELP["psi"]
        + " The autopilot checks this every hour and lists its decisions below."
    )
    st.caption(
        "Hourly, the autopilot compares live traffic with the training data (PSI per "
        "feature), retrains the detector from analyst reviews (kept only if it "
        "improves), and improves the assistant once a night. Auto-blocking stays manual."
    )
    det = detector()
    state = {}
    if autopilot.STATE.is_file():
        state = _json.loads(autopilot.STATE.read_text("utf-8"))
    if state.get("notice"):
        st.warning(f"⚠️ {state['notice']}")
    if st.button("Check drift now") and det is not None:
        state["drift"] = autopilot.drift_now(det)
    d = state.get("drift") or {}
    status = d.get("status", "no check yet")
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", DRIFT_LABEL.get(status, status))
    c2.metric(
        "Mean PSI",
        f"{d['mean_psi']:.3f}" if "mean_psi" in d else "-",
        help=HELP["psi"],
    )
    c3.metric(
        "Features shifted",
        f"{d['share_strong']:.0%}" if "share_strong" in d else "-",
    )
    if d.get("top"):
        st.subheader("Most shifted features")
        st.bar_chart({t["feature"]: t["psi"] for t in d["top"]})
        st.caption("🟢 PSI < 0.1 stable · 🟡 0.1-0.25 moderate · 🔴 > 0.25 large shift")
    elif status in ("not_enough_data", "no_reference"):
        st.info(
            d.get("note") or f"Need at least {d.get('min_rows', 200)} scored flows."
        )
    st.subheader("Self-evolving detection")
    from src.soc import evolve

    hist = evolve.history()
    if not hist:
        st.info("⏳ No evolution iterations yet (the autopilot runs one per hour).")
    else:
        curve = pd.DataFrame(
            [
                {
                    "iteration": h["iteration"],
                    "attack steps caught": h["champion"]["steps_caught"],
                    "false-alarm rate": h["champion"]["false_alarm_rate"],
                    "events / second": h["champion"]["events_per_second"],
                }
                for h in hist
                if "champion" in h
            ]
        ).set_index("iteration")
        c1, c2 = st.columns(2)
        c1.line_chart(curve[["attack steps caught", "false-alarm rate"]])
        c2.line_chart(curve[["events / second"]])
        kept = [h for h in hist if h.get("kept")]
        st.caption(
            f"{len(kept)} improvement(s) kept out of {len(hist)} iteration(s). Every change "
            "must keep detection, not add false alarms, and improve at least one of them "
            "or speed — otherwise it is rolled back."
        )
        for h in reversed(kept[-5:]):
            st.write(f"✅ iteration {h['iteration']}: {h['kept']}")

    st.subheader("Autopilot decisions")
    rows = engine.read_jsonl(autopilot.LOG, last=50) if autopilot.LOG.is_file() else []
    if not rows:
        st.write("⏳ No decisions yet.")
    for r in reversed(rows):
        st.write(
            f"{ACTION_ICON.get(r.get('action', ''), '•')} {_t(r['ts'])} - "
            f"**{r.get('action')}** {r.get('why', r.get('reason', ''))}"
        )


def _task(name: str) -> bool:
    from src.soc import osops

    return osops.task_installed(name)


def _service(name: str) -> bool:
    from src.soc import osops

    return osops.service_running(name)


SENSOR_TITLE = {"windows": "Sysmon installed (process / credential-theft visibility)",
                "linux": "auditd running (process / file / sign-in visibility)",
                "macos": "Audit logging on (process / sign-in visibility)"}
SENSOR_FIX = {"windows": "As Administrator: `powershell -ExecutionPolicy Bypass -File scripts\\install_sysmon.ps1`",
              "linux": "`sudo apt install auditd` (or dnf) and `sudo systemctl enable --now auditd`",
              "macos": "macOS audits by default; give the .venv Python Full Disk Access if logs are unreadable."}  # fmt: skip


def page_start() -> None:
    """Setup checklist (owner 2026-10-14): each item green/red with its fix."""
    from src.nlp import selfimprove
    from src.soc import osops, users

    st.header("Getting started")
    _howto("Work down the list. Each item marked To do says how to fix it. Commands run in a terminal "
           "opened in the project folder.")  # fmt: skip
    s = sinks.load_settings()
    siem = [k for k in ("syslog", "webhook") if s.get(k, {}).get("enabled")]
    events = engine.event_count()
    nlp_missing = selfimprove.ready()["missing"]
    items = [
        ("Detector trained", detector() is not None,
         "Run the one-click setup (LAUNCH.cmd); it trains the detector."),
        ("One account per person (Users & roles)", len(users.load()["users"]) > 1,
         "Admin: open **Users & roles** and add each analyst / viewer with their own role."),
        (SENSOR_TITLE[osops.PLATFORM], _service(osops.SENSOR_SERVICE[osops.PLATFORM]),
         SENSOR_FIX[osops.PLATFORM]),
        ("This computer's own logs are read", _task("SplitShield-XAI live logs"),
         "Run `python -m src.soc.services install` (the one-click setup does it)."),
        ("SIEM connected", bool(siem),
         "Open **Connect SOC**, pick your SIEM, then Save and Test connection."),
        ("Logs or traffic ingested", events > 0,
         "Open **Analyze logs** and drag in files, or press **Load demo attack data**."),
        ("Monitoring running", bool(engine.monitor_running()),
         "Open **Live monitor** and press **Start monitoring**."),
        ("Assistant trained", not nlp_missing,
         "Run the one-click setup; it trains the assistant." if nlp_missing else ""),
        ("Autopilot (hourly) scheduled", _task("SplitShield-XAI autopilot"),
         "Run `python -m src.soc.services install` (the one-click setup does it)."),
        ("Daily backup scheduled", _task("SplitShield-XAI backup"),
         "Run `python -m src.soc.services install` (the one-click setup does it)."),
    ]  # fmt: skip
    done = sum(ok for _, ok, _ in items)
    st.progress(done / len(items), text=f"{done} of {len(items)} done")
    for title, ok, fix in items:
        with st.container(border=True):
            c1, c2 = st.columns([6, 1])
            c1.markdown(f"**{title}**")
            with c2:
                st.badge("Done" if ok else "To do", color="green" if ok else "orange")
            if not ok and fix:
                st.caption(fix)
    _live_logs_card()


def _live_logs_card() -> None:
    """This PC's Windows logs, watched live (src/soc/winlive.py, owner 2026-10-16)."""
    import time as _t

    from src.soc import osops, winlive

    if osops.PLATFORM != "windows":
        from src.soc import oslive

        st.subheader("This computer's logs (live)")
        for src_name, s_txt in oslive.status().items():
            st.write(f"{'✅' if s_txt == 'ok' else '❌'} **{src_name}**: {'reading' if s_txt == 'ok' else s_txt}")
        if not _task("SplitShield-XAI live logs"):
            st.info("Not watching yet. Start it with: `python -m src.soc.services install`")
        if osops.PLATFORM == "linux":
            st.caption("No read permission? Once: `sudo usermod -aG adm,systemd-journal $USER`, "
                       "then sign out and in.")  # fmt: skip
        return
    st.subheader("This PC's Windows logs (live)")
    stt = winlive.load_state()
    last = stt.get("last_poll")
    if not last:
        st.info("Not watching yet. Start it with: "
                "`powershell -ExecutionPolicy Bypass -File scripts\\install_winlive_task.ps1`")  # fmt: skip
        return
    age = _t.time() - float(last.get("ts", 0))
    st.caption(
        f"Last check {int(age)} s ago"
        + (" ⚠️ (the watcher seems stopped)" if age > 120 else "")
    )
    for name, status in last.get("logs", {}).items():
        ok = status == "ok" or status.endswith("new records")
        st.write(f"{'✅' if ok else '❌'} **{name}**: {'reading' if ok else status}")
    if any("Event Log Readers" in str(v) for v in last.get("logs", {}).values()):
        st.caption("👉 Once, in an Administrator PowerShell, then sign out and in: "
                   '`net localgroup "Event Log Readers" $env:USERNAME /add`')  # fmt: skip


def _place_card() -> None:
    """Where the SOC runs: guessed, confirmed or picked (src/soc/places.py)."""
    from src.soc import places

    st.subheader("Where is this SOC running?")
    _howto("Alerting adapts to the place: how many incidents are forwarded per day and "
           "how strongly linked findings raise severity. It never switches a detector "
           "off, and critical incidents are always forwarded.")  # fmt: skip
    stt = places.check(engine.recent_events())
    profs = places.profiles()
    if stt.get("notice"):
        st.warning(f"⚠️ {stt['notice']}")
    cur = places.current()
    st.write(f"**Now:** {cur['name'].replace('_', ' ')}: {cur['description']}")
    if stt.get("guess"):
        st.caption(f"My guess from your logs: **{stt['guess'].replace('_', ' ')}** "
                   f"({'; '.join(stt.get('reasons', []))})")  # fmt: skip
    names = list(profs)
    pick = st.selectbox("Place", names, index=names.index(cur["name"]),
                        format_func=lambda n: f"{n.replace('_', ' ')}: {profs[n]['description']}")  # fmt: skip
    modes = {"confirm": "Guess it, I confirm (recommended)",
             "manual": "Always use my pick", "auto": "Fully automatic (tells me what changed)"}  # fmt: skip
    mode = st.radio("How to choose", list(modes), index=list(modes).index(stt["mode"]),
                    format_func=modes.get, horizontal=True)  # fmt: skip
    if st.button("Save place"):
        places.set_place(pick, mode=mode)
        _audit("place_set", {"place": pick, "mode": mode})
        st.success(f"Saved: {pick.replace('_', ' ')} ({modes[mode].lower()}).")


def _backup_card(s: dict) -> None:
    from scripts import backup

    st.subheader("Backup folder")
    dest = st.text_input("Daily backups go to", value=str(backup.destination()))
    if st.button("Save backup folder"):
        s["backup_dest"] = dest.strip()
        sinks.save_settings(s)
        _audit("backup_folder", {"to": dest.strip()})
        st.success("Saved; used from the next daily backup.")


def _fit_card() -> None:
    """Over- / underfitting per model from the weekly learning-curve report."""
    from src.eval import fitcheck

    fit = fitcheck.model_report()
    st.subheader("Over- / underfitting")
    if not fit["models"]:
        st.caption("No fit report yet: `python -m scripts.fit_report` (about 16 minutes, 11 GB RAM).")
        return
    rows = [{"model": m, "verdict": v["verdict"], "findings": "; ".join(v["findings"])[:160]}
            for m, v in fit["models"].items()]  # fmt: skip
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"Report {fit['age_days']} days old; rebuilt weekly at night by the autopilot. "
               "A flagged model is reported, never changed automatically.")  # fmt: skip


def page_evolution() -> None:
    """One self-improvement controller around all 9 models (owner 2026-10-15)."""
    import threading

    from src.soc import codeopt, controller

    st.header("Self-improvement & health")
    _howto("Every model learns from new labelled data and your feedback, but a change is "
           "kept only if it is **not worse on fixed held-out data and not slower**; "
           "otherwise it is rolled back. Code speed-ups are only *proposed*: nothing "
           "changes until you approve it below. Decisions also pop up as Windows "
           "notifications.")  # fmt: skip
    t_health, t_dec, t_teach, t_speed = st.tabs(
        ["Health", "Decisions", "Teach a model", "Speed & code proposals"]
    )
    with t_health:
        _fit_card()
        c1, c2 = st.columns(2)
        if c1.button("Check health now (about a minute)"):
            with st.spinner("Loading every model and timing it…"):
                st.session_state["health"] = controller.health()
        if c2.button("Learn from the inboxes now"):
            with st.spinner("Running one improvement cycle…"):
                st.session_state["cycle"] = controller.cycle()
        h = st.session_state.get("health")
        if h:
            st.subheader("Models")
            st.table({m["title"]: ["✅" if m["ok"] else "❌",
                                   f"{m['speed']:,} {m['unit']}" if m["ok"] else m.get("error", "")]
                      for m in h["models"].values()})  # fmt: skip
        st.subheader("Automatic parts")
        comps = (h or {}).get("components") or controller.components()
        st.table(
            {
                k.replace("_", " "): ["✅" if v["ok"] else "❌", v["detail"]]
                for k, v in comps.items()
            }
        )
        for x in st.session_state.get("cycle", []):
            st.write(f"**{x['model']}**: {x['decision']}")

    with t_dec:
        hist = controller.history(30)
        if hist:
            st.table([{"time": e["ts"], "model": e["model"], "why": e.get("why", ""),
                       "decision": e.get("decision", "")} for e in reversed(hist)])  # fmt: skip
        else:
            st.caption("No decisions yet.")

    with t_teach:
        reg = controller.registry()
        pick = st.selectbox("Model", [m.name for m in reg.values() if m.inbox],
                            format_func=lambda n: reg[n].title)  # fmt: skip
        st.caption({"flow": "CICFlowMeter CSV with a Label column.",
                    "netflow": "NetFlow v9 CSV with a Label column (0/1 or names).",
                    "zeek": "Labelled Zeek conn.log (IoT-23 style label field).",
                    "phishing": "CSV with url,label (1 = phishing).",
                    "malware": "EMBER feature JSONL with label 0/1.",
                    "ics": "Plant sensor CSV at 1 Hz (normal rows are learned).",
                    "insider": "User-day feature CSV with a label column."}.get(pick, ""))  # fmt: skip
        up = st.file_uploader("File", key="inbox_up")
        if up is not None and st.button("Add to inbox"):
            box = controller.INBOX / pick
            box.mkdir(parents=True, exist_ok=True)
            (box / Path(up.name).name).write_bytes(up.getvalue())
            st.success("Saved. It is learned in the next cycle (hourly), or press "
                       "*Learn from the inboxes now*.")  # fmt: skip

    with t_speed:
        st.subheader("Speed (measured nightly)")
        prof = codeopt._read(codeopt.PROFILE)
        if st.button("Profile now (a few minutes)"):
            with st.spinner("Timing every workload…"):
                prof = codeopt.profile()
        if prof:
            prev = (prof.get("previous") or {}).get("throughput") or {}
            st.table({k: [f"{v:,.0f}/s" if v else "error",
                          f"{prev[k]:,.0f}/s" if prev.get(k) else "—"]
                      for k, v in prof["throughput"].items()})  # fmt: skip
            st.caption(
                "Columns: now · previous profile. Slowest project functions (rule engine):"
            )
            st.table(prof.get("hotspots", [])[:10])

        st.subheader("Code speed-up proposals (you decide)")
        if st.button(
            "Look for speed-ups now (tests each one in a copy; can take 10+ min)"
        ):
            threading.Thread(target=codeopt.propose, daemon=True).start()
            st.info("Started in the background; refresh this page later.")
        props = codeopt.proposals()
        if not props:
            st.caption(
                "No proposals yet. They are produced nightly from the speed profile."
            )
        for p in props:
            with st.expander(f"{p['status']} · {p['file']} · {p['description']}"):
                if p.get("trial"):
                    t = p["trial"]
                    st.write(f"Benchmark: {t['before']:,.0f} → {t['after']:,.0f} lines/s "
                             f"({t['gain']:+.1%}); tests: {'passed' if t['tests_passed'] else 'FAILED'} "
                             f"({t['tests']})")  # fmt: skip
                if p.get("diff"):
                    st.code(p["diff"], language="diff")
                b1, b2, b3 = st.columns(3)
                if p["status"] == "ready for approval":
                    if b1.button("✅ Approve & apply", key=f"ap_{p['id']}"):
                        with st.spinner("Applying and re-running all tests…"):
                            st.write(codeopt.apply(p["id"]))
                            _audit("code_proposal_applied", {"id": p["id"]})
                    if b2.button("✖ Reject", key=f"rj_{p['id']}"):
                        codeopt.reject(p["id"])
                        _audit("code_proposal_rejected", {"id": p["id"]})
                        st.rerun()
                if p["status"] == "applied" and b3.button(
                    "↩ Revert", key=f"rv_{p['id']}"
                ):
                    st.write(codeopt.revert(p["id"]))
                    _audit("code_proposal_reverted", {"id": p["id"]})


def page_updates() -> None:
    """Internet update agent ON/OFF (owner 2026-10-15): Sigma rules, ATT&CK and
    threat-intel feeds, applied only when validation, the false-alarm check and the
    full test suite all pass. Off by default."""
    from src.soc import intel, updater

    st.header("Updates & threat intel")
    _howto("Switch the daily update agent on or off. When on, once a day it downloads "
           "new Sigma rules, the latest MITRE ATT&CK release and threat-intel feeds "
           "(URLhaus, ThreatFox, Feodo, OpenPhish). An update is applied only if it "
           "validates, does not raise false alarms on a benign sample and the full "
           "test suite still passes; otherwise it is rejected and logged.")  # fmt: skip
    on = st.toggle("Update agent enabled", value=updater.get_enabled())
    if on != updater.get_enabled():
        updater.set_enabled(on)
        st.success(f"Update agent switched {'ON' if on else 'OFF'}.")
    if not _task("SplitShield-XAI update agent (daily)"):
        st.caption("👉 To run it daily on its own: `python -m src.soc.services install --only "
                   "\"SplitShield-XAI update agent (daily)\"`. "
                   "It still does nothing while the switch above is off.")  # fmt: skip
    state = updater.load_state()
    c1, c2 = st.columns(2)
    c1.metric("Last run", str(state.get("last_run") or "never")[:16])
    c2.metric("Sources tracked", len(updater.SOURCES))
    if st.button("Check for updates now (takes a few minutes)"):
        with st.spinner("Downloading, validating and testing updates…"):
            rep = updater.run_once(force=True)
        st.write(f"Status: **{rep.get('status', '?')}**")
        if any(v.get("status") == "applied" and k == "sigma"
               for k, v in rep.get("sources", {}).items()):  # fmt: skip
            st.warning(
                "New Sigma rules applied: restart the dashboard and monitor to load them."
            )
        st.json(rep.get("sources", {}), expanded=False)
    if state.get("versions"):
        st.subheader("Current versions")
        st.table({k: [str(v)] for k, v in state["versions"].items()})
    feeds = intel.status()
    st.subheader("Threat-intel feeds")
    if feeds:
        st.table({k: [v["count"], v["fetched"][:16]] for k, v in feeds.items()})
    else:
        st.caption("No feeds downloaded yet.")
    q = st.text_input("Look up an IP, domain or URL in the feeds")
    if q:
        hits = (intel.lookup_url(q) if "://" in q else
                intel.lookup_ip(q) if intel.norm_ip(q) else intel.lookup_domain(q))  # fmt: skip
        st.write(hits or "No match in the downloaded feeds.")
    hist = state.get("history", [])[-15:]
    if hist:
        st.subheader("Recent decisions")
        st.json(list(reversed(hist)), expanded=False)
    for src in ("sigma", "attack"):
        if st.button(f"Roll back {src} to the previous version", key=f"rb_{src}"):
            st.write(updater.rollback(src))


def page_performance() -> None:
    """Memory / CPU the SOC may use (owner 2026-10-16): automatic, a memory budget in
    GB, or a target in logs per second (src/soc/perf.py)."""
    import json

    from src.soc import perf

    st.header("Performance")
    _howto("How much memory and CPU the SOC may use. More means big log files are analysed "
           "faster (several worker processes) and searches stay fast on large stores. "
           "Answers and detections are the same in every setting; only the speed changes.")  # fmt: skip
    s = perf.load_settings()
    cal = perf.calibration()
    labels = {
        "auto": "Automatic (recommended)",
        "memory": "Memory budget",
        "speed": "Target speed",
    }
    mode = st.radio("How to decide", list(labels), index=list(labels).index(s["mode"]),
                    format_func=labels.__getitem__, horizontal=True, key="perf_mode")  # fmt: skip
    total = perf.total_ram_gb()
    gb, lps = s["memory_gb"], s["target_lps"]
    if mode == "memory":
        gb = st.slider("Memory the SOC may use (GB)", 0.5, float(max(1, int(total) - 2)),
                       float(min(gb, max(1, int(total) - 2))), 0.5)  # fmt: skip
    elif mode == "speed":
        lps = float(
            st.number_input(
                "Logs per second to aim for", 1000, 10_000_000, int(lps), 5000
            )
        )
    preview = perf.plan({"mode": mode, "memory_gb": gb, "target_lps": lps})
    c1, c2, c3 = st.columns(3)
    c1.metric("Log analysis", f"{preview['predicted_lps']:,} logs/s")
    c2.metric("Workers", preview["workers"])
    c3.metric("Memory used", f"{preview['memory_mb'] / 1024:.1f} GB")
    st.caption(f"{preview['note']}. Search cache {preview['index_cache_mb']} MB. "
               f"This PC: {perf.cores()} CPU cores, {total:.1f} GB memory, "
               f"{perf.free_ram_gb():.1f} GB free now.")  # fmt: skip
    if mode == "speed" and not preview.get("target_reached", True):
        st.warning(preview["note"])
    ftd = int(st.number_input(
        "Full-text search index: keep the last N days (0 = all)", 0, 3650,
        int(perf.load_settings().get("fulltext_days", 0)), 1,
        help="A smaller index on disk. Older events stay searchable by IP, user, host, technique and time; "
             "a word search that reaches further back reads the stored events instead (slower, same results)."))  # fmt: skip
    if st.button("Save", type="primary"):
        perf.save_settings(mode, memory_gb=gb, target_lps=lps, fulltext_days=ftd)
        _audit("performance_settings", {"mode": mode, "memory_gb": gb, "target_lps": lps, "fulltext_days": ftd})
        st.success("Saved. It applies to the next file or search.")
    st.subheader("Measured on this PC")
    curve = cal.get("curve") or {}
    if curve:
        st.caption(f"Calibrated {cal['measured']}: one worker needs about {cal['worker_mb']:.0f} MB. "
                   "End-to-end speed (analyse + store + alerts) by number of workers:")  # fmt: skip
        st.dataframe(pd.DataFrame([{"workers": int(k), "logs per second": int(v)} for k, v in
                                   sorted(curve.items(), key=lambda kv: int(kv[0]))]),
                     hide_index=True, use_container_width=True)  # fmt: skip
        best = max(curve, key=lambda k: curve[k])
        st.caption(f"Above {best} worker(s) storing the events is the slower step on this PC, so more "
                   "workers do not add speed; extra memory then goes to the search cache.")  # fmt: skip
    else:
        st.info(
            "Not calibrated yet: predictions use cautious defaults. Calibrate once (about a minute)."
        )
    if st.button("Calibrate now (about a minute)"):
        with st.spinner("Measuring this PC..."):
            perf.calibrate()
        st.rerun()
    last = perf._soc_dir() / "performance_last.json"
    if last.is_file():
        d = json.loads(last.read_text("utf-8"))
        st.caption(f"Last big file: {d['lines']:,} lines in {d['seconds']} s "
                   f"({d['lps']:,} logs/s, {d['workers']} worker(s)) at {d['when']}.")  # fmt: skip
    bench = Path("results/extra/speed_benchmark.json")
    if bench.is_file():
        rows = json.loads(bench.read_text("utf-8")).get("rows", [])
        if rows:
            st.subheader("Search and answer speed at large stores")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _retention_card() -> None:
    """How long events and alerts stay in the live store (src/soc/retention.py)."""
    from src.soc import retention

    st.subheader("Data retention")
    s = retention.settings()
    sz = retention.sizes()
    st.caption(f"Live store {sz['store']:,.0f} MB · search index {sz['index']:,.0f} MB · "
               f"archive {sz['archive']:,.0f} MB. The audit log is never trimmed.")  # fmt: skip
    c1, c2, c3 = st.columns(3)
    on = c1.toggle("Retention on", value=bool(s["enabled"]))
    days = c2.number_input("Keep events for (days)", 1, 3650, int(s["days"]))
    mode = c3.selectbox("Older events", list(retention.MODES), index=list(retention.MODES).index(s["mode"]),
                        format_func=retention.MODES.__getitem__)  # fmt: skip
    keep_arch = st.number_input("Delete archives older than (days, 0 = keep forever)", 0, 36500,
                                int(s["archive_days"]), disabled=mode != "archive")  # fmt: skip
    if mode == "delete":
        st.warning("Deleted events cannot be recovered. Archive keeps them compressed instead.")
    b1, b2 = st.columns(2)
    if b1.button("Save retention"):
        try:
            retention.save({"enabled": on, "days": days, "mode": mode, "archive_days": keep_arch}, who=_me())
            st.success("Saved. It runs every night; 'Apply now' runs it at once.")
        except ValueError as exc:
            st.error(str(exc))
    if b2.button("Apply now"):
        with st.spinner("Moving old events (the search index rebuilds afterwards)..."):
            res = retention.apply(who=_me())
        st.write(res)


def page_places() -> None:
    """Where the SOC runs (alerting budget) and where backups go."""
    st.header("Places & backup")
    _howto("Where this SOC runs changes only how much is forwarded and how strongly linked "
           "findings raise severity; backups keep models, results and code for 7 days.")  # fmt: skip
    _place_card()
    _backup_card(sinks.load_settings())
    _retention_card()


def _me() -> str:
    return str(st.session_state.get("user", ""))


def _can(perm: str) -> bool:
    from src.soc import users

    return users.can(_me(), perm)


def _audit(action: str, details: dict | None = None) -> None:
    from src.soc import audit

    audit.record(_me(), action, details or {})


def page_users() -> None:
    """Users, roles and permissions - admin only (owner 2026-10-16)."""
    from src.soc import users

    st.header("Users & roles")
    _howto("Only an admin can add users, change usernames and passwords (the admin's own "
           "too), assign roles and decide what each role may do. Every change is in the "
           "audit log.")  # fmt: skip
    d = users.load()
    me = _me()
    t_users, t_roles = st.tabs(["Users", "Roles & permissions"])
    with t_users:
        st.dataframe(pd.DataFrame([{"user": n, "role": u["role"], "created": u.get("created", ""),
                                    "status": "disabled" if u.get("disabled") else "active"}
                                   for n, u in d["users"].items()]), hide_index=True,
                     use_container_width=True)  # fmt: skip
        with st.form("add_user", clear_on_submit=True):
            st.markdown("**Add a user**")
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Username")
            role = c2.selectbox("Role", list(d["roles"]))
            pw = c3.text_input("Password", type="password")
            if st.form_submit_button("Add user", type="primary"):
                _try(lambda: users.add_user(me, name, pw, role), f"User {name} added.")
        st.markdown("**Change a user**")
        who = st.selectbox("User", list(d["users"]), key="u_pick")
        c1, c2 = st.columns(2)
        with c1.form("u_role"):
            new_role = st.selectbox("Role", list(d["roles"]),
                                    index=list(d["roles"]).index(d["users"][who]["role"]))  # fmt: skip
            if st.form_submit_button("Save role"):
                _try(lambda: users.set_role(me, who, new_role), "Role saved.")
        with c2.form("u_pw", clear_on_submit=True):
            npw = st.text_input("New password", type="password")
            if st.form_submit_button("Set password"):
                _try(lambda: users.set_password(me, who, npw), "Password changed.")
        c3, c4 = st.columns(2)
        with c3.form("u_rename", clear_on_submit=True):
            newname = st.text_input("New username")
            if st.form_submit_button("Rename"):
                def _ren() -> None:
                    users.rename(me, who, newname)
                    if who == me:
                        st.session_state["user"] = newname

                _try(_ren, "Renamed.")
        with c4:
            disabled = bool(d["users"][who].get("disabled"))
            if st.button("Enable" if disabled else "Disable", key="u_dis"):
                _try(lambda: users.set_disabled(me, who, not disabled), "Saved.")
            if st.button("Remove user", key="u_rm"):
                _try(lambda: users.remove_user(me, who), "Removed.")
    with t_roles:
        perms = list(users.PERMISSIONS)
        st.dataframe(pd.DataFrame([{"role": r, **{p: ("yes" if p in ps else "") for p in perms}}
                                   for r, ps in d["roles"].items()]), hide_index=True,
                     use_container_width=True)  # fmt: skip
        st.caption(" · ".join(f"**{p}**: {t}" for p, t in users.PERMISSIONS.items()))
        with st.form("role_edit"):
            st.markdown("**Create a role or change what it may do** (admin is fixed)")
            rname = st.text_input("Role name", value="analyst")
            current = d["roles"].get(rname, [])
            chosen = st.multiselect("Permissions", [p for p in perms if p != "users"],
                                    default=[p for p in current if p != "users"])  # fmt: skip
            if st.form_submit_button("Save role", type="primary"):
                _try(lambda: users.set_role_permissions(me, rname.strip(), chosen), "Role saved.")
        drop = st.selectbox("Delete a role", [r for r in d["roles"] if r != "admin"], key="r_del")
        if st.button("Delete role"):
            _try(lambda: users.delete_role(me, drop), "Role deleted.")


def _try(fn: Any, ok_text: str) -> None:
    try:
        fn()
        st.success(ok_text)
        st.rerun()
    except (ValueError, PermissionError) as exc:
        st.error(str(exc))


def page_audit() -> None:
    """The tamper-evident audit log - admin only."""
    from src.soc import audit

    st.header("Audit log")
    _howto("Every sign-in and every change, with who and when. Each entry is chained to the "
           "previous one: editing or deleting history breaks the chain and shows here.")  # fmt: skip
    v = audit.verify()
    st.badge("Chain intact" if v["ok"] else f"Chain broken at entry {v.get('broken_at')}",
             color="green" if v["ok"] else "red")  # fmt: skip
    rows = audit.read(last=1000)[::-1]
    if rows:
        st.dataframe(pd.DataFrame([{"when": r.get("when"), "who": r.get("who"), "action": r.get("action"),
                                    "ok": r.get("ok"), "details": json.dumps(r.get("details", {}))[:200]}
                                   for r in rows]), hide_index=True, use_container_width=True)  # fmt: skip
    else:
        st.info("No entries yet.")


def page_evolution_mode() -> None:
    """Evolving or frozen, for all models or each one (owner 2026-10-16)."""
    from src.soc import evolution

    st.header("Evolution")
    _howto("Evolving: models keep learning from new labelled data and your reviews, and a "
           "change is kept only if held-out precision and recall do not drop and it is not "
           "slower. Frozen: models never change by themselves; they still detect, alert "
           "and keep collecting reviews, so they can learn later.")  # fmt: skip
    s = evolution.load()
    mode = st.radio("All models", list(evolution.MODES), index=list(evolution.MODES).index(s["mode"]),
                    format_func={"evolving": "Evolving (recommended)", "frozen": "Frozen"}.__getitem__,
                    horizontal=True)  # fmt: skip
    comps = evolution.components()
    frozen = st.multiselect("Freeze single models (while the rest evolves)", comps,
                            default=[c for c in s["frozen"] if c in comps],
                            disabled=mode == "frozen")  # fmt: skip
    if st.button("Save", type="primary"):
        evolution.save(mode, frozen if mode == "evolving" else s["frozen"], who=_me())
        st.success("Saved. The next self-improvement cycle follows it.")
    if s.get("changed"):
        st.caption(f"Last changed {s['changed']}.")


def _template_fields(t: dict, key: str) -> dict:
    """Inputs for a report template (the global one, or this report's copy of it)."""
    from src.soc import reports

    names = {k: v[0] for k, v in reports.SECTIONS.items()}
    out = dict(t)
    out["title"] = st.text_input("Title", t["title"], key=f"{key}_title")
    out["intro"] = st.text_area("Intro text (optional, shown under the title)", t["intro"],
                                key=f"{key}_intro", height=80)  # fmt: skip
    out["sections"] = st.multiselect(
        "Sections, in this order", list(names), default=t["sections"], key=f"{key}_sec",
        format_func=names.__getitem__,
        help=" · ".join(f"{v[0]}: {v[2]}" for v in reports.SECTIONS.values()),
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    out["top_n"] = c1.number_input("Top entities", 1, 50, int(t["top_n"]), key=f"{key}_top")
    out["max_details"] = c2.number_input("Incident write-ups", 1, 100, int(t["max_details"]),
                                         key=f"{key}_det")  # fmt: skip
    out["formats"] = c3.multiselect("Files", list(reports.FORMATS), default=t["formats"],
                                    format_func=reports.FORMATS.__getitem__, key=f"{key}_fmt")  # fmt: skip
    return out


def _period_inputs() -> dict:
    """Period choice -> keyword arguments for reports.generate."""
    import datetime as _dt

    from src.soc import reports

    opts = {**reports.PERIODS, "dates": "Date range", "times": "Time range", "incident": "One incident"}
    pick = st.radio("Period", list(opts), format_func=opts.__getitem__, horizontal=True)
    today = _dt.date.today()
    if pick == "dates":
        c1, c2 = st.columns(2)
        a = c1.date_input("From (first day)", today - _dt.timedelta(days=7))
        b = c2.date_input("To (last day, included)", today)
        start = _dt.datetime.combine(a, _dt.time())
        return {"start": start.timestamp(),
                "end": (_dt.datetime.combine(b, _dt.time()) + _dt.timedelta(days=1)).timestamp()}
    if pick == "times":
        c1, c2, c3, c4 = st.columns(4)
        a = c1.date_input("From day", today, key="rt_a")
        ta = c2.time_input("From time", _dt.time(9, 0), key="rt_ta")
        b = c3.date_input("To day", today, key="rt_b")
        tb = c4.time_input("To time", _dt.time(17, 0), key="rt_tb")
        return {"start": _dt.datetime.combine(a, ta).timestamp(),
                "end": _dt.datetime.combine(b, tb).timestamp()}
    if pick == "incident":
        incs = engine.incidents()
        if not incs:
            st.caption("No incidents stored yet.")
            return {"skip": True}
        iid = st.selectbox("Incident", [i["id"] for i in incs[:200]])
        return {"period": "incident", "incident_id": iid}
    return {"period": pick}


def page_reports() -> None:
    """Report builder (owner 2026-10-16): period, sections, formats; global template;
    schedules; every report can be edited after it is made."""
    from src.soc import reports

    st.header("Reports")
    _howto("Pick a period, press **Make report**, then download it as Word, Markdown or "
           "HTML. What goes in comes from the global template; change it here for one "
           "report, or for every report under **Global template**. A finished report "
           "can still be edited below and saved again.")  # fmt: skip
    t_make, t_tpl, t_sched, t_saved = st.tabs(
        ["Make a report", "Global template", "Schedules", "Saved reports"]
    )
    may_edit = _can("reports")
    with t_make:
        args = _period_inputs()
        with st.expander("What goes in (this report only)"):
            one = _template_fields(reports.load_template(), "one")
        if st.button("Make report", type="primary", disabled=bool(args.get("skip"))):
            try:
                kw = {k: v for k, v in args.items() if k != "skip"}
                made = reports.generate(template=one, who=_me(), **kw)
                st.session_state["report"] = {k: made[k] for k in ("label", "headline", "markdown",
                                                                    "stem", "formats")}  # fmt: skip
                st.session_state["report_files"] = {k: str(v) for k, v in made["files"].items()}
            except ValueError as exc:
                st.error(str(exc))
        rep = st.session_state.get("report")
        if rep:
            st.success(f"{rep['label']}: {rep['headline']}")
            cols = st.columns(max(1, len(st.session_state.get("report_files", {}))))
            for col, (kind, path) in zip(cols, st.session_state.get("report_files", {}).items(),
                                         strict=False):  # fmt: skip
                p = Path(path)
                if p.is_file():
                    col.download_button(f"Download {reports.FORMATS[kind]}", p.read_bytes(),
                                        file_name=p.name, key=f"dlr_{kind}")  # fmt: skip
            edited = st.text_area("Edit the report (Markdown), then save it again", rep["markdown"],
                                  height=360, key=f"edit_{rep['stem']}")  # fmt: skip
            if st.button("Save my edits"):
                files = reports.save_markdown(edited, rep["stem"], rep["formats"])
                rep["markdown"] = edited
                st.session_state["report_files"] = {k: str(v) for k, v in files.items()}
                _audit("report_edited", {"report": rep["stem"]})
                st.success("Saved - the downloads above now hold your edited version.")
            with st.expander("Preview"):
                st.markdown(rep["markdown"])
    with t_tpl:
        if not may_edit:
            st.info("Your role can make reports but not change the global template.")
        g = reports.load_template()
        if g.get("changed"):
            st.caption(f"Last changed {g['changed']}.")
        new = _template_fields(g, "glob")
        c1, c2 = st.columns(2)
        if c1.button("Save global template", type="primary", disabled=not may_edit):
            try:
                reports.save_template(new, who=_me())
                st.success("Saved. New reports and schedules use it.")
            except ValueError as exc:
                st.error(str(exc))
        if c2.button("Back to the default template", disabled=not may_edit):
            reports.save_template(dict(reports.DEFAULT_TEMPLATE), who=_me())
            st.rerun()
    with t_sched:
        rows = reports.load_schedules()
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if rows:
            st.dataframe(pd.DataFrame([{"name": s["name"], "period": reports.PERIODS.get(s["period"], s["period"]),
                                        "days": ", ".join(days[d] for d in s["days"]), "time": s["time"],
                                        "sections": "global template" if not s.get("sections") else f"{len(s['sections'])} own",
                                        "on": s["enabled"], "last run": s.get("last_run", "")}
                                       for s in rows]), hide_index=True, use_container_width=True)  # fmt: skip
        if not may_edit:
            st.info("Your role cannot change schedules.")
        with st.form("sched"):
            st.markdown("**Add or change a schedule** (same name = change it)")
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Name", "Weekly")
            period = c2.selectbox("Period", list(reports.PERIODS), index=3,
                                  format_func=reports.PERIODS.__getitem__)  # fmt: skip
            at_time = c3.text_input("Time (HH:MM)", "07:00")
            wd = st.multiselect("Days", list(range(7)), default=[0], format_func=days.__getitem__)
            own = st.checkbox("Own sections instead of the global template")
            names = {k: v[0] for k, v in reports.SECTIONS.items()}
            secs = st.multiselect("Sections", list(names), default=reports.load_template()["sections"],
                                  format_func=names.__getitem__)  # fmt: skip
            fmts = st.multiselect("Files", list(reports.FORMATS), default=["docx", "html"],
                                  format_func=reports.FORMATS.__getitem__)  # fmt: skip
            folder = st.text_input("Save to folder (empty = results/reports)")
            on = st.checkbox("On", value=True)
            if st.form_submit_button("Save schedule", disabled=not may_edit):
                s = {"name": name, "period": period, "days": wd, "time": at_time.strip(),
                     "enabled": on, "formats": fmts, "folder": folder,
                     "sections": secs if own else None}  # fmt: skip
                try:
                    keep = [r for r in rows if r["name"] != name.strip()]
                    reports.save_schedules([*keep, s], who=_me())
                    st.success("Saved.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        if rows and may_edit:
            drop = st.selectbox("Remove a schedule", [s["name"] for s in rows], key="sched_rm")
            if st.button("Remove schedule"):
                reports.save_schedules([r for r in rows if r["name"] != drop], who=_me())
                st.rerun()
    with t_saved:
        saved = sorted((p for p in reports.OUT.glob("*.*") if p.suffix in (".docx", ".md", ".html")),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:60] if reports.OUT.is_dir() else []  # fmt: skip
        if not saved:
            st.caption("No reports yet.")
        for p in saved:
            c1, c2 = st.columns([4, 1])
            c1.write(f"{p.name} · {datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M}")
            c2.download_button("Download", p.read_bytes(), file_name=p.name, key=f"sv_{p.name}")


def _receiver_up(port: int) -> bool:
    import socket as _socket

    try:
        with _socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def page_agents() -> None:
    """Central log collection: agent keys, API tokens, receiver settings (owner 2026-10-16)."""
    from src.soc import collector

    st.header("Agents & API")
    _howto("Other machines (Windows, macOS, Linux) send their logs here with an agent; "
           "SIEM / SOAR tools read incidents and alerts through the read-only API. Each "
           "agent and tool gets its own key, shown once; only a hash is stored.")  # fmt: skip
    s = sinks.load_settings()
    c = s["collector"]
    up = _receiver_up(int(c["port"]))
    st.badge(f"Receiver running on port {c['port']}" if up else "Receiver not running",
             color="green" if up else "orange")  # fmt: skip
    fp = collector.fingerprint()
    from src.soc import heartbeat

    if bad := heartbeat.clock_problems():
        st.warning("Agent clocks more than 2 minutes off (correlation needs correct times): "
                   + ", ".join(f"{n} {v:+.0f} s" for n, v in bad.items()))  # fmt: skip
    st.caption("Certificate fingerprint (SHA-256) - agents pin this value:")
    st.code(fp, language="text")
    if not up:
        st.caption("Start it: `python -m src.soc.collector serve` (or `python -m src.soc.services "
                   "install` to start it with the computer). Allow the port once as administrator: "
                   "`scripts\\allow_collector_firewall.ps1` (Windows) or `sudo sh install.sh "
                   "--firewall` (Linux / macOS).")  # fmt: skip
    with st.expander("Receiver settings"):
        cols = st.columns(3)
        c["port"] = cols[0].number_input("Port", 1024, 65535, int(c["port"]))
        c["max_mb_per_min"] = cols[1].number_input("Max MB per agent per minute", 1, 10_000,
                                                   int(c.get("max_mb_per_min", 200)))  # fmt: skip
        c["lan_only"] = cols[2].checkbox("Local network only", bool(c.get("lan_only", True)),
                                         help="Refuse machines outside private / loopback addresses")  # fmt: skip
        c["silence_minutes"] = st.number_input("Alert when an agent sends nothing for (minutes)", 5, 10_080,
                                               int(c.get("silence_minutes", 30)))  # fmt: skip
        lv = s.setdefault("live", {"poll_s": 2.0, "push_s": 1.0})
        lv["poll_s"] = st.number_input("Read this computer's logs every (seconds)", 0.5, 300.0,
                                       float(lv.get("poll_s", 2.0)), 0.5,
                                       help="Lower = faster alerts, a little more CPU (about 15 ms per log per pass)")  # fmt: skip
        lv["push_s"] = st.number_input("Push new incidents to the SIEM every (seconds)", 0.5, 300.0,
                                       float(lv.get("push_s", 1.0)), 0.5)  # fmt: skip
        if st.button("Save receiver settings"):
            sinks.save_settings(s)
            _audit("collector_settings", {**{k: c[k] for k in ("port", "max_mb_per_min", "lan_only", "silence_minutes")}, "live": lv})
            st.success("Saved. Restart the receiver and the live watcher to apply.")
    t_agents, t_tokens, t_install = st.tabs(["Agents", "API tokens", "Install an agent"])
    for tab, kind, label, make in ((t_agents, "agents", "agent", collector.new_agent),
                                   (t_tokens, "tokens", "API token", collector.new_token)):  # fmt: skip
        with tab:
            rows = collector.listing(kind)
            if rows:
                st.dataframe(pd.DataFrame([{"name": n, "status": "disabled" if e.get("disabled") else "active",
                                            "created": e.get("created"), "by": e.get("by"),
                                            "last seen": e.get("last_seen") or "never",
                                            "clock off by": (f"{e['skew_s']:+.0f} s" if e.get("skew_s") is not None else ""),
                                            "from": e.get("last_ip") or "",
                                            "records" if kind == "agents" else "requests": e.get("count", 0)}
                                           for n, e in rows.items()]), hide_index=True, use_container_width=True)  # fmt: skip
            with st.form(f"new_{kind}", clear_on_submit=True):
                name = st.text_input(f"New {label} - name (e.g. the machine or tool)")
                if st.form_submit_button(f"Create {label}", type="primary"):
                    try:
                        st.session_state[f"secret_{kind}"] = (name, make(_me(), name.strip()))
                    except ValueError as exc:
                        st.error(str(exc))
            if sec := st.session_state.pop(f"secret_{kind}", None):
                st.warning(f"Copy the {label} for **{sec[0]}** now - it is not shown again.")
                st.code(sec[1], language="text")
            if rows:
                c1, c2, c3 = st.columns([2, 1, 1])
                pick = c1.selectbox(label.capitalize(), list(rows), key=f"pick_{kind}")
                off = bool(rows[pick].get("disabled"))
                if c2.button("Enable" if off else "Disable", key=f"dis_{kind}"):
                    collector.set_disabled(kind, _me(), pick, not off)
                    st.rerun()
                if c3.button("Remove", key=f"rm_{kind}"):
                    collector.remove(kind, _me(), pick)
                    st.rerun()
            if kind == "tokens":
                st.caption("Read-only endpoints (send `Authorization: Bearer <token>`): "
                           "`GET /api/v1/health`, `/api/v1/incidents?since=<unix time>&level=high`, "
                           "`/api/v1/alerts?since=&limit=`, `/api/v1/search?q=failed logins for bob`. "
                           "Pin the certificate fingerprint above in your tool.")  # fmt: skip
    with t_install:
        here = PROJECT_ROOT / "scripts" / "agents"
        st.markdown("1. Create an agent above and copy its key. 2. Copy the agent file to the "
                    "machine. 3. Run the installer there; it asks for the address "
                    f"(`https://{_lan_ip()}:{c['port']}`), the key and the fingerprint.")  # fmt: skip
        for os_name, files, cmd in (
            ("Windows (no Python needed)", ["splitshield_agent.ps1", "install_agent.ps1"],
             "powershell -ExecutionPolicy Bypass -File install_agent.ps1   (as Administrator)"),
            ("Linux / macOS", ["splitshield_agent.py", "install_agent.sh"], "sudo sh install_agent.sh"),
        ):  # fmt: skip
            st.markdown(f"**{os_name}**")
            cols = st.columns(len(files) + 1)
            for col, f in zip(cols, files, strict=False):
                col.download_button(f, (here / f).read_bytes(), file_name=f, key=f"dl_agent_{f}")
            st.code(cmd, language="text")


def _lan_ip() -> str:
    import socket as _socket

    try:
        with _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM) as sk:
            sk.connect(("192.0.2.1", 9))  # no packet is sent; picks the LAN interface
            return str(sk.getsockname()[0])
    except OSError:
        return "THIS-PC-IP"


def page_home() -> None:
    """The first screen: what needs attention now (owner 2026-10-16)."""
    import time as _time

    from src.soc import perf, winlive

    st.header("Home")
    now = _time.time()
    day0 = _time.mktime((*_time.localtime(now)[:3], 0, 0, 0, 0, 0, -1))
    alerts = engine.read_jsonl(engine.ALERTS, last=50_000)
    today = [a for a in alerts if float(a.get("ts") or 0) >= day0]
    incs = [i for i in engine.incidents() if not i.get("allowlisted")]
    by = {
        lv: sum(i["level"] == lv for i in incs)
        for lv in ("critical", "high", "medium", "low")
    }
    from src.soc import osops

    if osops.PLATFORM == "windows":
        live = (winlive.load_state().get("last_poll") or {}).get("logs", {})
    else:
        from src.soc import oslive

        live = oslive.status()
    reading = [
        k for k, v in live.items() if v == "ok" or str(v).endswith("new records")
    ]
    plan = perf.plan()
    bad = [m for m, ok in _health_quick().items() if not ok]
    cards = [
        ("Alerts today", f"{len(today):,}", "", "live", "Live monitor", "monitor_heart"),
        ("Open incidents", f"{len(incs):,}", f"{by['critical']} critical, {by['high']} high",
         "incidents", "Incidents & review", "crisis_alert"),
        ("This PC's logs", f"{len(reading)} of {len(live) or 3} read", "",
         "start", "Fix in Getting started", "rocket_launch"),
        ("Stored events", f"{engine.event_count():,}", "", "ask", "Search or ask", "chat"),
        ("Log analysis speed", f"{plan['predicted_lps']:,}/s", f"setting: {plan['mode']}",
         "performance", "Performance", "speed"),
        ("Health", "All parts OK" if not bad else f"{len(bad)} need attention",
         ", ".join(bad)[:60], "health", "Self-improvement & health", "auto_mode"),
    ]  # fmt: skip
    for row in (cards[:3], cards[3:]):
        for col, (label, value, note, page, link, icon) in zip(
            st.columns(3), row, strict=True
        ):
            with col.container(border=True):
                st.metric(label, value)
                st.caption(note or " ")  # same card height with or without a note
                if page in PAGE_OBJ:  # the user may not have every page
                    st.page_link(PAGE_OBJ[page], label=link, icon=f":material/{icon}:")
    if incs:
        st.subheader("Most severe incidents")
        for inc in incs[:5]:
            who = inc["src_ip"] if not is_unknown(inc["src_ip"]) else unknown_label(inc["src_ip"])
            techs = ", ".join(t["name"] for t in inc["techniques"][:3])
            st.markdown(
                f"**{inc['level'].capitalize()} {inc['severity']}/10** · {md(who)} · {md(techs)}"
            )


def _health_quick() -> dict[str, bool]:
    """Status of the automatic parts without loading models (fast, for Home)."""
    from src.soc import controller

    try:
        return {
            k.replace("_", " "): bool(v.get("ok"))
            for k, v in controller.components().items()
        }
    except Exception:  # noqa: BLE001 - Home must open even if a part cannot be read
        return {}


def page_customize() -> None:
    """The owner's own rules, allow-list and stricter thresholds (owner 2026-10-15)."""
    from src.knowledge.attack_details import technique_info
    from src.soc import allowlist, custom, rulectl

    rulectl.ACTOR[0] = _me()
    st.header("Customise")
    _howto("Add your own detection rules, switch or change any built-in rule, tune the "
           "engine, mark known-good IPs, users, hosts, files or domains, and make a model "
           "stricter or looser. Every change is tested on your stored logs, can be undone "
           "and is written to the audit log.")  # fmt: skip
    events = engine.recent_events()  # rules are tested on the newest 200,000
    t_rules, t_built, t_engine, t_packs, t_props, t_allow, t_sens = st.tabs(
        ["My rules", "Built-in rules", "Engine settings", "Rule packs", "Proposals & history",
         "Allow-list", "Sensitivity"])  # fmt: skip
    with t_built:
        _builtin_rules_tab(events)
    with t_engine:
        _engine_settings_tab()
    with t_packs:
        _rule_packs_tab(events)
    with t_props:
        _proposals_tab()

    with t_rules:
        how = st.radio(
            "Write the rule as", ["Simple form", "Sigma YAML"], horizontal=True
        )
        name = st.text_input("Rule name", key="cr_name")
        fams = ["any", "network", "endpoint", "web", "auth", "ids", "cloud", "mail",
                "phishing", "malware", "insider", "intel"]  # fmt: skip
        doc: dict = {"name": name}
        if how == "Simple form":
            c1, c2 = st.columns(2)
            doc["family"] = c1.selectbox("Log family", fams)
            doc["field"] = c2.selectbox("Look in", list(custom.FIELDS),
                                        help="'any' = the whole event text")  # fmt: skip
            c3, c4 = st.columns([1, 3])
            doc["op"] = c3.selectbox("Match", list(custom.OPS))
            doc["value"] = c4.text_input("Value (text, or a regular expression)")
            c5, c6 = st.columns(2)
            doc["count"] = c5.number_input(
                "Fire after N matches (1 = every match)", 1, 10000, 1
            )
            doc["minutes"] = c6.number_input("…within M minutes, per IP / user / host", 0, 10080,
                                             0 if doc["count"] == 1 else 10)  # fmt: skip
        else:
            doc["kind"] = "sigma"
            doc["sigma"] = st.text_area("Sigma rule (YAML)", height=220,
                                        placeholder="title: ...\ndetection:\n  sel:\n"
                                                    "    detail|contains: 'mimikatz'\n"
                                                    "  condition: sel")  # fmt: skip
            st.caption("Matched against the stored event fields: source, kind, detail, "
                       "user, host, src_ip, dst_ip, dst_port, file, sha256, url (plus "
                       "keywords in any of them). Techniques come from its attack.tXXXX "
                       "tags unless you pick them below.")  # fmt: skip
        tids = st.text_input(
            "ATT&CK technique id(s), comma-separated (e.g. T1110, T1059.001)"
        )
        doc["techniques"] = [t.strip().upper() for t in tids.split(",") if t.strip()]
        if doc["techniques"]:
            st.caption(
                " · ".join(f"{t}: {technique_info(t)[0]}" for t in doc["techniques"])
            )
        doc["severity"] = st.slider("Severity (1 low - 10 critical)", 1, 10, 5)
        accept = st.checkbox(
            "I understand this rule fires often (save even if > 5 % of events)"
        )
        c7, c8 = st.columns(2)
        if c7.button("Test on my stored logs"):
            try:
                r = custom.test_rule(doc, events)
                st.info(f"Would have fired on **{r['fired']}** of {r['events']:,} stored "
                        f"events ({r['rate']:.2%})." + (" ⚠️ That is noisy." if r["too_noisy"] else ""))  # fmt: skip
                if r["examples"]:
                    st.table(r["examples"])
            except Exception as exc:  # noqa: BLE001 - show the problem, never crash
                st.error(f"Rule not valid: {exc}")
        if c8.button("Save rule"):
            try:
                r = custom.save_rule(doc, events, accept_noisy=accept)
                _audit("rule_saved", {"rule": name, "fired": r["fired"]})
                st.success(f"Saved '{name}' (tested: fired on {r['fired']} of {r['events']:,} events). "
                           "It runs on everything ingested from now on.")  # fmt: skip
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
        st.subheader("Saved rules")
        _, errors = custom.rules()
        for d in custom.load_docs():
            rn = str(d.get("name"))
            with st.container(border=True):
                t = d.get("tested") or {}
                st.markdown(f"{'🟢' if d.get('enabled', True) else '⚪'} **{rn}** · "
                            f"{', '.join(d.get('techniques') or []) or '(Sigma tags)'} · "
                            f"tested: {t.get('fired', '?')} of {t.get('events', '?')} events")  # fmt: skip
                if rn in errors:
                    st.error(f"Not running: {errors[rn]}")
                b1, b2 = st.columns(2)
                on = d.get("enabled", True)
                if b1.button("Disable" if on else "Enable", key=f"cr_t_{rn}"):
                    custom.set_enabled(rn, not on)
                    _audit("rule_enabled" if not on else "rule_disabled", {"rule": rn})
                    st.rerun()
                if b2.button("🗑 Delete", key=f"cr_d_{rn}"):
                    custom.delete_rule(rn)
                    _audit("rule_deleted", {"rule": rn})
                    st.rerun()

    with t_allow:
        st.caption("Allow-listed activity is still stored, searchable and shown here, but it "
                   "is not forwarded to your SIEM and not counted in reports. Critical "
                   "incidents are still forwarded, marked 'allow-listed entity involved'.")  # fmt: skip
        c1, c2 = st.columns([1, 3])
        kind = c1.selectbox("Type", list(allowlist.TYPES))
        value = c2.text_input(
            "Value (IP, user name, host name, SHA256 or file path, domain)"
        )
        reason = st.text_input("Why is it known-good? (required)")
        days = st.number_input(
            "Expires after (days)", 1, allowlist.MAX_DAYS, allowlist.DEFAULT_DAYS
        )
        if st.button("Add to allow-list"):
            try:
                allowlist.add(kind, value, reason, int(days))
                _audit("allowlist_added", {"type": kind, "value": value, "reason": reason, "days": int(days)})
                st.success("Added.")
            except ValueError as exc:
                st.error(str(exc))
        for e in allowlist.entries():
            with st.container(border=True):
                st.markdown(f"{'🟢' if e['active'] else '⌛ expired'} **{e['type']}** `{e['value']}` · "
                            f"{e['reason']} · added {e['added']}, until {e['expires']}")  # fmt: skip
                if st.button("Remove", key=f"al_{e['id']}"):
                    allowlist.remove(e["id"])
                    _audit("allowlist_removed", {"type": e["type"], "value": e["value"]})
                    st.rerun()
        muted = [
            a
            for a in engine.read_jsonl(engine.ALERTS, last=2000)
            if a.get("allowlisted")
        ]
        if muted:
            st.subheader(f"Muted alerts ({len(muted)} recent)")
            for a in muted[-50:][::-1]:
                st.caption(
                    f"⚪ {_t(a['ts'])} · {a.get('title', a.get('kind'))} · {a['allowlisted']}"
                )

    with t_sens:
        st.caption("Make a model stricter (fewer, surer alerts) or looser (more alerts, more "
                   "false alarms). The validated value was chosen on held-out data; a looser "
                   "one shows what it would add on your stored logs before you save it.")  # fmt: skip
        for model, label in custom.LIVE_MODELS.items():
            base = custom.validated(model)
            if base is None:
                st.caption(f"{label}: model not available.")
                continue
            cur = custom.threshold(model, base)
            v = st.slider(label, float(custom.LOOSEST), 0.999, float(round(cur, 4)), 0.001,
                          key=f"sens_{model}", help=f"validated: {base:.3f}")  # fmt: skip
            if v < base - 5e-4:
                d = custom.extra_alerts(model, v, events)
                st.warning(f"⚠️ Looser than the validated {base:.3f}: expect more false alarms. On your "
                           f"stored logs this would add {d['added']} alert(s) now (more as new logs arrive).")  # fmt: skip
            if abs(v - cur) > 1e-9 and st.button(
                f"Save {model}", key=f"sens_s_{model}"
            ):
                try:
                    custom.set_threshold(model, v, base)
                    _audit("sensitivity", {"model": model, "threshold": v})
                    st.success(f"{model}: now {v:.3f} (validated {base:.3f}).")
                except ValueError as exc:
                    st.error(str(exc))
            if abs(cur - base) > 5e-4 and st.button(
                f"Back to validated ({base:.3f})", key=f"sens_r_{model}"
            ):
                custom.set_threshold(model, None, base)
                _audit("sensitivity", {"model": model, "threshold": "validated"})
                st.rerun()


@st.cache_data(ttl=600, show_spinner="Loading the rule catalogue...")
def _rule_catalog() -> list[dict]:
    from src.soc import rulectl

    return rulectl.catalog()


def _builtin_rules_tab(events: list[dict]) -> None:
    """Search any built-in rule; switch it off, change severity / technique, add exceptions."""
    from src.soc import rulectl

    cat = _rule_catalog()
    fired = rulectl.fired(events)
    ov = rulectl.overrides()
    st.caption(f"{len(cat):,} built-in rules (Sigma, OWASP CRS, LOLBAS / GTFOBins) plus our own rules "
               "as they fire. Changes apply to new logs at once; the button re-checks stored logs too.")  # fmt: skip
    if st.button("Re-check stored logs with my rule changes", key="br_reapply"):
        with st.spinner("Re-checking the stored events..."):
            rr = rulectl.reapply_store()
        st.success(f"{rr['changed']:,} of {rr['events']:,} stored events changed; incidents and search are rebuilt.")
    c1, c2, c3 = st.columns([3, 2, 2])
    q = c1.text_input("Search (title, id or technique)", key="br_q").strip().lower()
    kinds = sorted({r["kind"] for r in cat})
    kind = c2.selectbox("Type", ["any", *kinds, "own (seen firing)"], key="br_kind")
    show = c3.selectbox("Show", ["all", "fired on my logs", "changed by a user"], key="br_show")
    rows = list(cat) + [{"id": rid, "kind": "own", "title": rid.split(":", 1)[1], "level": "-", "techniques": [],
                         "logsource": "-"} for rid in fired if rid.startswith("own:")]  # fmt: skip
    if kind == "own (seen firing)":
        rows = [r for r in rows if r["kind"] == "own"]
    elif kind != "any":
        rows = [r for r in rows if r["kind"] == kind]
    if q:
        rows = [r for r in rows if q in r["title"].lower() or q in r["id"].lower()
                or any(q in t.lower() for t in r["techniques"])]  # fmt: skip
    if show == "fired on my logs":
        rows = [r for r in rows if fired.get(r["id"])]
    elif show == "changed by a user":
        rows = [r for r in rows if r["id"] in ov]
    st.caption(f"{len(rows):,} match - first 50 shown")
    for r in rows[:50]:
        o = ov.get(r["id"], {})
        state = "⛔ off" if o.get("enabled") is False else ("✏️ changed" if o else "🟢 on")
        with st.expander(f"{state} · {md(r['title'])[:90]} · {r['kind']} {r['level']} · "
                         f"{', '.join(r['techniques'][:3]) or '-'} · fired {fired.get(r['id'], 0)}x"):  # fmt: skip
            st.caption(f"id `{r['id']}` · logs: {r['logsource']}")
            en = st.toggle("Enabled", value=o.get("enabled", True), key=f"br_en_{r['id']}")
            sev = st.number_input("Severity override (0 = keep the rule's own)", 0, 10, int(o.get("severity") or 0),
                                  key=f"br_sev_{r['id']}")  # fmt: skip
            techs = st.text_input("ATT&CK technique(s) instead, comma separated (empty = keep)",
                                  ", ".join(o.get("techniques") or []), key=f"br_t_{r['id']}")  # fmt: skip
            exc_txt = st.text_area("Exceptions, one per line: field op value (e.g. host equals backup01)",
                                   "\n".join(f"{e['field']} {e['op']} {e['value']}" for e in o.get("exceptions") or []),
                                   key=f"br_x_{r['id']}")  # fmt: skip
            reason = st.text_input("Why? (kept in the audit log)", key=f"br_r_{r['id']}")
            b1, b2 = st.columns(2)
            if b1.button("Save", key=f"br_s_{r['id']}"):
                try:
                    exc = []
                    for ln in exc_txt.splitlines():
                        parts = ln.strip().split(" ", 2)
                        if len(parts) == 3:
                            exc.append({"field": parts[0], "op": parts[1], "value": parts[2]})
                    change: dict = {"enabled": en, "severity": sev or None,
                                    "techniques": [t.strip() for t in techs.split(",") if t.strip()],
                                    "exceptions": exc}  # fmt: skip
                    rulectl.set_override(r["id"], change, reason)
                    st.success("Saved - applies to new logs.")
                except ValueError as exc_:
                    st.error(str(exc_))
            if o and b2.button("Back to the built-in rule", key=f"br_u_{r['id']}"):
                rulectl.set_override(r["id"], None, reason or "undo")
                st.rerun()


def _engine_settings_tab() -> None:
    from src.soc import rulectl

    st.caption("How the engine correlates and scores. The validated value (chosen on held-out data) "
               "is shown next to each; a change applies at once and can be undone.")  # fmt: skip
    cur, val = rulectl.engine_settings(), rulectl.validated_engine()
    for key, (_d, lo, hi, what) in rulectl.ENGINE.items():
        c1, c2, c3 = st.columns([4, 2, 2])
        step = 1.0 if hi - lo <= 10 else 60.0
        v = c1.number_input(f"{key.replace('_', ' ')} - {what}", float(lo), float(hi), float(cur[key]), step,
                            key=f"es_{key}")  # fmt: skip
        c2.caption(f"validated: {val[key]:g}")
        if abs(v - cur[key]) > 1e-9 and c3.button("Save", key=f"es_s_{key}"):
            try:
                rulectl.set_engine(key, v)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if abs(cur[key] - val[key]) > 1e-9 and c3.button("Validated", key=f"es_u_{key}"):
            rulectl.set_engine(key, None, "back to validated")
            st.rerun()


def _rule_packs_tab(events: list[dict]) -> None:
    import json as _json

    from src.soc import rulectl

    st.download_button("⬇️ Export my rules, rule changes and engine settings",
                       _json.dumps(rulectl.export_pack(), indent=1, default=str),
                       file_name="splitshield_rule_pack.json", mime="application/json")  # fmt: skip
    up = st.file_uploader("Import a rule pack (.json)", type="json", key="pack_up")
    if up is not None:
        try:
            pack = _json.loads(up.getvalue().decode("utf-8"))
            noisy = st.checkbox("Accept rules that fire on more than 5 % of my logs", key="pack_noisy")
            rep = rulectl.import_pack(pack, events, accept_noisy=noisy)
            st.json({k: v for k, v in rep.items() if k != "ok"}, expanded=False)
            if not rep["ok"]:
                st.error("The pack has problems (see above); nothing was changed.")
            elif st.button("Switch the pack on", type="primary"):
                rulectl.import_pack(pack, events, apply_now=True, accept_noisy=noisy)
                st.success("Pack switched on.")
        except (ValueError, KeyError) as exc:
            st.error(f"Not a valid pack: {exc}")


def _proposals_tab() -> None:
    from src.soc import rulectl

    st.caption("Suggested from your analysts' reviews (Real / False alarm). Nothing changes until "
               "someone accepts; accepted changes appear under Built-in rules and can be undone.")  # fmt: skip
    from src.soc import sitecal

    cs = sitecal.status()
    st.caption(f"Flow calibration for this site: {cs['seen']:,} flows over {cs.get('days', 0):.1f} days learned "
               f"(a threshold is proposed after {sitecal.MIN_FLOWS:,} flows and {sitecal.MIN_DAYS:g} days).")  # fmt: skip
    if st.button("Look for new proposals"):
        rulectl.propose()
        sitecal.propose()
    props = [p for p in rulectl._read(rulectl.PROPOSALS, []) if p.get("status") == "open"]
    if not props:
        st.info(f"No open proposals (a rule needs {rulectl.MIN_REVIEWS}+ reviewed alerts).")
    for p in props:
        with st.container(border=True):
            st.markdown(f"**{md(p['rule'])}** - {md(p['why'])}")
            st.caption(f"change: `{p['change']}` · {p['false_alarms']} false alarms, {p['real']} real")
            a, b = st.columns(2)
            if a.button("Accept", key=f"pa_{p['id']}"):
                rulectl.decide(p["id"], True)
                st.rerun()
            if b.button("Reject", key=f"pr_{p['id']}"):
                rulectl.decide(p["id"], False)
                st.rerun()
    hist = rulectl.history(50)
    if hist:
        st.subheader("Recent rule changes")
        for h in hist[::-1]:
            st.caption(f"{h.get('when')} · {md(h.get('who'))} · {h.get('action')} · "
                       f"{md(h.get('rule') or h.get('setting') or h.get('proposal') or '')}")  # fmt: skip


# (section, title, page function, Material icon, url path) - owner 2026-10-16 redesign:
# grouped navigation, plain titles, line icons instead of emoji
NAV = [
    ("Overview", "Home", page_home, "home", "home"),
    ("Overview", "Getting started", page_start, "rocket_launch", "start"),
    ("Overview", "Ask SplitShield", page_ask, "chat", "ask"),
    ("Overview", "Reports", page_reports, "description", "reports"),
    ("Monitor", "Live monitor", page_live, "monitor_heart", "live"),
    ("Monitor", "Incidents & review", page_incidents, "crisis_alert", "incidents"),
    ("Investigate", "Analyze logs", page_analyze, "folder_open", "analyze"),
    ("Investigate", "Explain attacks", page_explain, "menu_book", "explain"),
    ("Investigate", "Check URL / file / sensors", page_check, "biotech", "check"),
    (
        "Models & data",
        "Test any dataset",
        page_test_dataset,
        "fact_check",
        "test-dataset",
    ),
    ("Models & data", "New log formats", page_new_formats, "schema", "log-formats"),
    ("Models & data", "Add dataset", page_dataset, "add_circle", "add-dataset"),
    ("Models & data", "Learning (reviews)", page_learning, "school", "learning"),
    ("Models & data", "Drift & autopilot", page_drift, "trending_up", "drift"),
    (
        "Models & data",
        "Self-improvement & health",
        page_evolution,
        "auto_mode",
        "health",
    ),
    ("Models & data", "Updates & threat intel", page_updates, "update", "updates"),
    ("Settings", "Connect SOC", page_soc, "cable", "connect"),
    ("Settings", "Customise", page_customize, "tune", "customise"),
    ("Settings", "Performance", page_performance, "speed", "performance"),
    ("Settings", "Places & backup", page_places, "location_on", "places"),
    ("Models & data", "Evolve or freeze", page_evolution_mode, "ac_unit", "evolution"),
    ("Settings", "Agents & API", page_agents, "lan", "agents"),
    ("Settings", "Users & roles", page_users, "group", "users"),
    ("Settings", "Audit log", page_audit, "policy", "audit"),
    ("Research", "Project results", page_results, "insights", "results"),
]
# permission needed to see each page (src/soc/users.py; owner 2026-10-16)
PAGE_PERM = {
    "home": "view", "reports": "view", "start": "view", "ask": "view", "live": "view", "incidents": "view",
    "explain": "view", "results": "view", "analyze": "investigate", "check": "investigate",
    "test-dataset": "investigate", "customise": "rules", "log-formats": "rules",
    "add-dataset": "models", "learning": "models", "drift": "models", "health": "models",
    "evolution": "models", "connect": "settings", "performance": "settings",
    "places": "settings", "updates": "settings", "agents": "collection", "users": "users", "audit": "users",
}  # fmt: skip
# kept for scripts and tests that list the pages by title
PAGES = {title: fn for _s, title, fn, _i, _u in NAV}


def _login() -> None:
    """Sign-in with username and password (src/soc/users.py). A new install first asks
    for the admin's name and password; wrong attempts back off exponentially."""
    import time as _time

    from src.soc import auth, users

    if users.needs_first_admin():
        st.title("SplitShield-XAI")
        st.info("First start: create the admin account. Only the admin can add users, "
                "assign roles and change usernames, passwords and permissions.")  # fmt: skip
        with st.form("first_admin"):
            name = st.text_input("Admin username", value="admin")
            pw = st.text_input(f"Password (at least {auth.MIN_LENGTH} characters)", type="password")
            again = st.text_input("Repeat the password", type="password")
            ok = st.form_submit_button("Create admin", type="primary")
        if ok:
            if pw != again:
                st.error("The two entries differ.")
            else:
                try:
                    users.create_first_admin(name.strip(), pw)
                    st.session_state["user"] = name.strip()
                    st.rerun()
                except (ValueError, PermissionError) as exc:
                    st.error(str(exc))
        st.stop()
    me = st.session_state.get("user")
    if me and users.permissions_of(me):  # still exists, enabled and has a role
        with st.sidebar:
            st.caption(f":material/person: {me} ({users.load()['users'][me]['role']})")
            if st.button("Sign out", key="sign_out"):
                st.session_state.pop("user", None)
                st.rerun()
        return
    st.session_state.pop("user", None)
    st.title("SplitShield-XAI")
    with st.form("login"):
        name = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in")
    if ok:
        if users.authenticate(name.strip(), pw):
            st.session_state["user"] = name.strip()
            st.session_state["fails"] = 0
            st.rerun()
        fails = st.session_state.get("fails", 0) + 1
        st.session_state["fails"] = fails
        _time.sleep(min(2**fails, 30))  # slows down password guessing
        st.error("Wrong username or password.")
    st.stop()


_login()
st.logo(str(Path(__file__).with_name("assets") / "logo.svg"), size="large")
# ?page=incidents (or the page's own address, /incidents) opens that page
_want = str(st.query_params.get("page", "")).lower()
_default = next(
    (u for _s, t, _f, _i, u in NAV if _want and (_want in t.lower() or _want == u)),
    "home",
)
_sections: dict[str, list] = {}
PAGE_OBJ: dict[str, Any] = {}
_allowed = [n for n in NAV if _can(PAGE_PERM[n[4]])]
if not any(n[4] == _default for n in _allowed):
    _default = _allowed[0][4] if _allowed else "home"
for _sec, _title, _fn, _icon, _path in _allowed:
    PAGE_OBJ[_path] = st.Page(_fn, title=_title, icon=f":material/{_icon}:", url_path=_path,
                              default=_path == _default)  # fmt: skip
    _sections.setdefault(_sec, []).append(PAGE_OBJ[_path])
st.navigation(_sections, expanded=True).run()
