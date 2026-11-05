"""The final report as Word (owner 2026-10-22): everything from A to Z, with diagrams.

    python -m scripts.final_word_report [--rerun D:/SplitShield-rerun --rerun2 D:/SplitShield-rerun2]

Built from the project's own result files (missing ones are marked "not run yet"), so
it can be rebuilt at any time. Diagrams are drawn here (matplotlib) and embedded.
Output: results/SplitShield-XAI_final_report.docx
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from utils.paths import PROJECT_ROOT, RESULTS_DIR

EXTRA = RESULTS_DIR / "extra"
OUT = RESULTS_DIR / "SplitShield-XAI_final_report.docx"
FIG = Path(tempfile.mkdtemp(prefix="splitshield-report-"))
FIG_KEEP = RESULTS_DIR / "figures" / "report"  # copies of the diagrams for reuse (slides, paper)


def _load(p: Path) -> Any:
    try:
        return json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return None


# --- diagrams --------------------------------------------------------------------------


def _boxes(name: str, boxes: dict[str, tuple[float, float, str, str]], edges: list[tuple[str, str, str]],
           size: tuple[float, float] = (11, 6.5), title: str = "") -> Path:  # fmt: skip
    """boxes: key -> (x, y, label, colour); edges: (from, to, label). Coordinates 0..1."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(-0.03, 1.07)
    ax.set_ylim(-0.03, 1.0)
    ax.axis("off")
    w, h = 0.17, 0.075
    for x, y, label, col in boxes.values():
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008",
                                    fc=col, ec="#33415c", lw=1, zorder=3))  # fmt: skip
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5, zorder=4)
    for a, b, label in edges:
        xa, ya, *_ = boxes[a]
        xb, yb, *_ = boxes[b]
        if abs(xb - xa) > 0.05:  # left -> right (or back): leave and enter at the sides
            sx = xa + (w / 2 + 0.01) * (1 if xb > xa else -1)
            ex = xb - (w / 2 + 0.01) * (1 if xb > xa else -1)
            start, end = (sx, ya), (ex, yb)
        else:  # same column: top / bottom
            sy = ya - (h / 2 + 0.012) * (1 if yb < ya else -1)
            ey = yb + (h / 2 + 0.012) * (1 if yb < ya else -1)
            start, end = (xa, sy), (xb, ey)
        ax.annotate("", xy=end, xytext=start, zorder=2,
                    arrowprops={"arrowstyle": "->", "color": "#33415c", "lw": 0.8,
                                "connectionstyle": "arc3,rad=0.0"})  # fmt: skip
        if label:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, label, fontsize=6.5, color="#1f4e8c",
                    ha="center", va="center", zorder=5, bbox={"fc": "white", "ec": "none", "pad": 0.5})  # fmt: skip
    if title:
        ax.set_title(title, fontsize=11)
    p = FIG / f"{name}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    FIG_KEEP.mkdir(parents=True, exist_ok=True)
    (FIG_KEEP / p.name).write_bytes(p.read_bytes())
    return p


IN, MODEL, CORE, OUTC = "#dbe9ff", "#fff1c9", "#dff5e1", "#f6dde6"


def diagram_architecture() -> Path:
    b = {
        "flows": (0.09, 0.93, "Network flows\nCICFlowMeter / NetFlow", IN),
        "zeek": (0.09, 0.81, "Zeek conn.log", IN),
        "win": (0.09, 0.69, "Windows / Sysmon /\nPowerShell / .evtx", IN),
        "lin": (0.09, 0.57, "Linux / web / mail /\ncloud / firewall logs", IN),
        "agents": (0.09, 0.45, "Agents (Win / macOS /\nLinux) over HTTPS", IN),
        "files": (0.09, 0.33, "URLs, files, pages,\ne-mail attachments", IN),
        "ics": (0.09, 0.21, "Industrial sensors", IN),
        "users": (0.09, 0.09, "User activity", IN),
        "parse": (0.3, 0.63, "Parsers\n(format auto-detected)", CORE),
        "m_flow": (0.52, 0.95, "Flow stack v4 +\nDDoS specialist", MODEL),
        "m_nf": (0.52, 0.845, "NetFlow model", MODEL),
        "m_zeek": (0.52, 0.74, "Zeek model", MODEL),
        "rules": (0.52, 0.635, "Rules: 3,161 Sigma,\nOWASP CRS, LOLBAS", MODEL),
        "m_phish": (0.52, 0.53, "Phishing: URL model\n+ page model", MODEL),
        "m_mal": (0.52, 0.425, "Malware model", MODEL),
        "m_ics": (0.52, 0.32, "ICS model", MODEL),
        "m_ins": (0.52, 0.215, "Insider model", MODEL),
        "upd": (0.74, 0.93, "Update agent: new Sigma,\nATT&CK, OWASP, intel", CORE),
        "store": (0.74, 0.6, "Event store +\nshared risk board", CORE),
        "corr": (0.74, 0.42, "Correlation: entity,\n15 min + board boost", CORE),
        "inc": (0.74, 0.24, "Incidents (ATT&CK\npath, severity, cases)", CORE),
        "learn": (0.95, 0.06, "Gated retraining\n(kept only if not worse)", CORE),
        "siem": (0.95, 0.6, "SIEM (CEF / LEEF /\nECS / OCSF, TLS)", OUTC),
        "dash": (0.95, 0.24, "Dashboard +\nShieldLM assistant", OUTC),
        "api": (0.95, 0.42, "REST API / reports /\ncompliance", OUTC),
    }
    e = [("flows", "m_flow", ""), ("flows", "m_nf", ""), ("zeek", "m_zeek", ""), ("win", "parse", ""),
         ("lin", "parse", ""), ("agents", "parse", ""), ("parse", "rules", ""), ("files", "m_phish", ""),
         ("files", "m_mal", ""), ("ics", "m_ics", ""), ("users", "m_ins", ""),
         ("m_flow", "store", ""), ("m_nf", "store", ""), ("m_zeek", "store", ""), ("rules", "store", ""),
         ("m_phish", "store", ""), ("m_mal", "store", ""), ("m_ics", "store", ""), ("m_ins", "store", ""),
         ("upd", "rules", ""), ("store", "corr", ""), ("corr", "inc", ""), ("inc", "siem", ""),
         ("inc", "dash", ""), ("inc", "api", ""), ("dash", "learn", "analyst reviews")]  # fmt: skip
    return _boxes("architecture", b, e, size=(11, 7), title="SplitShield-XAI: from sources to the SIEM")


def diagram_flow_inside() -> Path:
    b = {
        "x": (0.06, 0.55, "Flow features\n(CICFlowMeter)", IN),
        "scale": (0.28, 0.55, "Robust scaling", CORE),
        "mlp": (0.5, 0.92, "MLP (4 x 128)", MODEL),
        "lgbm": (0.5, 0.7, "LightGBM", MODEL),
        "spec": (0.5, 0.45, "DDoS specialist\n(LightGBM)", MODEL),
        "gen": (0.5, 0.2, "Generalist LightGBM\n(4 datasets)", MODEL),
        "ens": (0.72, 0.8, "Ensemble\n(class probabilities)", CORE),
        "casc": (0.72, 0.55, "Cascade: 'benign' but\nspecialist >= 0.52: DDoS", CORE),
        "stack": (0.72, 0.25, "Stack LightGBM\n(adversarially trained)", MODEL),
        "energy": (0.94, 0.95, "Energy check: 'unlike\ntraining data' flag", OUTC),
        "shap": (0.94, 0.75, "SHAP top features:\n'why flagged'", OUTC),
        "thr": (0.94, 0.25, "Attack score >= 0.85:\nconfirmed alert", OUTC),
    }
    e = [("x", "scale", ""), ("scale", "mlp", ""), ("scale", "lgbm", ""), ("scale", "spec", ""),
         ("scale", "gen", ""), ("mlp", "ens", ""), ("lgbm", "ens", ""), ("ens", "casc", ""),
         ("spec", "casc", ""), ("casc", "stack", ""), ("gen", "stack", ""), ("stack", "thr", ""),
         ("mlp", "energy", ""), ("ens", "shap", "")]  # fmt: skip
    return _boxes("inside_flow", b, e, size=(11, 5), title="Inside the flow detector (stack v4)")


def diagram_phishing_inside() -> Path:
    b = {
        "url": (0.06, 0.85, "URL in a log\nor an e-mail", IN),
        "hf": (0.28, 0.85, "Host name: brand\ndistance, TLD, n-grams", CORE),
        "um": (0.5, 0.85, "URL LightGBM (v5)", MODEL),
        "v": (0.72, 0.85, "phishing / suspicious /\nlegitimate", CORE),
        "intel": (0.28, 0.55, "Threat-intel feeds\n(daily, cached)", MODEL),
        "top": (0.72, 0.55, "Top-10k sites never\n'suspicious'", CORE),
        "out": (0.94, 0.55, "Finding + reasons\nto the risk board", OUTC),
        "page": (0.06, 0.15, "Uploaded page or\ne-mail attachment", IN),
        "pf": (0.28, 0.15, "Page: forms, links,\nbrands, scripts", CORE),
        "pm": (0.5, 0.15, "Page LightGBM (v2)", MODEL),
    }
    e = [("url", "hf", ""), ("hf", "um", ""), ("um", "v", ""), ("v", "top", ""), ("top", "out", ""),
         ("intel", "top", "known-bad URLs"), ("page", "pf", ""), ("pf", "pm", ""), ("pm", "out", "")]  # fmt: skip
    return _boxes("inside_phishing", b, e, size=(11, 4.5), title="Inside the phishing detection")


def diagram_assistant_inside() -> Path:
    b = {
        "q": (0.06, 0.55, "Analyst question", IN),
        "intent": (0.28, 0.55, "Question model\n(intent + entities)", MODEL),
        "off": (0.28, 0.15, "Off-topic: declined", OUTC),
        "facts": (0.5, 0.85, "Facts from the\nevent store", CORE),
        "kb": (0.5, 0.3, "ATT&CK knowledge base\n(706 techniques)", CORE),
        "lm": (0.72, 0.55, "ShieldLM writes\nthe wording", MODEL),
        "chk": (0.94, 0.55, "Faithfulness check:\nevery fact in the data?", CORE),
        "ans": (0.94, 0.9, "Answer (ShieldLM\nwording)", OUTC),
        "tpl": (0.94, 0.15, "Answer (fixed\ntemplate wording)", OUTC),
    }
    e = [("q", "intent", ""), ("intent", "off", ""), ("intent", "facts", ""), ("intent", "kb", ""),
         ("facts", "lm", ""), ("kb", "lm", ""), ("lm", "chk", ""), ("chk", "ans", "pass"),
         ("chk", "tpl", "fail")]  # fmt: skip
    return _boxes("inside_assistant", b, e, size=(11, 4.5), title="Inside the assistant")


def diagram_paths() -> Path:
    b = {
        "live": (0.12, 0.8, "This computer's logs\n(read every 2 s)", IN),
        "agent": (0.12, 0.6, "Agent logs\n(sent every 2 s)", IN),
        "flow": (0.12, 0.4, "Live flows\n(batch <= 1 s)", IN),
        "upload": (0.12, 0.2, "Dataset / upload\n(any size)", IN),
        "parse": (0.38, 0.6, "Parse + rules +\nmodels (ms)", CORE),
        "store": (0.6, 0.6, "Event store +\nindex", CORE),
        "corr": (0.6, 0.35, "Incidents\n(pushed every 1 s)", CORE),
        "queue": (0.83, 0.6, "Delivery queue\n(retries, dead letter)", CORE),
        "siem": (0.83, 0.35, "SIEM", OUTC),
    }
    e = [("live", "parse", ""), ("agent", "parse", "HTTPS"), ("flow", "parse", ""), ("upload", "parse", "~140k logs/s"),
         ("parse", "store", ""), ("store", "corr", ""), ("parse", "queue", "alert"), ("corr", "queue", "incident"),
         ("queue", "siem", "")]  # fmt: skip
    return _boxes("paths", b, e, size=(11, 5), title="Real-time paths and the dataset path")


def diagram_correlation() -> Path:
    b = {
        "e1": (0.12, 0.85, "Windows: 40 failed\nlogons (T1110)", IN),
        "e2": (0.12, 0.65, "Windows: successful\nlogon (T1078)", IN),
        "e3": (0.12, 0.45, "Sysmon: powershell -enc\n(T1059.001)", IN),
        "e4": (0.12, 0.25, "NetFlow / IDS: scan\nfrom the same IP (T1046)", IN),
        "e5": (0.12, 0.07, "Phishing model: URL\nfrom that host", IN),
        "ent": (0.42, 0.55, "Same entity\n(IP / user / host)\nwithin 15 min", CORE),
        "board": (0.42, 0.15, "Risk board: other model\nfamilies, last 24 h", CORE),
        "inc": (0.72, 0.55, "ONE incident\nT1046 > T1110 >\nT1078 > T1059", CORE),
        "boost": (0.72, 0.25, "Severity + up to 1\n(never lowered)", CORE),
        "out": (0.93, 0.4, "SIEM + dashboard\n+ explanation", OUTC),
    }
    e = [("e1", "ent", ""), ("e2", "ent", ""), ("e3", "ent", ""), ("e4", "ent", ""), ("e5", "board", ""),
         ("ent", "inc", ""), ("board", "boost", ""), ("boost", "inc", ""), ("inc", "out", "")]  # fmt: skip
    return _boxes("correlation", b, e, size=(11, 5.5), title="How different logs become one incident")


def chart_latency(lat: dict[str, Any]) -> Path | None:
    if not lat:
        return None
    import matplotlib.pyplot as plt

    rows = [("Log line -> SIEM", lat.get("log_line_to_siem", {}), 10), ("Agent -> SIEM", lat.get("agent_to_siem", {}), 10),
            ("Flow batch -> alert", (lat.get("flow_to_alert") or {}).get("batch_of_256", {}), 1),
            ("Incident -> SIEM", lat.get("incidents_build_and_push", {}), 5)]  # fmt: skip
    rows = [r for r in rows if r[1].get("median_ms") is not None]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8, 3.2))
    names = [r[0] for r in rows]
    proc = [r[1]["p95_ms"] / 1000 for r in rows]
    wait = [r[2] for r in rows]
    ax.barh(names, wait, color="#c9d6ea", label="waiting by design (polling)")
    ax.barh(names, proc, left=wait, color="#1f4e8c", label="measured processing (95th pct)")
    ax.set_xlabel("seconds until the SIEM has the alert (worst case)")
    ax.legend(fontsize=8)
    p = FIG / "latency.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


# --- document helpers ------------------------------------------------------------------


class Doc:
    def __init__(self) -> None:
        from docx import Document
        from docx.shared import Pt

        self.d = Document()
        st = self.d.styles["Normal"]
        st.font.name = "Calibri"
        st.font.size = Pt(10.5)

    def h(self, text: str, level: int = 1) -> None:
        self.d.add_heading(text, level=level)

    def p(self, text: str) -> None:
        par = self.d.add_paragraph()
        pos = 0
        for m in re.finditer(r"\*\*(.+?)\*\*", text):
            par.add_run(text[pos : m.start()])
            par.add_run(m.group(1)).bold = True
            pos = m.end()
        par.add_run(text[pos:])

    def bullets(self, items: list[str]) -> None:
        for it in items:
            par = self.d.add_paragraph(style="List Bullet")
            pos = 0
            for m in re.finditer(r"\*\*(.+?)\*\*", it):
                par.add_run(it[pos : m.start()])
                par.add_run(m.group(1)).bold = True
                pos = m.end()
            par.add_run(it[pos:])

    def table(self, head: list[str], rows: list[list[Any]]) -> None:
        t = self.d.add_table(rows=1 + len(rows), cols=len(head))
        t.style = "Light Grid Accent 1"
        for i, hd in enumerate(head):
            t.cell(0, i).text = str(hd)
        for r, row in enumerate(rows, 1):
            for c, v in enumerate(row[: len(head)]):
                t.cell(r, c).text = "" if v is None else str(v)
        self.d.add_paragraph()

    def image(self, path: Path | None, width_in: float = 6.5, caption: str = "") -> None:
        if path is None or not Path(path).is_file():
            self.p("(figure not available yet)")
            return
        from docx.shared import Inches

        self.d.add_picture(str(path), width=Inches(width_in))
        if caption:
            self.d.add_paragraph().add_run(caption).italic = True

    def pending(self, what: str) -> None:
        self.p(f"{what}: not run yet - this section fills in when its run has finished.")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.d.save(str(path))


# --- sections --------------------------------------------------------------------------

TECH = [
    ("Language / runtime", "Python 3.12 (Windows, macOS, Linux)"),
    ("Machine learning", "PyTorch (split-learning MLPs, ShieldLM), LightGBM, scikit-learn, imbalanced-learn"),
    ("Explainability", "SHAP, LIME, ATT&CK knowledge base, plain-English write-ups"),
    ("Detection content", "3,161 SigmaHQ rules, own Windows / Linux / cloud rules, MITRE ATT&CK v17 (706 techniques)"),
    ("Log parsing", "own parsers (syslog, auditd, web, mail, Suricata, Zeek, CloudTrail, k8s, Okta, O365...), "
                    "Windows .evtx via Get-WinEvent / the evtx library, LIEF for PE files"),
    ("Storage / search", "JSON-lines store + SQLite index (FTS5 trigram full-text, column indexes), incremental incidents"),
    ("Dashboard", "Streamlit (localhost only), role-based pages"),
    ("SIEM integration", "syslog UDP / TCP / TLS (RFC 5424/5425), webhook, Splunk HEC; formats CEF, LEEF 2.0, ECS, OCSF 1.3"),
    ("Delivery", "SQLite queue with ordered retries, circuit breaker, dead letter"),
    ("Collection", "HTTPS receiver (TLS 1.2+, self-signed certificate pinned by SHA-256), Python and PowerShell agents"),
    ("Security", "PBKDF2 password hashes, SHA-256 hash-chained audit log, HMAC pseudonyms, per-username lockout"),
    ("Reports", "python-docx (Word), Markdown, self-contained HTML; NIST CSF 2.0 / ISO 27001:2022 evidence"),
    ("Scheduling", "Windows Task Scheduler, macOS launchd, Linux systemd user units"),
    ("Quality", "pytest (700+ tests), ruff, mypy, every-combination stress harness, pip-audit"),
]  # fmt: skip


def section_overview(doc: Doc) -> None:
    doc.h("1. What SplitShield-XAI is", 1)
    doc.p("SplitShield-XAI is **an explainable detection and response layer that works alongside a SIEM**. "
          "It reads network traffic, logs from Windows, Linux, macOS, web, mail and cloud services, URLs, files, "
          "industrial sensors and user activity; nine specialist models and 3,600+ rules find attacks; findings "
          "about the same IP, user or host are correlated into incidents with an ATT&CK attack path; every alert is "
          "explained in plain English, delivered to the SIEM in its own format, and the models keep improving "
          "only when they are not worse on fixed held-out data.")  # fmt: skip
    doc.p("It has two halves sharing one codebase: the **deployable SOC** and the **research** on split learning for "
          "intrusion detection (registered hypotheses H1-H5). Product category: SIEM companion / XDR-lite with "
          "NDR and log analytics - not a SIEM replacement; the SIEM stays the system of record.")  # fmt: skip


def section_architecture(doc: Doc) -> None:
    doc.h("2. Architecture: how everything is connected", 1)
    doc.image(diagram_architecture(), caption="Figure 1. Sources, models, correlation and outputs.")
    doc.h("Inside the models", 2)
    doc.p("Figures 2-4 show how the three most connected detectors are built inside: which parts score, "
          "which decide, and where the explanation comes from.")  # fmt: skip
    doc.image(diagram_flow_inside(), caption="Figure 2. Flow detector: ensemble, DDoS cascade, generalist and stack.")
    doc.image(diagram_phishing_inside(), caption="Figure 3. Phishing: URL model, threat intel and the page model.")
    doc.image(diagram_assistant_inside(), caption="Figure 4. Assistant: facts come from the data, wording is checked.")
    doc.p("Every source goes to the model or rules that fit it. Every finding becomes the same kind of event "
          "(time, source, entity, ATT&CK techniques, confidence, reason). Events are stored and indexed, posted "
          "to the shared risk board, correlated into incidents, forwarded and explained.")  # fmt: skip
    doc.h("Technologies", 2)
    doc.table(["Area", "Used"], [list(t) for t in TECH])


def section_models(doc: Doc) -> None:
    doc.h("3. The nine models and the rules", 1)
    fit = (_load(EXTRA / "fit_report.json") or {}).get("models", {})

    def v(name: str) -> str:
        m = fit.get(name, {})
        return (m.get("split") or {}).get("verdict") or (m.get("curve") or {}).get("verdict") or "-"

    rows = [
        ["Flow detector (stack v4) + DDoS specialist", "CICFlowMeter flows", "MLP + LightGBM ensemble, DDoS cascade, generalist + adversarially trained stack (Figure 2)", "P 0.9992 / R 0.9982; DDoS2019 R 0.9996; CIC-IDS2018 R 0.873", v("flow")],
        ["NetFlow", "NetFlow v9 records", "ensemble", "P 0.999 / R 0.961", v("netflow")],
        ["Zeek", "Zeek conn.log", "random forest + 60-s per-source context", "P 0.996 / R 0.9995", v("zeek")],
        ["Phishing URL (v5) + page model (v2)", "URLs in logs; uploaded pages / attachments", "LightGBM on host-name features; LightGBM on page features (Figure 3)", "URL: P 0.997 / R 0.602; page: R 0.64 at 1.2 % false alarms (CompPhish check)", v("phishing")],
        ["Malware", "PE files / hashes", "LightGBM on EMBER features", "92.6 % at 1 % false positives (v2)", v("malware")],
        ["ICS (v3)", "plant sensors", "autoencoder on HAI 22.04 + 23.05", "22.04: P 0.896 / R 0.139; 23.05: P 0.70 / R 0.23", v("ics")],
        ["Insider", "user activity", "behaviour model", "P 0.939 / R 0.953", v("insider")],
        ["Intent (assistant)", "questions", "text classifier (TF-IDF + linear model)", "accuracy 0.902 on unseen phrasings", v("intent")],
        ["ShieldLM (assistant)", "incidents", "small transformer (4 layers, 1.9 M parameters), RLCD-tuned, faithfulness gate", "perplexity 1.21; reward-model pair accuracy 0.94", v("shieldlm")],
        ["Rules", "all logs", "Sigma + own rules", "held-out attacks detected 71.6 %", "-"],
    ]  # fmt: skip
    doc.table(["Model", "Reads", "Method", "Held-out result", "Fit verdict"], rows)
    doc.p("Fit verdicts come from learning curves (score vs. training size) and the train / validation / test gap "
          "of every model (results/extra/fit_report.md); they are re-checked weekly and a changed verdict is "
          "notified, never acted on automatically.")  # fmt: skip


def section_ics(doc: Doc) -> None:
    doc.h("ICS: more data, what was tried, what is deployed", 2)
    d = _load(EXTRA / "ics_v3.json") or {}
    dep = _load(EXTRA / "ics_v3_deploy.json") or {}
    if not d:
        doc.pending("ICS v3 results")
        return
    c = d.get("champion", {})
    rows = [["deployed v2 (before)", "-", f"{c.get('test_A', {}).get('precision')} / {c.get('test_A', {}).get('recall')}",
             f"{c.get('test_B', {}).get('precision')} / {c.get('test_B', {}).get('recall')}"]]  # fmt: skip
    names = {"A": "A: autoencoder, normal data of both versions (deployed as v3)",
             "B": "B: hybrid, labelled attacks (per-sensor features)", "C": "C: hybrid, sensor-independent features"}  # fmt: skip
    for k, name in names.items():
        m = d.get("mean", {}).get(k, {})
        rows.append([name, f"{m.get('train', {}).get('f1')} / {m.get('val', {}).get('f1')}",
                     f"{m.get('test_A', {}).get('precision')} / {m.get('test_A', {}).get('recall')}",
                     f"{m.get('test_B', {}).get('precision')} / {m.get('test_B', {}).get('recall')}"])  # fmt: skip
    doc.table(["Model (3-seed mean)", "Train / val F1", "HAI 22.04 test P / R", "HAI 23.05 test P / R"], rows)
    doc.bullets([
        "Data: HAI 22.04 and HAI 23.05 (same plant, newer run; 7 renamed setpoint tags matched by value range); "
        "HAIEnd 23.05 (internal controller tags) and BATADAL (a water network) as report-only experiments; HAI 21.03 "
        "stays the blind set.",
        "v2 only worked on the plant version it was trained on: on HAI 23.05 it alarmed almost nonstop. v3 (A) works on "
        "both; its 22.04 recall is lower (14 vs 15 of 34 attacks), so it was deployed as an **owner exception**.",
        "Learning from labelled attacks (B, C) overfits: each HAI attack targets different sensors, so the training "
        "attacks do not generalise to new ones. The internal controller tags (HAIEnd) did not help; BATADAL reached test "
        "F1 0.31-0.43. More attack types (e.g. SWaT / WADI) would be needed.",
        f"Decision record: {dep.get('owner_exception', d.get('decision', ''))}",
    ])  # fmt: skip


def section_assistant(doc: Doc) -> None:
    doc.h("The assistant: asking it questions", 2)
    doc.p("The dashboard assistant answers questions about attacks and about the user's own logs. A question is "
          "spell-corrected, its intent is recognised (text classifier, 0.902 on unseen phrasings) and the facts are "
          "read from the event store and the MITRE ATT&CK knowledge base (706 techniques). **ShieldLM**, a small "
          "transformer trained in this project, writes the incident narratives; every narrative is checked for "
          "faithfulness against the facts and replaced by a fixed template if it fails, so a weak model can only "
          "change the wording, never the facts.")  # fmt: skip
    doc.table(["Question", "What it does"], [
        ["what is kerberoasting? / explain brute force", "explains the technique: what it is, types, tactic, platforms, signs, defences"],
        ["is lateral movement happening?", "checks the stored logs: ongoing / happened / not seen"],
        ["who is attacking us? / what did 203.0.113.45 do?", "top sources, or one entity's incidents and evidence"],
        ["how do I stop password spraying?", "defences, and contain / investigate / fix / verify steps"],
        ["show failed logins for bob yesterday", "plain-English log search"],
        ["make a report of this week", "a report from the template"],
        ["what is the weather today?", "declined: it answers security questions only"],
    ])  # fmt: skip


def section_correlation(doc: Doc) -> None:
    doc.h("4. How different logs are correlated", 1)
    doc.image(diagram_correlation(), caption="Figure 5. Cross-log correlation into one incident (validated settings).")
    doc.bullets([
        "**Entity:** events are grouped by the IP, user or host they are about.",
        "**Time:** a group is split where two consecutive events are more than 15 minutes apart (configurable).",
        "**Attack chain:** a group becomes an incident when it carries ATT&CK evidence, a failed-logon chain "
        "(T1110 -> T1078), cloud probing, web scanning or a port scan; techniques are ordered by tactic.",
        "**Shared risk board:** findings of other model families about the same entity, or one link away, "
        "within 24 h raise the severity by up to 2. Very connected entities (hubs) are not followed. "
        "The boost never lowers a severity or hides an alert.",
        "**Corroboration:** an incident seen by two or more independent sensors is marked as corroborated.",
        "**Events with no IP, user or host** are grouped per log source and file (they were one 'unknown' entity), "
        "and a group also ends after 24 h of nonstop activity, so one never-pausing source cannot become one "
        "ever-growing incident (owner decision, 2026-10-23).",
    ])  # fmt: skip
    ex = _load(EXTRA / "correlation_example.json")
    if ex:
        doc.h("Worked example (demo attack data)", 2)
        doc.p(ex.get("summary", ""))
        doc.table(["Time", "Sensor", "What", "ATT&CK"], [[r["time"], r["source"], r["what"][:70], r["techniques"]]
                                                       for r in ex.get("timeline", [])[:20]])  # fmt: skip


def section_paths(doc: Doc) -> None:
    doc.h("5. Real time and datasets: when an alert appears, and the delay", 1)
    doc.image(diagram_paths(), caption="Figure 6. Real-time paths and the dataset path.")
    lat = _load(EXTRA / "latency.json") or {}
    if not lat:
        doc.pending("Alert-delay measurement")
    else:
        def r(name: str, st: dict[str, Any], wait: float) -> list[Any]:
            if not st or st.get("median_ms") is None:
                return [name, "-", "-", f"{wait:g} s", "-"]
            return [name, f"{st['median_ms']} ms", f"{st['p95_ms']} ms", f"{wait:g} s", f"~{st['p95_ms'] / 1000 + wait:.1f} s"]

        w = list((lat.get("polling_s") or {}).values()) or [2.0, 2.0, 1.0, 0.5]  # logs, agents, incidents, flows
        doc.table(["Path", "Processing (median)", "Processing (95th pct)", "Waiting by design", "Worst case"], [
            r("Log line -> alert -> SIEM", lat.get("log_line_to_siem", {}), w[0]),
            r("Agent -> HTTPS receiver -> SIEM", lat.get("agent_to_siem", {}), w[1]),
            r("Live flow batch -> alert", (lat.get("flow_to_alert") or {}).get("batch_of_256", {}), w[3]),
            r("Incident built and forwarded", lat.get("incidents_build_and_push", {}), w[2]),
        ])  # fmt: skip
        doc.image(chart_latency(lat), width_in=6, caption="Figure 7. Worst-case delay per path.")
        up = lat.get("dataset_upload", {})
        if up:
            doc.p(f"Dataset upload where every line is an attack (worst case: each line becomes an alert): "
                  f"{up.get('lines', 0):,} lines in {up.get('seconds')} s = **{up.get('logs_per_s', 0):,} logs/s**. "
                  "Ordinary mixed logs run at about **60,000 logs/s**. Alerts appear while the file is analysed.")  # fmt: skip
    doc.bullets([
        "**Real time:** an alert carries the event's own time and is marked 'ongoing' while the attack goes on; "
        "it is forwarded at once (or queued if the SIEM is down) and the dashboard's live monitor shows it.",
        "**Datasets and uploads:** the original timestamps are kept, so an old attack shows as 'happened' with "
        "its real date; incidents and reports are built the same way; nothing is sent unless forwarding is on.",
        "**What an alert says:** what happened, the entity, the ATT&CK technique and tactic, confidence, which "
        "sensors agree, which other models saw the same entity, and what to do.",
    ])  # fmt: skip


def section_enterprise(doc: Doc) -> None:
    doc.h("6. Features for teams next to a SIEM", 1)
    doc.bullets([
        "**Users, roles, permissions** (admin / analyst / viewer + own roles; admin-only changes; lockout).",
        "**Tamper-evident audit log** (SHA-256 chain; optional forwarding).",
        "**Guaranteed SIEM delivery** (queue, ordered retries, dead letter) in CEF / LEEF / ECS / OCSF over UDP / TCP / TLS / webhook.",
        "**Central log collection** (HTTPS receiver, pinned certificate, per-agent keys) and a **read-only REST API**.",
        "**Incident cases** (status, assignee, notes) and **silent-agent alerts** (T1562) with clock-skew checks.",
        "**Reports** for any period from an editable template; **compliance evidence** (NIST CSF 2.0, ISO 27001:2022).",
        "**Retention** (30 days by default, archived), **privacy** option (pseudonymised names to the SIEM).",
        "**Evolve or freeze** the models; **Windows / macOS / Linux** installers and background jobs.",
    ])  # fmt: skip


def _md_table(path: Path, heading: str) -> tuple[list[str], list[list[str]]]:
    """The first Markdown table under ``heading`` in a doc (header, rows)."""
    try:
        lines = path.read_text("utf-8").splitlines()
    except OSError:
        return [], []
    def title(ln: str) -> str:
        t = re.sub(r"^[#\s]+", "", ln)
        return "".join(ch for ch in t if not (0x2600 <= ord(ch) <= 0x27BF or ord(ch) >= 0x1F000 or ord(ch) == 0xFE0F)).strip()

    heads = [(i, title(ln)) for i, ln in enumerate(lines) if ln.startswith("#")]
    start = next((i for i, t in heads if t == heading), None)
    if start is None:
        start = next((i for i, t in heads if heading in t), None)
    if start is None:
        return [], []
    rows = []
    for ln in lines[start + 1 :]:
        if ln.startswith("#"):
            break
        if ln.startswith("|"):
            rows.append([re.sub(r"\*\*|`|✅|❌", "", c).strip() for c in ln.strip().strip("|").split("|")])
        elif rows:
            break
    if len(rows) < 2:
        return [], []
    return rows[0], [r for r in rows[2:]]


def section_research_design(doc: Doc) -> None:
    doc.h("7. The research: split learning for intrusion detection", 1)
    doc.p("**Split learning** cuts a neural network in two: each client (a site holding traffic) runs the first "
          "layers and sends only the intermediate activations; the server runs the rest. Raw traffic never leaves "
          "the client - but activations can still leak information, attackers can perturb inputs, and "
          "explanations can drift. The study measures these trade-offs against centralized training, FedAvg and "
          "SplitFed, with five registered hypotheses, 5 seeds per headline number and a control for every metric.")  # fmt: skip
    rd = PROJECT_ROOT / "docs" / "RESEARCH.md"
    head, rows = _md_table(rd, "Research hypotheses (5 seeds, controls applied)")
    if rows:
        doc.table(head, rows)
    head, rows = _md_table(rd, "Detection across datasets (macro-F1, test set, mean over seeds)")
    if rows:
        doc.h("Detection across datasets", 2)
        doc.table(head, rows)
    head, rows = _md_table(rd, "Research pipeline (`run_pipeline.py`, stages 0–9)")
    if rows:
        doc.h("Research pipeline", 2)
        doc.table(head, rows)
    doc.bullets([
        "**No metric without its control** (random-matched noise, shuffled labels).",
        "**Selection on validation, test reported once; blind sets scored once at the end.**",
        "**Registered design kept:** corrections are labelled follow-ups shown next to the registered numbers.",
    ])  # fmt: skip


def section_datasets(doc: Doc) -> None:
    doc.h("Datasets", 2)
    dd = PROJECT_ROOT / "docs" / "DATASETS.md"
    head, rows = _md_table(dd, "Datasets")
    if rows:
        doc.table(head, rows)
    head, rows = _md_table(dd, "Sources and citations")
    if rows:
        doc.table(head, rows)


def section_operations(doc: Doc) -> None:
    doc.h("Operating it day to day", 2)
    doc.bullets([
        "**Morning:** Home shows alerts today, open incidents, this computer's logs, health; the daily report "
        "(yesterday) is in Reports.",
        "**Triage:** Incidents & review - open cases first; assign, add notes, set status; mark detector alerts "
        "Real / False alarm (this teaches the models).",
        "**Investigate:** search in plain English, ask the assistant, open the ATT&CK path and the write-up.",
        "**Tune:** own rules (tested on stored logs before saving), allow-list, stricter thresholds.",
        "**Weekly:** the weekly and compliance reports; the fit check and self-improvement decisions on "
        "Self-improvement & health.",
    ])  # fmt: skip


def section_appendix(doc: Doc) -> None:
    doc.h("Appendix A. Install and run", 1)
    doc.table(["", "Install", "Start"], [["Windows", "double-click INSTALL.cmd", "START.cmd"],
                                       ["macOS / Linux", "sh install.sh", "sh START.sh"],
                                       ["Agents", "install_agent.ps1 / sudo sh install_agent.sh", "start with the computer"],
                                       ["Research", "LAUNCH.cmd / python ai_engineer.py", "about 12 h on this PC"]])  # fmt: skip
    doc.h("Self-improvement, safely", 2)
    doc.bullets([
        "All 9 models learn from new labelled data and analyst feedback; a change is kept only if it is not worse on "
        "fixed held-out data and not slower, otherwise it is rolled back automatically.",
        "The assistant learns from every thumbs up / down on its replies, under the same rule.",
        "Nightly speed profiling proposes tested speed-ups; code never changes on its own (each one is approved).",
        "The optional update agent (off by default) brings new Sigma rules, ATT&CK and threat-intel feeds only if the "
        "tests pass and false alarms do not rise.",
        "Health watch, a weekly fit check and desktop notifications report every decision and failure.",
    ])  # fmt: skip
    doc.h("Frequently asked questions", 2)
    doc.bullets([
        "**Does any data leave the PC?** Only what is sent to the SIEM, logs the user's own agents send over the LAN, and "
        "public rules / feeds the update agent downloads if switched on. The dashboard is reachable from the PC only.",
        "**Does it block attackers on its own?** Only if auto-block is switched on (Windows Firewall, pf or nftables).",
        "**How big is it?** The models are about 0.4 GB; the working copy is mostly datasets, needed only to rebuild them.",
        "**Can it test my own data?** Yes: 'Test any dataset' on the dashboard gives a report card.",
        "**Is it a SIEM?** No. It works next to Splunk, Elastic, QRadar, Sentinel, Wazuh or any syslog / OCSF lake.",
    ])  # fmt: skip
    doc.h("Appendix B. What is in the project", 1)
    py = [f for f in PROJECT_ROOT.rglob("*.py") if not any(x in f.parts for x in (".venv", ".claude", "third_party", "__pycache__"))]
    tests = [f for f in py if f.parent.name == "tests"]
    soc = [f for f in py if "soc" in f.parts]
    doc.table(["Part", "Count"], [["Python files", len(py)], ["Test files", len(tests)], ["SOC modules (src/soc)", len(soc)],
                                  ["Sigma rules loaded", "2,507"], ["Dashboard pages", 25], ["Trained models", 9],
                                  ["Documentation pages (docs/)", len(list((PROJECT_ROOT / 'docs').glob('*.md')))]])  # fmt: skip
    doc.table(["Folder", "Holds"], [["dashboard.py", "the SOC dashboard"], ["src/soc", "parsers, rules, Sigma, correlation, "
              "users, audit, delivery, collector, reports, retention, cases, controller"], ["src/mlareas", "phishing, malware, ICS, insider"],
              ["src/nlp", "the ShieldLM assistant"], ["src/{data,models,training,attack,defense,explain,eval}", "the research"],
              ["scripts", "installers, agents, benchmarks, stress tests, reports"], ["tests", "700+ automated tests"],
              ["docs", "all documentation"], ["artifacts", "trained models"], ["results", "every result and report"]])  # fmt: skip


def section_research(doc: Doc, rerun: Path | None, rerun2: Path | None) -> None:
    doc.h("Reproducibility: registered results vs. two re-runs", 2)
    reg = _load(RESULTS_DIR / "final_verdicts.json") or {}
    r1 = _load(rerun / "results" / "final_verdicts.json") if rerun else None
    r2 = _load(rerun2 / "results" / "final_verdicts.json") if rerun2 else None
    rep0 = _load(EXTRA / "reproducibility_summary.json") or {}
    if r1 is None and r2 is None and rep0.get("verdicts_same"):  # copies deleted: the recorded comparison
        r1 = r2 = reg
    doc.table(["Hypothesis", "Registered", "Re-run 1", "Re-run 2"],
              [[h, reg.get(h, "-"), (r1 or {}).get(h, "-"), (r2 or {}).get(h, "-")] for h in ("H1", "H2", "H3", "H4", "H5")])  # fmt: skip
    h2b = _load(EXTRA / "H2b.json")
    if h2b:
        doc.h("H2b: pre-registered confirmatory test", 2)
        doc.p(f"Registered before any run (docs/PREREG_H2b.md): the corrected pipeline on new seeds 5-9, the "
              f"registered per-seed Wilcoxon rule, verdict = at least 3 of 5 seeds. Result: **{h2b['verdict']}**; "
              f"Bonferroni (p < 0.01) {h2b['seeds_supported_bonferroni']} / {len(h2b['rows'])}. The registered H2 "
              "verdict (not supported) and this confirmatory result agree; the exploratory follow-up did not replicate.")  # fmt: skip
        doc.table(["Seed", "PGD drift", "Control drift", "Median paired diff", "p"],
                  [[r["seed"], round(r["pgd_drift"], 3), round(r["control_drift"], 3),
                    round(r["median_paired_diff"], 3), round(r["p"], 4)] for r in h2b["rows"]])  # fmt: skip
    h2c = _load(EXTRA / "H2c.json")
    if h2c:
        doc.h("H2c: pre-registered test with a power analysis", 2)
        doc.p(f"H2b had only about 35 % power at 16 instances per seed. H2c (docs/PREREG_H2c.md, registered before "
              f"any run) repeats the same test and decision rule on new seeds "
              f"{h2c['seeds'][0]}-{h2c['seeds'][-1]} with {h2c['rows'][0]['n_instances']} instances per seed. "
              f"Result: **{h2c['verdict']}**; Bonferroni (p < 0.01) {h2c['seeds_supported_bonferroni']} / "
              f"{len(h2c['rows'])}; Stouffer z = {h2c['stouffer_z']}. PGD changes SHAP explanations more than "
              "random noise of the same size, but the effect is small (a few hundredths on a 0-1 scale). The "
              "registered H2 verdict and H2b stay as recorded; H2c is the adequately powered test.")  # fmt: skip
        doc.table(["Seed", "PGD drift", "Control drift", "Median paired diff", "p"],
                  [[r["seed"], round(r["pgd_drift"], 3), round(r["control_drift"], 3),
                    round(r["median_paired_diff"], 3), f"{r['p']:.1e}"] for r in h2c["rows"]])  # fmt: skip
    rep = _load(EXTRA / "reproducibility_summary.json")
    if rep:
        doc.p(rep.get("text", ""))
        if rep.get("differences"):
            doc.table(["Result", "Registered", "Re-run", "Why"], [[d["what"], d["registered"], d["rerun"], d["why"]]
                                                              for d in rep["differences"]])  # fmt: skip
    else:
        doc.pending("Reproducibility summary")


def section_testing(doc: Doc) -> None:
    doc.h("8. Testing, stress and security", 1)
    cs = _load(EXTRA / "combo_stress.json")
    if cs:
        doc.p(f"Every-combination stress test: **{cs['passed']:,} of {cs['combinations']:,} combinations passed** "
              f"({cs['failed']} failed) in {cs['hours']} h; each ran the whole SOC pipeline end to end.")  # fmt: skip
    else:
        doc.pending("Every-combination stress test")
    sec = PROJECT_ROOT / "docs" / "SECURITY.md"
    if sec.is_file():
        rows = []
        for ln in sec.read_text("utf-8").splitlines():
            m = re.match(r"^\|\s*(\d+)\s*\|(.+)\|$", ln)
            if m:
                c = [x.strip() for x in m.group(2).split("|")]
                rows.append([m.group(1), c[0], c[1][:120], c[2], c[3][:90]])
        doc.p("Penetration test (internal): every finding fixed or listed as an accepted risk.")
        doc.table(["#", "Area", "Finding", "Severity", "Fix"], rows)
    soak = _load(EXTRA / "soak_test.json")
    if soak:
        doc.h("End-to-end soak test", 2)
        doc.p(f"{soak['agents']} agents sending {soak['lines_per_s']:.0f} lines/s for {soak['minutes']} min through the real "
              f"HTTPS receiver, detection, incidents and guaranteed delivery to a mock SIEM, with a "
              + ", ".join(f"{o['what']} outage of {o['seconds']} s" for o in soak["outages"])
              + f": {soak['stored_once']:,} / {soak['lines_written']:,} lines stored exactly once, "
              f"{soak['siem_received']:,} / {soak['alerts']:,} alerts at the SIEM ({soak['siem_duplicates']} duplicates), "
              f"delay median {soak['delay_ms']['median']} ms / 95th percentile {soak['delay_ms']['p95']} ms, "
              f"no memory growth after warm-up.")  # fmt: skip
    st = _load(EXTRA / "stress_test.json")
    if st:
        doc.p(f"Stress test ({st['when']}): **{st['passed']} passed, {st['failed']} failed** (hostile input, "
              "concurrency, faults and recovery, a 1 M-event volume).")  # fmt: skip
    samp = _load(EXTRA / "combo_stress_newest_sample.json")
    if samp:
        doc.p(f"Every-combination sample on the newest code (every 40th): **{samp['passed']:,} of "
              f"{samp['combinations']:,} passed**.")  # fmt: skip
    lin = _load(EXTRA / "linux_host_test.json")
    if lin:
        doc.p(f"Real Linux host ({lin['distro']} on WSL 2, systemd, {lin['python']}): the agent sent its test log, "
              f"auth.log and the systemd journal to the Windows receiver over pinned HTTPS; "
              f"{lin['test_lines_stored_once']} / {lin['test_lines']} test lines stored once, the host name kept, the SSH "
              "brute force became an incident: " + ("all checks passed." if lin["ok"] else "a check failed."))  # fmt: skip
    crs = _load(EXTRA / "crs_evaluation.json")
    if crs:
        h = crs["heldout"]["result"]
        doc.p(f"OWASP Core Rule Set on access logs ({crs['crs_rules_used']} rules), held-out AIT web servers: attack steps "
              f"{h['rules']['steps_caught']} -> {h['rules+crs']['steps_caught']} of {h['rules']['steps']}, attack lines "
              f"{h['rules']['line_recall']:.1%} -> {h['rules+crs']['line_recall']:.1%}, false alarms "
              f"{h['rules']['normal_alerted']} -> {h['rules+crs']['normal_alerted']}; decision: {crs['decision']}.")  # fmt: skip
    rep = _load(EXTRA / "final_copy_reproduction.json")
    if rep:
        doc.h("Final copy: the frozen code reproduces every deployed model", 2)
        cmpd = rep["compared"]
        doc.p(f"A fresh copy of the frozen code re-ran every area step and the promotions behind each deployed model. "
              f"Of {cmpd['numbers']:,} result numbers, {cmpd['identical']:,} are identical, {cmpd['within_1pct_or_0.01']:,} "
              f"within 1 % and {cmpd['different']:,} different; every difference is explained below.")  # fmt: skip
        doc.table(["Deployed model", "Original", "Final copy"], [[m["model"], m["original"], m["final_copy"]]
                                                                 for m in rep["deployed_models"]])  # fmt: skip
        doc.table(["File", "Different numbers", "Why"], [[d["file"], d["numbers"], d["why"]]
                                                          for d in rep["differences_explained"]])  # fmt: skip
    sp = _load(EXTRA / "speed_benchmark.json") or {}
    big = _load(EXTRA / "speed_10m_finding.json") or {}
    if sp or big:
        doc.h("Scale", 2)
        rows = [[f"{r['events']:,}", r.get("profile"), r.get("index_build_s"), r.get("incidents_first_s"),
                 max(list(r.get("search_ms", {}).values()) or [0]), r.get("db_mb")] for r in sp.get("runs", [])]  # fmt: skip
        if rows:
            doc.table(["Events", "Setting", "Index build (s)", "First incidents (s)", "Slowest search (ms)", "Index (MB)"], rows)
        if big:
            doc.p(f"10 million events: {big.get('conclusion', '')} Run 1: {big['run_1']['cause']} "
                  f"({big['run_1']['memory_gb']} GB), fixed. Run 2: {big['run_2']['cause']} "
                  f"({big['run_2']['incidents_stored']:,} synthetic incidents).")  # fmt: skip
    doc.p("Libraries: re-audited after the freeze; pillow 13.0.0 (with streamlit 1.66.1), scapy 2.8.0, certifi and black "
          "updated; one accepted advisory remains (lightgbm, see Security). The SBOM is security/sbom.cyclonedx.json.")  # fmt: skip
    bugs = _load(EXTRA / "bug_log.json") or []
    if bugs:
        doc.h("Bugs found and fixed in the final phase", 2)
        doc.table(["When", "Area", "Bug", "Fix", "Found by"], [[b.get("when"), b.get("area"), b.get("bug"), b.get("fix"),
                                                             b.get("found_by")] for b in bugs])  # fmt: skip


def section_blind(doc: Doc) -> None:
    doc.h("9. Blind test (scored once, at the very end)", 1)
    d = _load(EXTRA / "blind_final.json")
    if not d:
        doc.pending("Final blind test")
        return
    doc.p(d.get("note", ""))
    doc.table(["Area", "Result"], [[k, v] for k, v in d.get("summary", {}).items()])
    r2 = _load(EXTRA / "blind_round2.json")
    if not r2:
        return
    doc.h("Blind round 2 (new data sets, after the improvements)", 2)
    doc.p("The first blind sets were used up, so new ones were downloaded and kept untouched until one "
          "scoring run on the frozen code (tag freeze-2026-10-24-round2). Results as they came out:")  # fmt: skip
    rows = []
    f = r2.get("flows", {})
    for k in ("before (deployed until 2026-10-24)", "now (flow stack)"):
        if k in f:
            c = f[k]["confirmed"]
            rows.append([f"Flows, HIKARI - {k}", f"{c['detection_rate']:.1%} caught, {c['false_alarm_rate']:.1%} false alarms"])
    w = r2.get("windows", {})
    if w:
        rows.append(["Windows, EVTX-to-MITRE-Attack", f"{w['detected_rate']:.1%} detected, {w['right_tactic_rate']:.1%} right tactic"])
    ph = r2.get("phishing", {})
    for k in ("url v4 (before)", "url v5 (now)", "page model"):
        if k in ph:
            rows.append([f"Phishing, CompPhish - {k}", f"recall {ph[k]['detection_rate']:.3f}, "
                         f"{ph[k]['false_alarm_rate']:.1%} false alarms, precision {ph[k]['precision']:.3f}"])  # fmt: skip
    lg = r2.get("logs", {})
    for k in ("russellmitchell", "santos"):
        if isinstance(lg.get(k), dict) and "steps_caught" in lg[k]:
            v = lg[k]
            rows.append([f"Logs, AIT-LDS v2 {k}", f"{v['steps_caught']} / {v['n_steps']} attack steps, "
                         f"{v['line_recall']:.1%} attack lines, {v['false_alarm_rate']:.3%} false alarms"])  # fmt: skip
    doc.table(["Area", "Result"], rows)
    after = _load(EXTRA / "ait2_rules_after.json")
    if after and "santos" in after:
        s = after["santos"]
        doc.p(f"After the blind round (NOT blind; rules written from russellmitchell, santos is the check): santos "
              f"{s['steps_caught']} / {s['n_steps']} attack steps, {s['line_recall']:.1%} attack lines, "
              f"{s['false_alarm_rate']:.3%} false alarms; incident precision {s['incidents']['board_on']['precision']:.2f} "
              "(open: see Limitations).")  # fmt: skip


def section_limits_product(doc: Doc) -> None:
    doc.h("10. Limitations", 1)
    lim = PROJECT_ROOT / "docs" / "LIMITATIONS.md"
    rows = []
    if lim.is_file():
        for ln in lim.read_text("utf-8").splitlines():
            if ln.startswith("|") and not ln.startswith("|---") and "Limitation" not in ln:
                c = [x.strip() for x in ln.strip("|").split("|")]
                if len(c) >= 3:
                    rows.append(c[:3])
    doc.table(["Limit", "What was done", "Still open"], rows)
    doc.h("11. From project to product", 1)
    doc.bullets([
        "**Real-world validation:** trained on public academic datasets; pilots with real organisations are the proof.",
        "**Licences:** some datasets are research-only; Sigma rules need attribution (DRL); commercial use needs a review.",
        "**Scale:** tested on one 16 GB PC; multi-machine deployment and high availability are future work.",
        "**Trust:** an independent penetration test, SOC 2 / ISO 27001 for a company, signed installers and updates.",
        "**Focus:** one sharp use case, e.g. explainable detection for small SOCs that already have a SIEM.",
    ])  # fmt: skip
    doc.h("12. Publication potential", 1)
    doc.bullets([
        "**Paper A (research):** split learning for intrusion detection - privacy vs. accuracy vs. explainability; "
        "registered hypotheses, 5 seeds, controls, exact reproducibility, an honest negative result (H2).",
        "**Paper B (system):** an explainable, self-improving SIEM companion with cross-model risk-board "
        "correlation and gated updates; blind tests on 6 datasets reported as they came out.",
    ])  # fmt: skip


def build(rerun: Path | None = None, rerun2: Path | None = None) -> Path:
    doc = Doc()
    doc.d.add_heading("SplitShield-XAI - final report", 0)
    doc.p(f"Built {time.strftime('%Y-%m-%d %H:%M')} from the project's result files.")
    section_overview(doc)
    section_architecture(doc)
    section_models(doc)
    section_ics(doc)
    section_assistant(doc)
    section_correlation(doc)
    section_paths(doc)
    section_enterprise(doc)
    section_operations(doc)
    section_research_design(doc)
    section_datasets(doc)
    section_research(doc, rerun, rerun2)
    section_testing(doc)
    section_blind(doc)
    section_limits_product(doc)
    section_appendix(doc)
    doc.save(OUT)
    return OUT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun", type=Path, default=None)
    ap.add_argument("--rerun2", type=Path, default=None)
    a = ap.parse_args()
    print(build(a.rerun, a.rerun2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
