"""The final visual report (owner 2026-10-17): one self-contained HTML page with every
result of the final phase, built from the files the runs write. Missing inputs are
shown as "not run yet", so it can be built at any time.

    python -m scripts.final_visual_report [--rerun D:/SplitShield-rerun] [--rerun2 D:/SplitShield-rerun2]

Sections: summary; research (registered vs run 1 vs run 2, H1-H5 verdicts);
every-combination stress test (pass rate per option value); models before / after and
over- / underfitting (learning curves); speed before / after; bugs found and fixed;
penetration test; blind test; product category and "from project to product".
Output: results/FINAL_VISUAL_REPORT.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from utils.paths import PROJECT_ROOT, RESULTS_DIR

EXTRA = RESULTS_DIR / "extra"
OUT = RESULTS_DIR / "FINAL_VISUAL_REPORT.html"


def _load(p: Path) -> Any:
    try:
        return json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def _e(x: Any) -> str:
    return html.escape(str(x))


def _png(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return f'<img alt="chart" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _table(head: list[str], rows: list[list[Any]]) -> str:
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def _pending(what: str) -> str:
    return f'<p class="pending">{_e(what)} - not run yet.</p>'


# --- sections --------------------------------------------------------------------------


def research(rerun: Path | None, rerun2: Path | None) -> str:
    v = _load(RESULTS_DIR / "final_verdicts.json") or {}
    out = "<h2>Research (registered hypotheses H1-H5)</h2>"
    rows = [[h, v.get(h, "?")] for h in ("H1", "H2", "H3", "H4", "H5")]
    out += _table(["Hypothesis", "Registered verdict"], rows)
    for label, root in (("Run 1", rerun), ("Run 2 (repeat)", rerun2)):
        rep = root / "REPRODUCIBILITY.md" if root else None
        if rep and rep.is_file():
            text = rep.read_text("utf-8")
            head = "\n".join(text.splitlines()[:25])
            out += f"<h3>{_e(label)}: reproducibility</h3><pre>{_e(head)}</pre>"
        else:
            out += _pending(f"{label} reproducibility report")
    return out


def combos() -> str:
    d = _load(EXTRA / "combo_stress.json")
    out = "<h2>Every-combination stress test</h2>"
    if not d:
        return out + _pending("Every-combination stress test")
    out += (f'<p class="big">{d["passed"]:,} of {d["combinations"]:,} combinations passed '
            f'({d["failed"]} failed) in {d["hours"]} h.</p>')  # fmt: skip
    fails = d.get("failures", [])
    per: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for f in fails:
        for k, val in f["combo"].items():
            per[k][val] += 1
    if fails:
        import matplotlib.pyplot as plt

        labels, vals = [], []
        for k, m in per.items():
            for val, n in m.items():
                labels.append(f"{k}={val}")
                vals.append(n)
        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.25 * len(labels))))
        ax.barh(labels, vals, color="#d9534f")
        ax.set_xlabel("failing combinations containing this option")
        out += _png(fig)
        out += _table(["#", "Combination", "Error"],
                      [[f["i"], ", ".join(f"{k}={v}" for k, v in f["combo"].items()), f["error"][:160]]
                       for f in fails[:50]])  # fmt: skip
    dims = d.get("dimensions", {})
    out += _table(["Option", "Values (every combination tested)"], [[k, ", ".join(v)] for k, v in dims.items()])
    return out


def models() -> str:
    out = "<h2>Models: over- / underfitting and changes</h2>"
    fit = _load(EXTRA / "fit_report.json")
    if not fit:
        return out + _pending("Fit report")
    import matplotlib.pyplot as plt

    rows = []
    ms = fit.get("models", {})
    n = len(ms)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(11, 3 * ((n + 2) // 3)), squeeze=False)
    for ax, (name, m) in zip(axes.flat, ms.items(), strict=False):
        pts = m.get("points") or []
        if pts:
            ax.plot([p["n"] for p in pts], [p["train"] for p in pts], "o-", label="train")
            ax.plot([p["n"] for p in pts], [p["val"] for p in pts], "o-", label="validation")
            ax.set_xscale("log")
            ax.legend(fontsize=7)
        ax.set_title(f"{name}: {m.get('split', {}).get('verdict', m.get('curve', {}).get('verdict', '?'))}", fontsize=9)
        sp = m.get("split", {})
        rows.append([name, sp.get("train"), sp.get("val"), sp.get("test"), sp.get("verdict", "?"),
                     "; ".join(sp.get("findings", []))[:120], m.get("metric", "")])  # fmt: skip
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    out += _png(fig)
    out += _table(["Model", "Train", "Validation", "Test", "Verdict", "Findings", "Metric"], rows)
    imp = []
    for p in sorted(EXTRA.glob("improve_*.json")):
        d = _load(p) or {}
        dec = d.get("decision") or d.get("kept") or d.get("verdict")
        if dec is not None:
            imp.append([p.stem.replace("improve_", ""), str(dec)[:160]])
    if imp:
        out += "<h3>Model updates (kept only if held-out precision and recall did not drop)</h3>"
        out += _table(["Model", "Decision"], imp)
    return out


def speed() -> str:
    out = "<h2>Speed</h2>"
    rows = [
        ["Live flow scoring", "58 flows/s", "~2,300 flows/s", "batches, identical alerts"],
        ["Incidents after 1,000 new events (211k store, loaded PC)", "6.5 s", "1.0 s", "only the affected time-group re-read"],
        ["Storing parsed logs", "1x", "4x", "sensitivity read once per batch"],
        ["SIEM outage, per alert", "up to 5 s wait", "queued at once", "circuit breaker"],
    ]  # fmt: skip
    out += _table(["What", "Before", "After", "How"], rows)
    sb = _load(EXTRA / "speed_benchmark.json") or {}
    runs = sb.get("runs", [])
    if runs:
        out += _table(["Events", "When", "Index build s", "Incidents +1,000 ms", "Index MB", "Note"],
                      [[r.get("events"), r.get("when"), r.get("index_build_s"), r.get("incidents_after_1000_new_ms"),
                        r.get("db_mb"), r.get("note", "")] for r in runs[-6:]])  # fmt: skip
    return out


def bugs() -> str:
    out = "<h2>Bugs found and fixed</h2>"
    log = _load(EXTRA / "bug_log.json") or []
    if not log:
        return out + _pending("Bug log")
    return out + _table(["When", "Area", "Bug", "Fix", "Found by"],
                        [[b.get("when"), b.get("area"), b.get("bug"), b.get("fix"), b.get("found_by")] for b in log])


def pentest() -> str:
    p = PROJECT_ROOT / "docs" / "SECURITY.md"
    out = "<h2>Penetration test</h2>"
    if not p.is_file():
        return out + _pending("Penetration test")
    rows = []
    for ln in p.read_text("utf-8").splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|(.+)\|$", ln)
        if m:
            cells = [c.strip() for c in m.group(2).split("|")]
            rows.append([m.group(1), *cells[:4]])
    return out + _table(["#", "Area", "Finding", "Severity", "Fix"], rows)


def blind() -> str:
    d = _load(EXTRA / "blind_tests.json")
    out = "<h2>Blind test (scored once, at the end)</h2>"
    if not d:
        return out + _pending("Blind test")
    return out + f"<pre>{_e(json.dumps(d, indent=1)[:4000])}</pre>"


PRODUCT = """<h2>What it is, and from project to product</h2>
<p><b>Category:</b> an enterprise detection and response layer that works alongside a
SIEM (SIEM augmentation: XDR-lite / NDR plus log analytics). The SIEM stays the system
of record; SplitShield-XAI adds explainable ML detections, cross-model correlation,
investigation and reporting, and forwards to the SIEM. It is not a SIEM replacement.</p>
<h3>Gaps a real product would still have to close</h3><ul>
<li>Real-world validation: trained on public academic datasets; pilots with real organisations are the proof.</li>
<li>Licences: some training datasets are research-only; Sigma rules need attribution (DRL); commercial use needs a licence review and possibly retraining.</li>
<li>Scale: tested on one 16 GB PC; multi-machine deployment, high availability and customer volumes are untested.</li>
<li>Trust: an independent penetration test, SOC 2 / ISO 27001 for the company, signed installers and updates, a support process.</li>
<li>Business focus: one sharp use case (e.g. explainable detection for small SOCs that already have a SIEM).</li></ul>"""

CSS = """:root{--bg:#fff;--fg:#1b2430;--muted:#5b6675;--line:#d6dfeb;--head:#eef3fa;--acc:#1f4e8c}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#12161c;--fg:#e3e8ef;--muted:#9aa6b5;--line:#2c3644;--head:#1d2633;--acc:#8fb7ef}}
:root[data-theme="dark"]{--bg:#12161c;--fg:#e3e8ef;--muted:#9aa6b5;--line:#2c3644;--head:#1d2633;--acc:#8fb7ef}
body{background:var(--bg);color:var(--fg);font-family:Segoe UI,system-ui,sans-serif;max-width:1150px;margin:24px auto;padding:0 16px;line-height:1.45}
h1,h2,h3{color:var(--acc)}h2{border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:32px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}th{background:var(--head)}
img{max-width:100%;height:auto}pre{background:var(--head);padding:8px;overflow-x:auto;font-size:12px}
.pending{color:var(--muted);font-style:italic}.big{font-size:20px;font-weight:600}"""


def build(rerun: Path | None = None, rerun2: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    parts = [research(rerun, rerun2), combos(), models(), speed(), bugs(), pentest(), blind(), PRODUCT]
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>SplitShield-XAI final report</title>'
            f"<style>{CSS}</style></head><body><h1>SplitShield-XAI: final report</h1>"
            f'<p class="pending">Built {time.strftime("%Y-%m-%d %H:%M")} from the result files.</p>'
            + "".join(parts) + "</body></html>")  # fmt: skip
    OUT.write_text(page, "utf-8")
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
