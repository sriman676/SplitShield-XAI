"""Blind round 2 (owner 2026-10-24): new data sets nobody has looked at, scored ONCE.

After the 2026-10-24 improvements the first blind sets were used up (the owner released
them as normal data), so new ones were downloaded (owner-approved) and kept untouched in
D:/SplitShield-data/blind2 until this run:

  flows     HIKARI-2021 + HIKARI-2022 (Zeek FlowMeter CSV; real encrypted traffic, another
            network): the deployed detector BEFORE the improvements (backup folder) and NOW
  logs      AIT-LDS v2 (russellmitchell, santos): every labelled log file - attack steps
            caught, attack lines alerted, false alarms on sampled normal lines; incidents
            with the risk board off / on (entity truth: the attackers' addresses)
  windows   EVTX-to-MITRE-Attack (mdecrevoisier): one recording per technique folder -
            detected / right tactic / right technique
  phishing  CompPhish v4 (2024-25): URL model v4 (before) and v5 (now), and the page model

Each section is written once to results/extra/blind_round2.json; a section that is
already there is not scored again (``--section`` picks one).

    python -m scripts.blind_test_v2 [--section flows|logs|windows|phishing]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET  # noqa: N817

import numpy as np
import pandas as pd

from utils.logging_config import configure_logging, get_logger
from utils.paths import PROJECT_ROOT, RESULTS_DIR

_log = get_logger(__name__)
BLIND2 = Path("D:/SplitShield-data/blind2")
OUT = RESULTS_DIR / "extra" / "blind_round2.json"
NORMAL_EVERY = 200  # every 200th normal log line is sampled
NORMAL_CAP = 20_000  # normal lines per log file at most
ATTACK_CAP = 50_000  # attack lines per log file at most (evenly spread)
MAX_LINES = 3_000_000  # lines read per log file at most


def _rates(y: np.ndarray, a: np.ndarray) -> dict[str, Any]:
    tp, fp = int((a & y).sum()), int((a & ~y).sum())
    return {"attacks": int(y.sum()), "caught": tp, "normal": int((~y).sum()), "false_alarms": fp,
            "detection_rate": tp / max(1, int(y.sum())), "false_alarm_rate": fp / max(1, int((~y).sum())),
            "precision": tp / max(1, tp + fp)}  # fmt: skip


# --- flows ---------------------------------------------------------------------------


def _hikari() -> pd.DataFrame:
    parts = []
    z = BLIND2 / "hikari" / "ALLFLOWMETER_HIKARI2021.csv.zip"
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        parts.append(pd.read_csv(zf.open(name), low_memory=False).assign(_set="HIKARI-2021"))
    parts.append(pd.read_csv(BLIND2 / "hikari" / "ALLFLOWMETER_HIKARI2022.csv", low_memory=False).assign(_set="HIKARI-2022"))
    return pd.concat(parts, ignore_index=True)


def blind_flows() -> dict[str, Any]:
    from src.soc import engine
    from src.soc.columns import align
    from src.soc.model import Detector
    from src.soc.rcl import softmax

    backups = sorted((PROJECT_ROOT / "artifacts" / "deploy_backup").glob("*_pre_flow_stack"))
    dets = {"before (deployed until 2026-10-24)": Detector(backups[0]) if backups else None,
            "now (flow stack)": Detector(PROJECT_ROOT / "artifacts" / "deploy")}  # fmt: skip
    df = _hikari()
    y = pd.to_numeric(df["Label"], errors="coerce").fillna(0).to_numpy() > 0
    cat = df["attack_category"].astype(str).to_numpy()
    out: dict[str, Any] = {"dataset": "HIKARI-2021 + HIKARI-2022 (Zeek FlowMeter, converted)", "rows": len(df),
                           "by_category_rows": dict(Counter(cat))}  # fmt: skip
    for name, det in dets.items():
        if det is None:
            continue
        benign = det.label_names.index("BENIGN")
        aligned, info = align(df, det.feature_names)
        probs = []
        for i in range(0, len(aligned), 100_000):
            probs.append(softmax(engine.flow_logits(det, aligned.iloc[i : i + 100_000]), det.temperature))
        p = np.vstack(probs)
        att = 1.0 - p[:, benign]
        anyalert = att >= 0.5
        thr = float(det.alert_threshold_high) if det.alert_threshold_high is not None else 0.5
        confirmed = att >= thr
        out[name] = {"features_matched": info["matched"], "any_alert": _rates(y, anyalert),
                     "confirmed": _rates(y, confirmed), "confirmed_threshold": thr,
                     "confirmed_by_category": {c: round(float(confirmed[cat == c].mean()), 4) for c in sorted(set(cat))}}  # fmt: skip
    return out


# --- logs: AIT-LDS v2 ------------------------------------------------------------------


def _ait_server(zpath: Path) -> dict[str, Any]:
    from src.soc.correlate import correlate, entity_of
    from src.soc.parsers import assume_year, parse_line
    from src.soc.riskboard import board_for

    z = zipfile.ZipFile(zpath)
    names = set(z.namelist())
    labels = [n for n in names if n.startswith("labels/") and not n.endswith("/")]
    labelled_data = {"gather/" + n[len("labels/"):] for n in labels}
    logs = sorted(n for n in names if n.startswith("gather/") and "/logs/" in n and not n.endswith("/"))
    step_lines: dict[str, int] = Counter()
    step_hit: dict[str, int] = Counter()
    att_n = att_hit = norm_n = norm_hit = 0
    evs: list[dict[str, Any]] = []
    per_file = []
    with assume_year(2022):  # AIT-LDS v2 was recorded in January 2022 (dated logs say so)
        for data in logs:
            host = data.split("/")[1]
            lab: dict[int, list[str]] = {}
            if data in labelled_data:
                for ln in z.read("labels/" + data[len("gather/"):]).decode("utf-8", "replace").splitlines():
                    try:
                        r = json.loads(ln)
                        lab[int(r["line"])] = list(r.get("labels") or [])
                    except (ValueError, KeyError, TypeError):
                        continue
            every_att = max(1, len(lab) // ATTACK_CAP)
            n_norm = a0 = n0 = 0
            t0 = time.time()
            with z.open(data) as fh:
                for i, raw in enumerate(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), start=1):
                    if i > MAX_LINES:
                        break
                    is_att = i in lab
                    if is_att:
                        a0 += 1
                        if a0 % every_att:
                            continue
                    elif i % NORMAL_EVERY or n_norm >= NORMAL_CAP:
                        continue
                    ev = parse_line(raw)
                    if ev is None:
                        continue
                    ev.setdefault("observer", host)
                    hit = bool(ev.get("techniques"))
                    if is_att:
                        att_n += 1
                        att_hit += hit
                        for s in lab[i]:
                            step_lines[s] += 1
                            step_hit[s] += hit
                        ev["_attack"] = True
                    else:
                        n_norm += 1
                        n0 += 1
                        norm_n += 1
                        norm_hit += hit
                        ev["_attack"] = False
                    evs.append(ev)
            per_file.append({"file": data, "attack_lines": len(lab), "normal_sampled": n0, "seconds": round(time.time() - t0)})
    # incidents: entity truth = every entity with attack-labelled events
    actors = {entity_of(e) for e in evs if e["_attack"]}
    evs.sort(key=lambda e: e["ts"])
    inc: dict[str, Any] = {}
    for name, board in (("board_off", None), ("board_on", board_for(evs))):
        incs = correlate(evs, board=board)
        high = [i for i in incs if i["severity"] >= 6]
        pos = [i for i in incs if i["src_ip"] in actors]
        tp = sum(i["src_ip"] in actors for i in high)
        inc[name] = {"incidents": len(incs), "high": len(high), "actor_incidents": len(pos), "actor_high": tp,
                     "precision": tp / max(1, len(high)), "recall": tp / max(1, len(pos))}  # fmt: skip
    steps = {s: {"lines": step_lines[s], "alerted": step_hit[s]} for s in sorted(step_lines)}
    return {"steps": steps, "steps_caught": sum(1 for v in steps.values() if v["alerted"]), "n_steps": len(steps),
            "attack_lines": att_n, "attack_lines_alerted": att_hit, "line_recall": att_hit / max(1, att_n),
            "normal_sampled": norm_n, "normal_alerted": norm_hit, "false_alarm_rate": norm_hit / max(1, norm_n),
            "incidents": inc, "files": per_file}  # fmt: skip


def blind_logs() -> dict[str, Any]:
    out: dict[str, Any] = {"dataset": "AIT-LDS v2.0 (russellmitchell, santos)"}
    for name in ("russellmitchell", "santos"):
        p = BLIND2 / "ait-lds-v2" / f"{name}.zip"
        try:
            out[name] = _ait_server(p)
        except zipfile.BadZipFile:
            out[name] = {"error": "archive incomplete (download not finished)"}
    return out


# --- Windows: EVTX-to-MITRE-Attack -------------------------------------------------------


def blind_windows() -> dict[str, Any]:
    from src.knowledge.attack_details import tactics_of
    from src.soc import parsers

    root = BLIND2 / "evtx-to-mitre" / "repo"
    rows: list[dict[str, Any]] = []
    for f in sorted(root.glob("TA*/*/*.evtx")):
        tac_dir, tech_dir = f.parent.parent.name, f.parent.name
        m = re.match(r"(T\d{4})", tech_dir)
        if not m:
            continue
        truth_t = m.group(1)
        truth_tac = set(tactics_of(truth_t))
        parsers.reset_sessions()
        try:
            evs = parsers.read_evtx(f)
        except Exception as exc:  # noqa: BLE001 - recorded (unreadable / quarantined)
            rows.append({"file": f"{tac_dir}/{tech_dir}/{f.name}", "error": str(exc)[:120]})
            continue
        techs = {t.split(".")[0] for e in evs for t in e.get("techniques", [])}
        tacs = {x for t in techs for x in tactics_of(t)}
        rows.append({"file": f"{tac_dir}/{tech_dir}/{f.name}", "tactic_folder": tac_dir, "technique": truth_t,
                     "events": len(evs), "detected": bool(techs), "right_tactic": bool(tacs & truth_tac),
                     "right_technique": truth_t in techs})  # fmt: skip
    ok = [r for r in rows if "error" not in r]
    n = max(1, len(ok))
    by: dict[str, Counter[str]] = defaultdict(Counter)
    for r in ok:
        c = by[r["tactic_folder"]]
        c["files"] += 1
        for k in ("detected", "right_tactic", "right_technique"):
            c[k] += r[k]
    return {"dataset": "EVTX-to-MITRE-Attack (technique folders)", "files": len(ok), "unreadable": len(rows) - len(ok),
            "detected_rate": sum(r["detected"] for r in ok) / n, "right_tactic_rate": sum(r["right_tactic"] for r in ok) / n,
            "right_technique_rate": sum(r["right_technique"] for r in ok) / n,
            "by_tactic_folder": {k: dict(v) for k, v in sorted(by.items())}, "rows": rows}  # fmt: skip


# --- phishing: CompPhish ---------------------------------------------------------------------


def _xlsx_rows(path: Path) -> list[list[str]]:
    """First sheet of an .xlsx as rows of text (no Excel library needed)."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows = []
        for r in sheet.iter(f"{{{ns['m']}}}row"):
            vals = []
            for c in r.findall("m:c", ns):
                v = c.find("m:v", ns)
                txt = "" if v is None else (v.text or "")
                if c.get("t") == "s" and txt:
                    txt = shared[int(txt)]
                elif c.get("t") == "inlineStr":
                    txt = "".join(t.text or "" for t in c.iter(f"{{{ns['m']}}}t"))
                vals.append(txt)
            rows.append(vals)
    return rows


def blind_phishing() -> dict[str, Any]:
    import joblib

    from src.mlareas import phishing, phishing_page

    d = BLIND2 / "compphish"
    rows = _xlsx_rows(d / "Mapping_File.xlsx")
    head = [h.strip().lower() for h in rows[0]]
    found = {k: next((i for i, h in enumerate(head) if k in h), None) for k in ("serial", "url", "label")}
    idx: dict[str, int] = ({k: int(v) for k, v in found.items() if v is not None} if None not in found.values()
                           else {"serial": 0, "url": 1, "label": 2})  # fmt: skip
    items: list[tuple[str, str, int]] = []
    for r in rows[1:]:
        try:
            serial, url, lab = r[idx["serial"]], r[idx["url"]], r[idx["label"]]
        except IndexError:
            continue
        lab_s = str(lab).strip().lower()
        lab_y = 1 if lab_s in ("1", "phishing", "phish", "1.0") else 0 if lab_s in ("0", "legitimate", "legit", "benign", "0.0") else None
        if lab_y is not None and url:
            items.append((str(serial).split(".")[0], str(url).strip(), lab_y))
    out: dict[str, Any] = {"dataset": "CompPhish v4 (Mendeley fmbs4kp9wz, 2024-09 .. 2025-08)", "items": len(items),
                           "phishing": sum(y for *_, y in items), "mapping_columns": head}  # fmt: skip
    urls, y = [u for _, u, _ in items], np.array([v for *_, v in items], bool)
    prev = sorted((PROJECT_ROOT / "artifacts" / "mlareas" / "phishing").glob("prev_v4_*"))
    bundles = {"url v5 (now)": phishing.load(refresh=True)}
    if prev and (prev[0] / "model.joblib").is_file():
        bundles["url v4 (before)"] = joblib.load(prev[0] / "model.joblib")
    for name, b in bundles.items():
        res = phishing.check_urls(urls, b)
        out[name] = _rates(y, np.array([x.get("verdict") == "phishing" for x in res], bool))
    page_alert: list[bool] = []
    page_y: list[bool] = []
    with zipfile.ZipFile(d / "All_HTML.zip") as z:
        files = {Path(m).stem: m for m in z.namelist() if not m.endswith("/")}
        pb = phishing_page.load()
        for sid, page_url, is_phish in items:
            member = files.get(sid)
            if member is None:
                continue
            html = z.read(member).decode("utf-8", "replace")
            page_alert.append(phishing_page.check_page(html, page_url, pb)["verdict"] == "phishing")
            page_y.append(bool(is_phish))
    out["page model"] = _rates(np.array(page_y, bool), np.array(page_alert, bool)) | {"pages": len(page_y)}
    return out


TESTS = {"flows": blind_flows, "logs": blind_logs, "windows": blind_windows, "phishing": blind_phishing}


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", choices=list(TESTS), default=None)
    a = ap.parse_args()
    doc = json.loads(OUT.read_text("utf-8")) if OUT.is_file() else {}
    for name in [a.section] if a.section else list(TESTS):
        if name in doc and "error" not in json.dumps(doc[name])[:2000]:
            print(f"{name}: already scored once ({doc[name].get('when')}), not scored again")
            continue
        t0 = time.time()
        res = TESTS[name]()
        res["when"], res["seconds"] = time.strftime("%Y-%m-%d %H:%M"), round(time.time() - t0)
        doc[name] = res
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=1, default=float), "utf-8")
        print(name, json.dumps({k: v for k, v in res.items() if k not in ("rows", "files")}, default=float)[:1500])


if __name__ == "__main__":
    main()
