"""Blind final tests (owner 2026-10-15): every frozen model, scored ONCE on a whole
dataset it never saw and nothing was tuned on. Run at the very end.

    python -m scripts.blind_test [--only flows,malware,ics,insider,phishing,logs]

Data under data/blind/ (downloaded unopened):
  flows    CIC-IDS2018 (8 days, CICFlowMeter)  -> deployed flow detector
  malware  EMBER 2017 v2 features              -> malware model
  ics      HAI 21.03                           -> ICS model
  insider  CMU CERT r4.1                       -> insider model
  phishing OpenPhish feed (today) + Tranco top sites (legitimate) -> phishing model
  logs     Splunk attack_data sample (labelled by ATT&CK technique) -> parsers + rules + Sigma
Results: results/extra/blind_tests.json + BLIND_TESTS.md.
"""

from __future__ import annotations

import argparse
import json
import random
import tarfile
import time
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from utils.paths import PROJECT_ROOT, RESULTS_DIR

BLIND = PROJECT_ROOT / "data" / "blind"
OUT = RESULTS_DIR / "extra"
PER_LABEL = 40_000  # rows per label and day (memory); recorded as a deviation


def _rates(y: np.ndarray, alert: np.ndarray) -> dict[str, Any]:
    tp, fn = int((y & alert).sum()), int((y & ~alert).sum())
    fp, tn = int((~y & alert).sum()), int((~y & ~alert).sum())
    return {"attacks": tp + fn, "caught": tp, "normal": fp + tn, "false_alarms": fp,
            "detection_rate": tp / max(1, tp + fn), "false_alarm_rate": fp / max(1, fp + tn),
            "precision": tp / max(1, tp + fp)}  # fmt: skip


def blind_flows() -> dict[str, Any]:
    """Deployed CICFlowMeter detector on CIC-IDS2018 (2018 column names matched)."""
    from src.soc import engine
    from src.soc.columns import align

    det = engine.load_detector()
    if det is None:
        raise RuntimeError("deployed detector missing")
    rng = np.random.default_rng(0)
    per_family: dict[str, list[int]] = {}
    ys, alerts = [], []
    for f in sorted((BLIND / "cic-ids2018").glob("*.csv")):
        parts = []
        for chunk in pd.read_csv(f, chunksize=300_000, low_memory=False):
            chunk = chunk[
                chunk["Label"] != "Label"
            ]  # repeated header rows in the 2018 CSVs
            for _lab, g in chunk.groupby("Label"):
                keep = min(len(g), max(1, PER_LABEL // 10))
                parts.append(g.iloc[rng.choice(len(g), keep, replace=False)])
        df = pd.concat(parts, ignore_index=True)
        df = df.groupby("Label", group_keys=False).apply(lambda g: g.head(PER_LABEL))
        aligned, cols = align(df, det.feature_names)
        logits = engine.flow_logits(det, aligned)
        pred = np.asarray(logits).argmax(1)
        benign = det.label_names.index("BENIGN")
        alert = pred != benign
        y = (df["Label"].astype(str).str.upper() != "BENIGN").to_numpy()
        ys.append(y)
        alerts.append(alert)
        for lab in df["Label"].unique():
            m = (df["Label"] == lab).to_numpy()
            per_family.setdefault(str(lab), [0, 0])
            per_family[str(lab)][0] += int(alert[m].sum())
            per_family[str(lab)][1] += int(m.sum())
    y, a = np.concatenate(ys), np.concatenate(alerts)
    return {"dataset": "CIC-IDS2018 (8 days)", "model": "deployed CICFlowMeter ensemble",
            **_rates(y, a), "by_label": {k: {"alerted": v[0], "rows": v[1], "rate": v[0] / v[1]}
                                         for k, v in sorted(per_family.items())},
            "note": f"<= {PER_LABEL} rows per label and day, seeded"}  # fmt: skip


def blind_malware() -> dict[str, Any]:
    from src.mlareas.malware import blind_test

    d = BLIND / "ember2017"
    tar = d / "ember_dataset_2017_2.tar.bz2"
    if not list(d.rglob("test_features.jsonl")):
        with tarfile.open(tar) as t:
            members = [
                m
                for m in t.getmembers()
                if m.name.endswith(".jsonl") and "test" in m.name
            ]
            t.extractall(d, members=members)  # noqa: S202 - names filtered to .jsonl
    files = sorted(str(p) for p in d.rglob("test_features*.jsonl"))
    # files whose sha256 also occurs in the EMBER 2018 data the model was built
    # from are left out, so every scored file is truly unseen
    overlap = set(json.loads((OUT / "ember2017_overlap_sha256.json").read_text("utf-8")))
    return {"dataset": "EMBER 2017 v2 (test features, files also in EMBER 2018 removed)",
            **blind_test(files, exclude_sha256=overlap)}  # fmt: skip


def blind_ics() -> dict[str, Any]:
    from src.mlareas.ics import blind_test

    return {"dataset": "HAI 21.03", **blind_test(BLIND / "hai-21.03")}


def blind_insider() -> dict[str, Any]:
    from src.mlareas.insider import blind_test

    d = BLIND / "cert-r4.1"
    if not (d / "r4.1").is_dir():
        with tarfile.open(d / "r4.1.tar.bz2") as t:
            t.extractall(d)  # noqa: S202 - CMU's own archive
    answers = PROJECT_ROOT / "data" / "raw" / "cert" / "answers"  # CMU key, all releases
    return {"dataset": "CMU CERT r4.1 (3 insiders)",
            **blind_test(d / "r4.1", answers_dir=answers if answers.is_dir() else None)}  # fmt: skip


def blind_phishing() -> dict[str, Any]:
    import urllib.request

    from src.mlareas.phishing import blind_test

    ua = {"User-Agent": "splitshield-blind"}
    phish = urllib.request.urlopen(urllib.request.Request("https://openphish.com/feed.txt", headers=ua),
                                   timeout=60).read().decode().split()  # fmt: skip
    # legitimate sites: ranks 1-20,000 of the SAVED Tranco snapshot (2026-10-15), the list the
    # blind pool was defined on - today's list would bring validation sites (ranks 20,001-
    # 100,000 on that day) into the top 20,000 (owner decision 2026-10-23)
    from scripts.improve_phishing import tranco

    snap = tranco()
    top = snap[snap["rank"] <= 20_000]
    legit = [f"https://{d}" for d in top["domain"].astype(str)]
    random.Random(0).shuffle(legit)
    legit = legit[: max(len(phish) * 3, 3000)]
    urls, labels = phish + legit, [1] * len(phish) + [0] * len(legit)
    return {"dataset": f"OpenPhish feed ({time.strftime('%Y-%m-%d')}) + Tranco top-20k sample (snapshot 2026-10-15)",
            "phishing": len(phish), "legitimate": len(legit), **blind_test(urls, labels)}  # fmt: skip


def blind_logs() -> dict[str, Any]:
    from src.knowledge.attack_details import tactics_of
    from src.soc.parsers import parse_text

    d = BLIND / "splunk"
    manifest = json.loads((d / "manifest.json").read_text("utf-8"))
    rows = []
    unreadable: list[str] = []
    for m in manifest:
        f = d / str(m.get("file", ""))
        if "error" in m or "skipped" in m or not f.is_file():
            continue
        truth_t = sorted({t.split(".")[0] for t in m["techniques"]})
        truth_tac = sorted({x for t in truth_t for x in tactics_of(t)})
        try:
            evs = parse_text(f.read_text("utf-8", errors="replace"), f.name)
        except OSError as exc:  # e.g. locked / quarantined by Windows Defender
            unreadable.append(f"{f.name}: {exc.strerror}")
            continue
        techs = sorted({t.split(".")[0] for e in evs for t in e.get("techniques", [])})
        tacs = sorted({x for t in techs for x in tactics_of(t)})
        rows.append({"file": f.name, "sourcetype": m["sourcetype"], "events": len(evs),
                     "detected": bool(techs), "right_tactic": bool(set(tacs) & set(truth_tac)),
                     "right_technique": bool(set(techs) & set(truth_t))})  # fmt: skip
    by: dict[str, Counter[str]] = {}
    for r in rows:
        c = by.setdefault(r["sourcetype"], Counter())
        c["files"] += 1
        c["detected"] += r["detected"]
        c["right_tactic"] += r["right_tactic"]
        c["right_technique"] += r["right_technique"]
    n = max(1, len(rows))
    return {"dataset": "Splunk attack_data (seeded sample)", "files": len(rows),
            "detected_rate": sum(r["detected"] for r in rows) / n,
            "right_tactic_rate": sum(r["right_tactic"] for r in rows) / n,
            "right_technique_rate": sum(r["right_technique"] for r in rows) / n,
            "by_sourcetype": {k: dict(v) for k, v in sorted(by.items())}, "rows": rows,
            "unreadable": unreadable}  # fmt: skip


TESTS = {"flows": blind_flows, "malware": blind_malware, "ics": blind_ics,
         "insider": blind_insider, "phishing": blind_phishing, "logs": blind_logs}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(TESTS))
    ap.add_argument("--store", default="blind_tests.json",
                    help="result file in results/extra (the final round: blind_final.json; "
                         "the 2026-10-15 round stays in blind_tests.json)")  # fmt: skip
    a = ap.parse_args()
    store = OUT / a.store
    for name in a.only.split(","):
        t0 = time.time()
        try:
            out = TESTS[name]() | {"seconds": round(time.time() - t0)}
        except Exception as exc:  # noqa: BLE001 - report, never hide, a failed blind test
            out = {"error": f"{type(exc).__name__}: {exc}"[:400]}
        # re-read just before writing: several blind tests may run side by side
        res = json.loads(store.read_text("utf-8")) if store.is_file() else {}
        res[name] = out
        store.write_text(json.dumps(res, indent=1, default=str), "utf-8")
        print(name, json.dumps({k: v for k, v in res[name].items() if k not in ("rows", "by_label")},
                               default=str)[:600], flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
