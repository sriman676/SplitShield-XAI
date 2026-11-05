"""Held-out tests for the non-Windows log areas (owner 2026-10-15).

Fair-test protocol, per area:
  * rules are written from public ATT&CK knowledge or a DEVELOPMENT split only;
  * scores come from held-out data the rules never saw;
  * labels that would give the answer away (e.g. the attack tool's own user-agent)
    are hidden from the detectors before scoring;
  * Splunk attack_data is the BLIND set for every log area and is scored once, at
    the very end (``--blind``), never during development.

    python -m scripts.extra_logs --area cloud
    python -m scripts.extra_logs --area zeek

Results: results/extra/logs_<area>.json (+ a section in EXTRA_DATASETS.md).
"""

from __future__ import annotations

import argparse
import gzip
import json
import tarfile
import time
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from utils.logging_config import get_logger
from utils.paths import RAW_DIR, RESULTS_DIR

_log = get_logger(__name__)
OUT = RESULTS_DIR / "extra"


# --- cloud: AWS CloudTrail --------------------------------------------------------------


def _invictus() -> Iterator[tuple[dict[str, Any], bool, str]]:
    """(record, is_attack, group). Attack = Stratus Red Team's own detonation calls
    (user-agent 'stratus-red-team_<id>'); its Terraform setup/teardown and all other
    activity are not attacks. The tell-tale user-agent is replaced before scoring."""
    z = zipfile.ZipFile(RAW_DIR / "invictus-aws" / "aws_dataset.zip")
    for n in sorted(z.namelist()):
        if not n.endswith(".json"):
            continue
        for r in json.loads(z.read(n)).get("Records", []):
            ua = str(r.get("userAgent", ""))
            attack = ua.startswith("stratus-red-team_")
            group = (
                "attack"
                if attack
                else "setup"
                if "stratus" in ua.lower()
                else "background"
            )
            if "stratus" in ua.lower():
                r = {**r, "userAgent": "aws-sdk-go/1.44.0 (go1.20; linux; amd64)"}
            yield r, attack, group


def _flaws() -> Iterator[dict[str, Any]]:
    with tarfile.open(RAW_DIR / "flaws-cloudtrail" / "flaws_cloudtrail_logs.tar") as t:
        for m in t.getmembers():
            if m.name.endswith(".json.gz") and (fh := t.extractfile(m)):
                yield from json.loads(gzip.decompress(fh.read())).get("Records", [])


def run_cloud() -> dict[str, Any]:
    from src.soc.logformats import parse_json_record

    t0 = time.time()
    rows = []
    for rec, attack, group in _invictus():
        ev = parse_json_record(rec)
        rows.append({"attack": attack, "group": group, "event": rec.get("eventName"),
                     "alert": bool(ev and ev["techniques"]),
                     "techniques": ev["techniques"] if ev else []})  # fmt: skip
    att = [r for r in rows if r["attack"]]
    caught_names = sorted({r["event"] for r in att if r["alert"]})
    missed_names = sorted(
        {r["event"] for r in att if not r["alert"]} - set(caught_names)
    )
    by_group = {
        g: {"events": sum(r["group"] == g for r in rows),
            "alerts": sum(r["group"] == g and r["alert"] for r in rows)}
        for g in ("attack", "setup", "background")
    }  # fmt: skip
    inv = {
        "events": len(rows),
        "attack_events": len(att),
        "attack_events_caught": sum(r["alert"] for r in att),
        "attack_api_calls_caught": caught_names,
        "attack_api_calls_missed": missed_names,
        "by_group": by_group,
        "false_alarm_rate_background": by_group["background"]["alerts"]
        / max(1, by_group["background"]["events"]),
        "false_alarm_rate_setup": by_group["setup"]["alerts"]
        / max(1, by_group["setup"]["events"]),
    }
    # flaws.cloud: real attackers, unlabelled -> volume and what fired
    n = alerts = 0
    kinds: Counter[str] = Counter()
    techs: Counter[str] = Counter()
    for rec in _flaws():
        n += 1
        ev = parse_json_record(rec)
        if ev and ev["techniques"]:
            alerts += 1
            kinds[
                ev["kind"]
                if " | Sigma: " not in ev["detail"]
                else ev["detail"].split(" | Sigma: ")[-1]
            ] += 1
            techs.update(ev["techniques"])
    flaws = {"events": n, "alerts": alerts, "alert_rate": alerts / max(1, n),
             "top_alerts": kinds.most_common(15), "top_techniques": techs.most_common(10),
             "events_per_second": n / max(1e-9, time.time() - t0)}  # fmt: skip
    return {"invictus": inv, "flaws_cloud": flaws}


# --- network: Zeek conn logs (IoT-23), machine learning split BY CAPTURE ---------------

# split BY MALWARE CAPTURE (family in brackets); test captures are never seen in
# training, validation captures choose the model and threshold
ZEEK_SPLIT = {
    "train": [
        "CTU-IoT-Malware-Capture-34-1",  # Mirai
        "CTU-IoT-Malware-Capture-1-1",  # Hide and Seek
        "CTU-IoT-Malware-Capture-8-1",  # Hakai
        "CTU-IoT-Malware-Capture-3-1",  # Muhstik
        "CTU-IoT-Malware-Capture-48-1",  # Mirai
        "CTU-IoT-Malware-Capture-20-1",  # Torii
        "CTU-Honeypot-Capture-4-1",  # benign
    ],
    "val": [
        "CTU-IoT-Malware-Capture-60-1",  # Gafgyt
        "CTU-IoT-Malware-Capture-42-1",  # Trojan
    ],
    "test": [
        "CTU-IoT-Malware-Capture-9-1",  # Hajime (family unseen in training)
        "CTU-IoT-Malware-Capture-49-1",  # Mirai
        "CTU-IoT-Malware-Capture-44-1",  # Mirai
        "CTU-IoT-Malware-Capture-21-1",  # Torii
        "CTU-Honeypot-Capture-5-1",  # benign
    ],
}
ZEEK_CAP = 100_000  # per class per capture (seeded reservoir sample of big captures)


def _zeek_rows(capture: str) -> list[dict[str, Any]]:
    """Stream one labelled conn.log; keep at most ZEEK_CAP rows per class, chosen by
    a seeded reservoir so big captures neither dominate nor run out of memory.
    Rows keep their time order (the rolling per-source context needs it)."""
    import random
    import re

    rng = random.Random(f"{capture}:0")
    path = RAW_DIR / "iot-23" / capture / "conn.log.labeled"
    fields: list[str] = []
    keep: dict[bool, list[tuple[int, dict[str, Any]]]] = {True: [], False: []}
    seen = {True: 0, False: 0}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, raw in enumerate(fh):
            raw = raw.rstrip("\n")
            if raw.startswith("#fields"):
                # IoT-23 joins the last header names with spaces too (`label   detailed-label`)
                fields = [
                    n
                    for c in raw.split("\t")[1:]
                    for n in re.split(r"\s{2,}", c.strip())
                ]
                continue
            if raw.startswith("#") or not raw.strip():
                continue
            vals = raw.split("\t")
            if len(vals) < len(fields):
                vals = vals[:-1] + re.split(r"\s{2,}", vals[-1])
            row: dict[str, Any] = dict(zip(fields, vals, strict=False))
            mal = str(row.get("label", "")).lower().startswith("malicious")
            row["_malicious"], row["_capture"] = mal, capture
            seen[mal] += 1
            bucket = keep[mal]
            if len(bucket) < ZEEK_CAP:
                bucket.append((i, row))
            else:
                j = rng.randrange(seen[mal])
                if j < ZEEK_CAP:
                    bucket[j] = (i, row)
    return [r for _, r in sorted(keep[True] + keep[False], key=lambda t: t[0])]


def run_zeek() -> dict[str, Any]:
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, precision_score, recall_score

    from src.soc import zeekml

    data = {}
    for split, caps in ZEEK_SPLIT.items():
        rows = [
            r for c in caps if (RAW_DIR / "iot-23" / c).is_dir() for r in _zeek_rows(c)
        ]
        X = np.vstack([zeekml.matrix([r for r in rows if r["_capture"] == c])
                       for c in caps if any(r["_capture"] == c for r in rows)])  # fmt: skip
        y = np.array(
            [r["_malicious"] for c in caps for r in rows if r["_capture"] == c],
            dtype=int,
        )
        data[split] = (X, y)

    def metrics(m: Any, split: str, thr: float = 0.5) -> dict[str, float]:
        X, y = data[split]
        p = (m.predict_proba(X)[:, 1] >= thr).astype(int)
        neg = y == 0
        return {"f1": float(f1_score(y, p, zero_division=0)),
                "precision": float(precision_score(y, p, zero_division=0)),
                "recall": float(recall_score(y, p, zero_division=0)),
                "false_alarm_rate": float(p[neg].mean()) if neg.any() else 0.0,
                "n": int(len(y)), "malicious": int(y.sum())}  # fmt: skip

    cands = {}
    try:
        from lightgbm import LGBMClassifier

        cands["lightgbm"] = lambda s: LGBMClassifier(n_estimators=300, learning_rate=0.05,
            num_leaves=31, class_weight="balanced", random_state=s, verbose=-1)  # fmt: skip
    except ImportError:
        pass
    cands["random_forest"] = lambda s: RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=s,
    )
    out: dict[str, Any] = {
        "split": ZEEK_SPLIT,
        "features": len(zeekml.FEATURES),
        "models": {},
    }
    best: tuple[float, str, Any, float] | None = None
    for name, make in cands.items():
        seeds: list[dict[str, Any]] = []
        for seed in (0, 1, 2):
            m = make(seed).fit(*data["train"])
            # threshold chosen on VALIDATION, then frozen for test: maximise
            # detection rate minus false-alarm rate (Youden's J). In IoT-23 the
            # benign connections come from the same infected devices, so a strict
            # 1 % alert budget leaves nothing detectable (recall 0, tried first).
            pv = m.predict_proba(data["val"][0])[:, 1]
            yv = data["val"][1]
            thrs = np.linspace(0.05, 0.99, 95)
            j = [(pv[yv == 1] >= t).mean() - (pv[yv == 0] >= t).mean() for t in thrs]
            thr = float(thrs[int(np.argmax(j))])
            t0 = time.time()
            m.predict_proba(data["test"][0])
            speed = len(data["test"][1]) / max(1e-9, time.time() - t0)
            seeds.append({"threshold": thr, "train": metrics(m, "train", thr),
                          "val": metrics(m, "val", thr), "test": metrics(m, "test", thr),
                          "conns_per_second": speed})  # fmt: skip
            score = float(seeds[-1]["val"]["f1"])
            if best is None or score > best[0]:
                best = (score, name, m, thr)
        out["models"][name] = seeds
    assert best is not None
    zeekml.MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best[2], "threshold": best[3], "name": best[1],
                 "features": zeekml.FEATURES}, zeekml.MODEL)  # fmt: skip
    out["chosen"] = {"model": best[1], "val_f1": best[0], "threshold": best[3]}
    return out


# --- web / Linux / mail / IDS: AIT Log Data Set v1.1 (4 real mail+web servers) ---------

AIT_HOSTS = {
    "dev": ["mail.cup.com", "mail.insect.com"],  # rules were written from these
    "heldout": ["mail.onion.com", "mail.spiral.com"],  # scored only
}
NORMAL_SAMPLE = 4000  # normal lines scored per log file (false-alarm rate)
STEP_CAP = 300  # attack lines scored per step and file (one alert catches a step)


def _scan_labels(lab: Path) -> tuple[dict[int, str], int, int]:
    """Attack line numbers -> step (capped per step), plus the last attack line and
    the number of lines; label files are small even when the log is 24 GB."""
    attacks: dict[int, str] = {}
    per_step: Counter[str] = Counter()
    last = n = 0
    with open(lab, encoding="utf-8", errors="replace") as fh:
        for n, raw in enumerate(fh, 1):
            if raw.startswith("0,0"):
                continue
            parts = raw.strip().split(",")
            step = parts[1] if len(parts) > 1 else "0"
            if (
                step == "0"
            ):  # inside an attack time window only: neither attack nor normal
                attacks[n - 1] = ""
                continue
            if step != "0":
                last = n
                per_step[step] += 1
                # beyond the cap: skipped (""), never counted as a normal line
                attacks[n - 1] = step if per_step[step] <= STEP_CAP else ""
    return attacks, last, n


def run_ait(which: str) -> dict[str, Any]:
    """Step-level score: an attack step on a server is CAUGHT when at least one line
    the dataset attributes to that step raises an alert. (AIT labels every line
    written during a step, including normal-looking background lines, so per-line
    recall is reported for information only.) Huge logs are read only up to just
    after their last attack line; normal lines are sampled evenly over that span."""
    from src.soc.parsers import parse_line

    base = RAW_DIR / "ait-lds-v1.1"
    per_file: list[dict[str, Any]] = []
    steps: dict[tuple[str, str], dict[str, Any]] = {}
    for host in AIT_HOSTS[which]:
        for lab in sorted((base / "labels" / host).rglob("*")):
            if lab.is_dir() or lab.suffix == ".zip":
                continue
            rel = lab.relative_to(base / "labels" / host)
            data = base / "data" / host / rel
            if not data.is_file():
                continue
            t0 = time.time()
            attacks, last, n_lines = _scan_labels(lab)
            span = min(n_lines, (last + 200_000) if last else 2_000_000)
            every = max(1, span // NORMAL_SAMPLE)
            att = att_hit = norm = norm_hit = 0
            kinds: Counter[str] = Counter()
            with open(data, encoding="utf-8", errors="replace") as df:
                for i, line in enumerate(df):
                    if i >= span:
                        break
                    step = attacks.get(i)
                    if step == "" or (step is None and i % every):
                        continue
                    ev = parse_line(line)
                    alert = bool(ev and ev["techniques"])
                    if step is None:
                        norm += 1
                        norm_hit += alert
                        continue
                    att += 1
                    att_hit += alert
                    st = steps.setdefault(
                        (host, step), {"lines": 0, "caught": False, "by": Counter()}
                    )
                    st["lines"] += 1
                    if alert and ev:
                        st["caught"] = True
                        st["by"][f"{rel.as_posix()}:{ev['kind']}"] += 1
                        kinds[ev["kind"]] += 1
            per_file.append({"host": host, "log": rel.as_posix(), "attack_lines_scored": att,
                             "attack_lines_alerted": att_hit, "normal_sampled": norm,
                             "normal_alerted": norm_hit,
                             "false_alarm_rate": norm_hit / max(1, norm),
                             "top_kinds": kinds.most_common(5),
                             "seconds": round(time.time() - t0)})  # fmt: skip
            print(f"{host} {rel}: attack {att_hit}/{att}  normal FP {norm_hit}/{norm}  "
                  f"({round(time.time() - t0)} s)", flush=True)  # fmt: skip
    step_rows = [{"host": h, "step": st, "lines_scored": v["lines"], "caught": v["caught"],
                  "by": v["by"].most_common(3)} for (h, st), v in sorted(steps.items())]  # fmt: skip
    norm_all = sum(f["normal_sampled"] for f in per_file)
    return {
        "hosts": AIT_HOSTS[which],
        "steps": step_rows,
        "steps_caught": sum(r["caught"] for r in step_rows),
        "steps_total": len(step_rows),
        "attack_line_recall": sum(f["attack_lines_alerted"] for f in per_file)
        / max(1, sum(f["attack_lines_scored"] for f in per_file)),
        "false_alarm_rate": sum(f["normal_alerted"] for f in per_file)
        / max(1, norm_all),
        "files": per_file,
    }


AREAS = {
    "cloud": run_cloud,
    "zeek": run_zeek,
    "ait-dev": lambda: run_ait("dev"),
    "ait-heldout": lambda: run_ait("heldout"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", choices=sorted(AREAS), required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = AREAS[a.area]()
    (OUT / f"logs_{a.area}.json").write_text(json.dumps(res, indent=1), "utf-8")
    print(json.dumps(res, indent=1)[:4000])


if __name__ == "__main__":
    main()
