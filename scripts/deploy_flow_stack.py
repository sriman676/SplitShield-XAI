"""Deploy the flow stack (OWNER-APPROVED EXCEPTION, 2026-10-24).

``python -m scripts.improve_flows_v2`` trained a multi-dataset generalist and a stack
over it and the deployed scorer. On the TEST splits the stack lifts attack recall on
networks the deployed detector never saw (CIC-IDS2018 0.37 -> 0.85, CIC IoT-DIAD 2024
0.14 -> 0.84) and lowers false alarms on every dataset, but loses a little recall on the
home datasets (CIC-IDS2017 0.99986 -> 0.99728, CIC-DDoS2019 0.99983 -> 0.99963), so
the precision-and-recall rule alone kept the champion. The owner approved it as an
exception. This script:

  1. builds the stack into a COPY of artifacts/deploy (temporary folder) and checks
     through the live code path (src/soc/model.Detector + engine.flow_logits, the
     detector's confirmed-alert threshold) that it reproduces the offline test numbers;
  2. with ``--apply``: backs up artifacts/deploy (artifacts/deploy_backup/<time>_
     pre_flow_stack), copies the two models in and records them in columns.json.

Roll back: copy the backup folder over artifacts/deploy.

    python -m scripts.deploy_flow_stack [--apply]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from scripts import improve_flows as imf
from scripts import improve_flows_v2 as v2
from utils.paths import PROJECT_ROOT

DEPLOY = PROJECT_ROOT / "artifacts" / "deploy"
OPT: dict[str, Any] = {"candidate": None, "results": "improve_flows_v2.json", "exception": None, "hikari": False}
EXCEPTION = ("owner-approved exception 2026-10-24: flow stack deployed although home recall drops "
             "(CIC-IDS2017 0.99986 -> 0.99728, CIC-DDoS2019 0.99983 -> 0.99963), for recall on new "
             "networks (CIC-IDS2018 0.37 -> 0.85, IoT-DIAD 0.14 -> 0.84) and fewer false alarms everywhere")  # fmt: skip


def _install(dst: Path) -> float:
    cand = Path(OPT["candidate"] or v2.CANDIDATE)
    s = joblib.load(cand / "stack.joblib")
    shutil.copy2(cand / "generalist.joblib", dst / "flow_generalist.joblib")
    shutil.copy2(cand / "stack.joblib", dst / "flow_stack.joblib")
    cols = json.loads((dst / "columns.json").read_text("utf-8"))
    thr = float(s["threshold_stack"])
    cols["flow_stack"] = {"generalist_file": "flow_generalist.joblib", "stack_file": "flow_stack.joblib",
                          "threshold": thr, "trained_on": ["cicids2017-original", "cic-ddos2019 (03-11)",
                                                           "cic-ids2018", "cic-iot-diad-2024"],
                          "exception": OPT["exception"] or EXCEPTION, "added": time.strftime("%Y-%m-%d %H:%M")}  # fmt: skip
    if OPT["hikari"]:
        cols["flow_stack"]["trained_on"].append("hikari (owner-released after blind round 2)")
    (dst / "columns.json").write_text(json.dumps(cols, indent=2), "utf-8")
    return thr


def check(cache: Path) -> dict[str, Any]:
    from src.soc import engine
    from src.soc.model import Detector, StackCascade
    from src.soc.rcl import softmax

    champ = Detector(DEPLOY)
    names = list(champ.label_names)
    benign = names.index("BENIGN")
    data = {"cic-ids2017": imf.cic17(champ), "cic-ddos2019": imf.ddos19(champ),
            "cic-ids2018": v2.cic18(champ, cache, v2.PER_LABEL), "cic-iot-diad-2024": v2.diad(champ, cache, v2.PER_LABEL)}  # fmt: skip
    if OPT["hikari"]:
        data["hikari"] = v2.hikari(champ, cache)
    offline = json.loads((PROJECT_ROOT / "results" / "extra" / OPT["results"]).read_text("utf-8"))
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "deploy"
        shutil.copytree(DEPLOY, dst)
        thr = _install(dst)
        det = Detector(dst)
        assert isinstance(det.ensemble, StackCascade)
        for ds, d in data.items():
            df = pd.DataFrame(d["X_test"], columns=det.feature_names)
            t0 = time.perf_counter()
            p = softmax(engine.flow_logits(det, df), det.temperature)
            secs = time.perf_counter() - t0
            alert = (1.0 - p[:, benign]) >= thr
            y = d["y_test"] != benign
            tp, fp = int((alert & y).sum()), int((alert & ~y).sum())
            live = {"precision": tp / max(1, tp + fp), "recall": tp / max(1, int(y.sum())),
                    "false_alarm_rate": fp / max(1, int((~y).sum()))}  # fmt: skip
            off = offline["rows"]["stack"]["test"][ds]
            out[ds] = {"live": live, "offline": {k: off[k] for k in live},
                       "max_abs_diff": max(abs(live[k] - off[k]) for k in live),
                       "flows_per_s": round(len(df) / max(secs, 1e-9))}  # fmt: skip
    out["threshold"] = thr
    out["reproduces_offline"] = all(v["max_abs_diff"] < 2e-3 for k, v in out.items() if isinstance(v, dict))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--cache", type=Path, default=Path("D:/SplitShield-data/cache/flows_v2"))
    ap.add_argument("--candidate", default=None, help="candidate folder (default: flow_generalist)")
    ap.add_argument("--results", default="improve_flows_v2.json")
    ap.add_argument("--exception", default=None, help="owner exception text recorded with the deployment")
    ap.add_argument("--hikari", action="store_true")
    a = ap.parse_args()
    OPT.update(candidate=a.candidate, results=a.results, exception=a.exception, hikari=a.hikari)
    res = check(a.cache)
    print(json.dumps(res, indent=1, default=float))
    if not a.apply:
        return
    if not res["reproduces_offline"]:
        raise SystemExit("live path does not reproduce the offline numbers: not applied")
    bak = PROJECT_ROOT / "artifacts" / "deploy_backup" / f"{time.strftime('%Y%m%d-%H%M')}_pre_flow_stack"
    shutil.copytree(DEPLOY, bak)
    _install(DEPLOY)
    doc_p = PROJECT_ROOT / "results" / "extra" / OPT["results"]
    doc = json.loads(doc_p.read_text("utf-8"))
    doc["deployed"] = {"variant": "stack", "exception": OPT["exception"] or EXCEPTION, "live_check": res, "backup": str(bak),
                       "when": time.strftime("%Y-%m-%d %H:%M")}  # fmt: skip
    doc_p.write_text(json.dumps(doc, indent=1, default=float), "utf-8")
    print("deployed; backup at", bak)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
