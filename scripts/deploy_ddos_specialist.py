"""Deploy the CIC-DDoS2019 specialist as an OWNER-APPROVED EXCEPTION (2026-10-17).

``python -m scripts.improve_flows cascade`` trained the specialist and showed that
champion OR specialist lifts CIC-DDoS2019 attack recall from 0.0003 to 0.9998 but
costs a little on CIC-IDS2017 test (false alarms 0.178 % -> 0.267 %, precision
0.99945 -> 0.99917) and ~14 % speed - so the precision rule alone kept the champion.
The owner approved it as an exception. This script:

  1. builds the cascade from a COPY of artifacts/deploy (temporary folder) and checks
     through the live code path (src/soc/model.Detector + engine.flow_logits) that it
     reproduces the offline numbers and changes nothing on flows the specialist
     does not fire on;
  2. with ``--apply``: backs up artifacts/deploy (artifacts/deploy_backup/<time>_
     pre_ddos_specialist), copies the specialist in, records it in columns.json, and
     writes the exception with before/after numbers to results/extra/improve_flows.json.

Roll back: copy the backup folder over artifacts/deploy.

    python -m scripts.deploy_ddos_specialist [--apply]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts import improve_flows as imf
from utils.paths import PROJECT_ROOT

DEPLOY = PROJECT_ROOT / "artifacts" / "deploy"
CANDIDATE = imf.CANDIDATES / "specialist.joblib"
EXCEPTION = ("owner-approved exception 2026-10-17: DDoS specialist added although "
             "CIC-IDS2017 test precision drops 0.99945 -> 0.99917 and false alarms rise "
             "0.178 % -> 0.267 %, for CIC-DDoS2019 attack recall 0.0003 -> 0.9998")  # fmt: skip


def _install(dst: Path, threshold: float) -> None:
    shutil.copy2(CANDIDATE, dst / "ddos_specialist.joblib")
    cols = json.loads((dst / "columns.json").read_text("utf-8"))
    cols["ddos_specialist"] = {"file": "ddos_specialist.joblib", "threshold": threshold,
                               "trained_on": ["cicids2017-original", "cic-ddos2019 (03-11)"],
                               "exception": EXCEPTION, "added": time.strftime("%Y-%m-%d %H:%M")}  # fmt: skip
    (dst / "columns.json").write_text(json.dumps(cols, indent=2), "utf-8")


def _metrics(pred_attack: np.ndarray, y_attack: np.ndarray) -> dict[str, float]:
    tp = int((pred_attack & y_attack).sum())
    fp = int((pred_attack & ~y_attack).sum())
    fn = int((~pred_attack & y_attack).sum())
    return {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "false_alarm_rate": fp / max(1, int((~y_attack).sum()))}  # fmt: skip


def check(threshold: float) -> dict[str, Any]:
    from src.soc import engine
    from src.soc.model import Detector, SpecialistCascade

    champ = Detector(DEPLOY)
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "deploy"
        shutil.copytree(DEPLOY, dst)
        _install(dst, threshold)
        casc = Detector(dst)
        assert isinstance(casc.ensemble, SpecialistCascade)
        names = list(champ.label_names)
        benign = names.index("BENIGN")
        out: dict[str, Any] = {}
        for ds, d in (("cic-ids2017", imf.cic17(champ)), ("cic-ddos2019", imf.ddos19(champ))):
            df = pd.DataFrame(d["X_test"], columns=champ.feature_names)
            y_att = d["y_test"] != benign
            t0 = time.perf_counter()
            lc = engine.flow_logits(champ, df)
            tc = time.perf_counter() - t0
            t0 = time.perf_counter()
            ln = engine.flow_logits(casc, df)
            tn = time.perf_counter() - t0
            fired = np.asarray(casc.ensemble.last_specialist)
            same = bool(np.array_equal(lc[~fired], ln[~fired]))
            pc, pn = lc.argmax(1) != benign, ln.argmax(1) != benign
            out[ds] = {"champion": _metrics(pc, y_att), "cascade": _metrics(pn, y_att),
                       "specialist_fired": int(fired.sum()), "other_rows_unchanged": same,
                       "flows_per_s": {"champion": round(len(df) / tc), "cascade": round(len(df) / tn)}}  # fmt: skip
            print(ds, json.dumps(out[ds]), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    import joblib

    threshold = float(joblib.load(CANDIDATE)["threshold"])
    res = check(threshold)
    ok = all(r["other_rows_unchanged"] for r in res.values())
    if not ok:
        raise SystemExit("cascade changed rows the specialist did not fire on - not deployed")
    p = PROJECT_ROOT / "results" / "extra" / "improve_flows.json"
    doc = json.loads(p.read_text("utf-8"))
    doc["cascade"]["live_path_check"] = res
    if a.apply:
        backup = PROJECT_ROOT / "artifacts" / "deploy_backup" / f"{time.strftime('%Y%m%d-%H%M')}_pre_ddos_specialist"
        shutil.copytree(DEPLOY, backup / "deploy")
        _install(DEPLOY, threshold)
        doc["cascade"]["decision"] = "promoted as an owner-approved exception (2026-10-17)"
        doc["cascade"]["exception"] = EXCEPTION
        doc["cascade"]["backup"] = str(backup)
        print("deployed; backup:", backup)
    p.write_text(json.dumps(doc, indent=1, default=str), "utf-8")


if __name__ == "__main__":
    main()
