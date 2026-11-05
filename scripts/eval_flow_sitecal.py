"""Per-site flow calibration measured on HIKARI (owner 2026-10-24) - AFTER the blind round.

HIKARI was scored once in blind round 2 (results/extra/blind_round2.json, recorded
unchanged). The owner then allowed it as development data for this fix, so these
numbers are labelled NOT BLIND. Each HIKARI year is one "site": a random 30 % of its
flows (labels unused) is what the detector watches first; the threshold that keeps
alerts at 1 % there is applied to the other 70 % (labels used only to score).

    python -m scripts.eval_flow_sitecal
Writes results/extra/flow_sitecal_hikari.json.
"""

from __future__ import annotations

import json
import time

import numpy as np

from utils.paths import PROJECT_ROOT, RESULTS_DIR


def main() -> None:
    from scripts.blind_test_v2 import _hikari, _rates
    from src.soc import engine, sitecal
    from src.soc.columns import align
    from src.soc.model import Detector
    from src.soc.rcl import softmax

    det = Detector(PROJECT_ROOT / "artifacts" / "deploy")
    benign = det.label_names.index("BENIGN")
    base = float(det.alert_threshold_high or 0.5)
    df = _hikari()
    aligned, _ = align(df, det.feature_names)
    att = np.concatenate([1.0 - softmax(engine.flow_logits(det, aligned.iloc[i : i + 100_000]), det.temperature)[:, benign]
                          for i in range(0, len(aligned), 100_000)])  # fmt: skip
    y = df["Label"].fillna(0).astype(float).to_numpy() > 0
    out: dict = {"label": "NOT BLIND: HIKARI was scored once in blind round 2, then used to develop this fix",
                 "target_alert_share": sitecal.TARGET, "validated_threshold": base, "sites": {}}  # fmt: skip
    rng = np.random.default_rng(0)
    for site in sorted(df["_set"].unique()):
        idx = np.flatnonzero((df["_set"] == site).to_numpy())
        rng.shuffle(idx)
        cut = int(0.3 * len(idx))
        cal, test = idx[:cut], idx[cut:]
        thr = sitecal.threshold_for(att[cal], base)
        out["sites"][site] = {"calibration_flows": int(len(cal)), "test_flows": int(len(test)), "threshold": thr,
                              "validated": _rates(y[test], att[test] >= base),
                              "calibrated": _rates(y[test], att[test] >= thr)}  # fmt: skip
    out["when"] = time.strftime("%Y-%m-%d %H:%M")
    (RESULTS_DIR / "extra" / "flow_sitecal_hikari.json").write_text(json.dumps(out, indent=1, default=float), "utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
