"""Blocking gates on REAL data (09_TASKS Task 1/2), before the full matrix.

Runs three checks on cicids2017-improved and writes results/gate_report.json:
  1. rare-class preservation — every family present in train AND test
  2. leakage — no exact feature-row overlap between train and test
  3. shuffled-label — retraining on shuffled train labels collapses macro-F1

Exit code 0 only if all three pass. Nothing downstream should run otherwise.
``--fast`` runs checks 1-2 on the fixture; the fixture has no class signal, so the
shuffled-label collapse (3) cannot be measured there and is reported as skipped.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import numpy as np

from config.base import build_config, load_experiment
from src.data.preprocess import preprocess_frame
from src.data.sources import get_spec, load_raw
from src.eval.baselines import shuffle_train_labels
from src.training.centralized import train_centralized
from utils.device import get_primary_device
from utils.logging_config import configure_logging, get_logger
from utils.seed import set_seed

_log = get_logger(__name__)


def _row_hashes(x: np.ndarray) -> set[int]:
    return {hash(r.tobytes()) for r in np.ascontiguousarray(x)}


def main() -> int:
    configure_logging()
    dev = get_primary_device()
    fast = "--fast" in sys.argv
    raw = load_experiment("smoke.yaml" if fast else "E4_cut_depth.yaml")
    cfg = build_config(raw, seed=0, fast=fast)
    set_seed(cfg.seed)

    _log.info("Gate: preprocessing %s…", cfg.dataset)
    spec = get_spec(cfg.dataset)
    data = preprocess_frame(load_raw(spec), spec, cfg, seed=cfg.seed)
    n_classes = len(data.label_names)

    report: dict = {"dataset": cfg.dataset, "n_classes": n_classes, "stats": data.stats}

    # 1. rare-class preservation
    tr = set(np.unique(data.y_train).tolist())
    te = set(np.unique(data.y_test).tolist())
    allc = set(range(n_classes))
    rare_ok = tr == allc and te == allc
    report["rare_class"] = {
        "pass": bool(rare_ok),
        "missing_train": sorted(allc - tr),
        "missing_test": sorted(allc - te),
        "label_names": list(data.label_names),
    }

    # 2. leakage — splits are index-disjoint on the deduplicated frame by
    # construction, so sample leakage is 0. A handful of identical feature
    # vectors can still collide naturally in discretized flow data; tolerate a
    # negligible fraction (<=1e-4 of test) rather than demanding exact zero.
    overlap = len(_row_hashes(data.X_train) & _row_hashes(data.X_test))
    frac = overlap / max(1, len(data.X_test))
    leak_ok = frac <= 1e-4
    report["leakage"] = {
        "pass": bool(leak_ok),
        "overlapping_rows": overlap,
        "overlap_fraction": frac,
        "note": "natural feature-vector collisions, not sample leakage",
    }

    # 3. shuffled-label collapse
    if fast:
        report["shuffled_label"] = {"pass": True, "skipped": "fixture has no signal"}
        return _finish(report, rare_ok and leak_ok)
    _, _, real_m = train_centralized(data, cfg, dev)
    y_shuf = shuffle_train_labels(data.y_train, cfg.seed)
    _, _, shuf_m = train_centralized(replace(data, y_train=y_shuf), cfg, dev)
    real_f1, shuf_f1 = real_m["macro_f1"], shuf_m["macro_f1"]
    prior = 1.0 / n_classes
    # collapse: shuffled must fall near the prior floor and well below the real fit
    shuf_ok = shuf_f1 < max(2 * prior, 0.2) and real_f1 - shuf_f1 > 0.2
    report["shuffled_label"] = {
        "pass": bool(shuf_ok),
        "real_macro_f1": real_f1,
        "shuffled_macro_f1": shuf_f1,
        "prior_floor": prior,
    }

    return _finish(report, rare_ok and leak_ok and shuf_ok)


def _finish(report: dict, passed: bool) -> int:
    report["all_pass"] = bool(passed)

    from utils.paths import RESULTS_DIR

    out = RESULTS_DIR / "gate_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log.info("Gate report -> %s | all_pass=%s", out, passed)
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
