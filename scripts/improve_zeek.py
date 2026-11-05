"""Improve the Zeek connection model (owner 2026-10-16): fewer false alarms.

Same IoT-23 capture split as the deployed model (scripts/extra_logs.py ZEEK_SPLIT:
test captures and malware families never seen in training). A small grid of
LightGBM / random-forest / extra-trees settings is fitted on the training captures;
model and threshold are chosen on the VALIDATION captures (highest F1 among settings
whose validation false-alarm rate is not above the champion's), then frozen.
Replace only if, on the TEST captures, precision AND recall are both not worse than
the deployed champion's (owner rule), and it is not more than 10 % slower.

    python -m scripts.improve_zeek [--no-save]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from typing import Any

import joblib
import numpy as np

from scripts.extra_logs import ZEEK_SPLIT, _zeek_rows
from src.soc import zeekml
from utils.paths import PROJECT_ROOT, RAW_DIR, RESULTS_DIR

OUT = RESULTS_DIR / "extra" / "improve_zeek.json"
SPEED_TOL = 0.10


def _data() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = {}
    for split, caps in ZEEK_SPLIT.items():
        rows = [
            r for c in caps if (RAW_DIR / "iot-23" / c).is_dir() for r in _zeek_rows(c)
        ]
        present = [c for c in caps if any(r["_capture"] == c for r in rows)]
        X = np.vstack(
            [zeekml.matrix([r for r in rows if r["_capture"] == c]) for c in present]
        )
        y = np.array(
            [r["_malicious"] for c in present for r in rows if r["_capture"] == c],
            dtype=int,
        )
        data[split] = (X, y)
    return data


def _metrics(p: np.ndarray, y: np.ndarray, thr: float) -> dict[str, float]:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-12, prec + rec),
            "false_alarm_rate": fp / max(1, int((y == 0).sum()))}  # fmt: skip


def _grid() -> list[tuple[str, dict[str, Any], Any]]:
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    out: list[tuple[str, dict[str, Any], Any]] = []
    for nl in (15, 63):
        for mcs in (20, 200):
            kw = {"num_leaves": nl, "min_child_samples": mcs, "n_estimators": 400,
                  "learning_rate": 0.05, "class_weight": "balanced", "subsample": 0.8,
                  "subsample_freq": 1, "colsample_bytree": 0.7}  # fmt: skip
            out.append(
                (
                    "lightgbm",
                    kw,
                    lambda s, kw=kw: LGBMClassifier(**kw, random_state=s, verbose=-1),
                )
            )
    for leaf in (2, 10):
        for mf in ("sqrt", 0.5):
            kw = {"n_estimators": 200, "min_samples_leaf": leaf, "max_features": mf,
                  "class_weight": "balanced"}  # fmt: skip
            out.append(
                (
                    "random_forest",
                    kw,
                    lambda s, kw=kw: RandomForestClassifier(
                        **kw, n_jobs=-1, random_state=s
                    ),
                )
            )
    kw = {"n_estimators": 300, "min_samples_leaf": 5, "class_weight": "balanced"}
    out.append(
        (
            "extra_trees",
            kw,
            lambda s, kw=kw: ExtraTreesClassifier(**kw, n_jobs=-1, random_state=s),
        )
    )
    return out


def _threshold(p: np.ndarray, y: np.ndarray, max_far: float) -> float | None:
    """Validation threshold: best F1 with false-alarm rate <= max_far (None if none)."""
    best: tuple[float, float] | None = None
    for t in np.linspace(0.05, 0.99, 95):
        m = _metrics(p, y, float(t))
        if m["false_alarm_rate"] <= max_far and (best is None or m["f1"] > best[0]):
            best = (m["f1"], float(t))
    return None if best is None else best[1]


def _speed(model: Any, X: np.ndarray) -> float:
    t0 = time.perf_counter()
    model.predict_proba(X[:50_000])
    return min(len(X), 50_000) / max(1e-9, time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    data = _data()
    champ = joblib.load(zeekml.MODEL)
    cm, cthr = champ["model"], float(champ["threshold"])
    pv_c = cm.predict_proba(data["val"][0])[:, 1]
    pt_c = cm.predict_proba(data["test"][0])[:, 1]
    c_val = _metrics(pv_c, data["val"][1], cthr)
    c_test = _metrics(pt_c, data["test"][1], cthr)
    rows: list[dict[str, Any]] = []
    best: tuple[float, str, dict[str, Any], Any, float] | None = None
    for name, kw, make in _grid():
        m = make(0).fit(*data["train"])
        pv = m.predict_proba(data["val"][0])[:, 1]
        thr = _threshold(pv, data["val"][1], c_val["false_alarm_rate"])
        if thr is None:
            rows.append({"model": name, "params": kw, "val": None})
            continue
        v = _metrics(pv, data["val"][1], thr)
        rows.append({"model": name, "params": kw, "threshold": thr, "val": v})
        print(
            f"{name} {kw} thr={thr:.2f} val {json.dumps({k: round(x, 4) for k, x in v.items()})}",
            flush=True,
        )
        if best is None or v["f1"] > best[0]:
            best = (v["f1"], name, kw, m, thr)
    doc: dict[str, Any] = {
        "what": "Zeek conn model (IoT-23): settings grid, chosen on validation captures",
        "rule": "best validation F1 with validation false-alarm rate <= champion's; replace only "
                "if TEST precision AND recall are not worse and not >10 % slower",
        "champion": {"name": champ.get("name"), "threshold": cthr, "val": c_val, "test": c_test},
        "candidates": rows, "blind_used": False,
    }  # fmt: skip
    decision = "kept champion: no setting met the validation false-alarm limit"
    if best is not None:
        _, name, kw, m, thr = best
        t_new = _metrics(m.predict_proba(data["test"][0])[:, 1], data["test"][1], thr)
        sp_new, sp_old = _speed(m, data["test"][0]), _speed(cm, data["test"][0])
        doc["challenger"] = {"model": name, "params": kw, "threshold": thr, "val": best and _metrics(
            m.predict_proba(data["val"][0])[:, 1], data["val"][1], thr), "test": t_new,
            "conns_per_s": sp_new, "champion_conns_per_s": sp_old}  # fmt: skip
        if (
            t_new["precision"] < c_test["precision"]
            or t_new["recall"] < c_test["recall"]
        ):
            decision = "kept champion: test precision or recall would drop"
        elif sp_new < sp_old * (1 - SPEED_TOL):
            decision = "kept champion: more than 10 % slower"
        else:
            decision = "promoted"
            if not a.no_save:
                backup = zeekml.MODEL.with_name(
                    f"model_backup_{time.strftime('%Y%m%d-%H%M')}.joblib"
                )
                shutil.copy2(zeekml.MODEL, backup)
                joblib.dump({"model": m, "threshold": thr, "name": name, "params": kw,
                             "features": zeekml.FEATURES}, zeekml.MODEL)  # fmt: skip
                doc["backup"] = str(backup.relative_to(PROJECT_ROOT))
    doc["decision"] = decision
    doc["seconds"] = round(time.time() - t0)
    OUT.write_text(json.dumps(doc, indent=1, default=str), "utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "champion_test": c_test,
                "challenger_test": doc.get("challenger", {}).get("test"),
            },
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
