"""Fit the ICS model to ONE plant from its own normal data (owner 2026-10-24).

The deployed ICS model (HAI 22.04 + 23.05) does not transfer to another plant set-up:
on HAI 21.03 it alarmed nonstop (blind round 2026-10-24). Industrial plants differ in
sensors, ranges and control loops, so the product fits a model per plant instead:

  1. the plant's normal operation (no attack labels needed) is split in time order:
     first 70 % train, next 15 % per-sensor scaling, last 15 % threshold;
  2. the same autoencoder as the deployed model is trained on the train part;
  3. threshold = the most sensitive one with at most 5 false alarm events per day on
     the last normal part (the owner's alarm budget);
  4. the plant's labelled attack files (if any) are scored once.

Used here on plants the deployed model never saw: HAI 21.03 (owner-released old blind
set) and BATADAL (a water network), with a learning curve over how many days of normal
data the plant gives. SWaT / WADI can be added when the owner receives them.

    python -m scripts.ics_site_fit hai2103 [--days 1 3 7 0] [--seeds 0 1 2]
    python -m scripts.ics_site_fit batadal
    python -m scripts.ics_site_fit fit --normal DIR --name PLANT [--attack DIR]   (product use)
Results: results/extra/ics_site_fit.json; a fitted plant bundle goes to
artifacts/mlareas/ics/sites/<name>/ (``fit``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.mlareas import ics
from utils.logging_config import configure_logging, get_logger
from utils.paths import PROJECT_ROOT, RESULTS_DIR

_log = get_logger(__name__)
H2103 = PROJECT_ROOT / "data" / "blind" / "hai-21.03"  # used blind set, owner-released 2026-10-24
BATADAL = Path("D:/SplitShield-data/ics/batadal")
META = ("timestamp", "time", "datetime", "attack", "label", "att_flag")


def _meta(c: str) -> bool:
    s = c.strip().lower()
    return s in META or s.startswith(("attack_", "attack"))


def _labels(df: pd.DataFrame) -> np.ndarray:
    for c in df.columns:
        if c.strip().lower() in ("attack", "att_flag", "label"):
            return (pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy() > 0).astype(np.int8)
    return np.zeros(len(df), np.int8)


def _sensors(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if not _meta(c)]].apply(pd.to_numeric, errors="coerce").ffill().fillna(0.0)


def _time_split(frames: list[pd.DataFrame], cuts: tuple[float, ...]) -> list[list[pd.DataFrame]]:
    """Split a time-ordered list of frames at the given shares of total rows."""
    total = sum(len(f) for f in frames)
    edges = [0, *[int(c * total) for c in cuts], total]
    parts: list[list[pd.DataFrame]] = [[] for _ in range(len(edges) - 1)]
    start = 0
    for f in frames:
        a, b = start, start + len(f)
        for i in range(len(edges) - 1):
            lo, hi = max(a, edges[i]), min(b, edges[i + 1])
            if hi - lo > ics.LOOKBACK:
                parts[i].append(f.iloc[lo - a : hi - a])
        start = b
    return parts


def fit_site(normal: list[pd.DataFrame], seed: int = 0, epochs: int = 5,
             rows_per_day: int = 86_400) -> dict[str, Any]:  # fmt: skip
    """Train + calibrate on one plant's normal data; returns a scoring bundle."""
    ics.torch.set_num_threads(4)
    sens = [_sensors(f) for f in normal]
    tr, sc, th = _time_split(sens, (0.70, 0.85))
    prep = ics.fit_prep(pd.concat(tr, ignore_index=True))

    def part(frames: list[pd.DataFrame]) -> tuple[np.ndarray, list[int]]:
        return np.vstack([prep.transform(f) for f in frames]), [len(f) for f in frames]

    a_tr, l_tr = part(tr)
    det = ics.fit_detector("autoencoder", a_tr, ics.valid_index(l_tr), seed, epochs=epochs,
                           per_epoch=min(200_000, len(a_tr)))  # fmt: skip
    a_sc, l_sc = part(sc)
    det = ics.rescale_per_sensor(det, a_sc, ics.valid_index(l_sc), 0.99, seed)
    a_th, l_th = part(th)
    clean = ics.full_scores(det, a_th, l_th)
    days = max(len(clean) / rows_per_day, 1e-9)
    thr = float(np.max(clean)) + 1e-9
    for q in (90.0, 95.0, 97.0, 98.0, 99.0, 99.5, 99.9, 99.95, 99.99, 99.995, 99.999, 100.0):
        t = float(np.percentile(clean, q))
        if len(ics.alarm_events((clean > t).astype(int))) / days <= ics.ALARM_BUDGET:
            thr = t
            break
    return {"area": ics.AREA, "family": det.kind, "seed": seed, "version": 1, "model": ics._export_model(det),
            "prep": prep, "norm": det.norm, "mean": det.mean, "threshold": thr,
            "site_fit": {"normal_rows": int(sum(len(f) for f in normal)), "threshold_rows": int(len(clean)),
                         "false_alarms_per_day_on_threshold_part":
                             round(len(ics.alarm_events((clean > thr).astype(int))) / days, 2)}}  # fmt: skip


def score_site(b: dict[str, Any], attack: list[pd.DataFrame], rows_per_day: int = 86_400) -> dict[str, Any]:
    prep = b["prep"]
    frames, ys = [], []
    for f in attack:
        s = _sensors(f)
        for i, c in enumerate(prep.columns):
            if c not in s.columns:
                s[c] = prep.lo[i] + prep.span[i] / 2
        frames.append(prep.transform(s))
        ys.append(_labels(f))
    arr, y = np.vstack(frames), np.concatenate(ys)
    sc = ics.bundle_scores(b, arr, [len(f) for f in frames])
    alarm = (sc > b["threshold"]).astype(int)
    ev = ics.event_metrics(alarm, y)
    if rows_per_day != 86_400 and "false_alarms_per_day" in ev:  # hourly data (BATADAL)
        ev["false_alarms_per_day"] = ev["false_alarms_per_day"] * rows_per_day / 86_400
    keys = ("precision", "recall", "f1")
    pm = ics.point_metrics(alarm, y)
    return {"point": {k: round(float(pm[k]), 4) for k in keys if k in pm},
            "events": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()}}  # fmt: skip


def _read_gz(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def hai2103(days_list: list[int], seeds: list[int]) -> dict[str, Any]:
    normal = [_read_gz(p) for p in sorted(H2103.glob("train*.csv.gz"))]
    attack = [_read_gz(p) for p in sorted(H2103.glob("test*.csv.gz"))]
    total_days = sum(len(f) for f in normal) / 86_400
    out: dict[str, Any] = {"plant": "HAI 21.03 (another set-up of the HAI testbed)",
                           "normal_days_available": round(total_days, 1), "curve": []}  # fmt: skip
    for days in days_list:
        n_rows = int(days * 86_400) if days else None
        use: list[pd.DataFrame] = []
        left = n_rows if n_rows else sum(len(f) for f in normal)
        for f in normal:
            if left <= 0:
                break
            use.append(f.iloc[:left])
            left -= len(use[-1])
        per_seed = []
        for s in seeds:
            t0 = time.time()
            b = fit_site(use, seed=s)
            r = score_site(b, attack)
            r |= {"seed": s, "seconds": round(time.time() - t0), "threshold": b["threshold"], **b["site_fit"]}
            per_seed.append(r)
            _log.info("HAI 21.03 site fit, %s days, seed %s: %s", days or "all", s, r["events"])
        out["curve"].append({"normal_days": days or round(total_days, 1), "seeds": per_seed})
    return out


def batadal(seeds: list[int]) -> dict[str, Any]:
    normal = [pd.read_csv(BATADAL / "train_dataset.csv")]
    attack = [pd.read_csv(BATADAL / f) for f in ("test_dataset_1.csv", "test_dataset_2.csv")]
    for f in attack:
        f.columns = [c.strip() for c in f.columns]
        if "ATT_FLAG" in f.columns:
            f["ATT_FLAG"] = (pd.to_numeric(f["ATT_FLAG"], errors="coerce").fillna(0) > 0).astype(int)
    per_seed = []
    for s in seeds:
        b = fit_site(normal, seed=s, rows_per_day=24)
        r = score_site(b, attack, rows_per_day=24)
        per_seed.append(r | {"seed": s, "threshold": b["threshold"]})
    return {"plant": "BATADAL (water distribution network, hourly)", "seeds": per_seed,
            "note": "hourly data: LOOKBACK / smoothing are in rows (hours) here"}  # fmt: skip


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("hai2103", "batadal", "fit"))
    ap.add_argument("--days", type=int, nargs="+", default=[1, 3, 7, 0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--normal", type=Path)
    ap.add_argument("--attack", type=Path)
    ap.add_argument("--name", default="plant")
    ap.add_argument("--activate", action="store_true", help="use this plant model from now on (fit)")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR / "extra")
    a = ap.parse_args()
    if a.what == "fit":
        import joblib

        if a.normal is None:
            raise SystemExit("--normal DIR (CSV files of normal operation) is needed")
        normal = [pd.read_csv(p) for p in sorted(a.normal.glob("*.csv*"))]
        b = fit_site(normal, seed=a.seeds[0])
        res = {"site": a.name, **b["site_fit"], "threshold": b["threshold"]}
        if a.attack:
            res |= score_site(b, [pd.read_csv(p) for p in sorted(a.attack.glob("*.csv*"))])
        import re

        name = re.sub(r"[^\w.-]+", "_", a.name)[:60] or "plant"
        dest = PROJECT_ROOT / "artifacts" / "mlareas" / "ics" / "sites" / name
        dest.mkdir(parents=True, exist_ok=True)
        b["dataset"] = f"site fit: {name}"
        joblib.dump(b, dest / "bundle.joblib", compress=3)
        if a.activate:
            (dest.parent / "ACTIVE").write_text(name, "utf-8")
            res["active"] = True
        print(json.dumps(res, indent=1, default=float))
        return
    res = hai2103(a.days, a.seeds) if a.what == "hai2103" else batadal(a.seeds)
    a.results.mkdir(parents=True, exist_ok=True)
    path = a.results / "ics_site_fit.json"
    doc = json.loads(path.read_text("utf-8")) if path.is_file() else {}
    doc[a.what] = res | {"when": time.strftime("%Y-%m-%d %H:%M")}
    path.write_text(json.dumps(doc, indent=1, default=float), "utf-8")
    print(json.dumps(res, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
