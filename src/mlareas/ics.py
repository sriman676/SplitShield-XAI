"""OT/ICS anomaly detection on the HAI 22.04 testbed dataset (owner 2026-10-15).

HAI gives normal-only training files and labelled test files with attacks, sampled
at 1 Hz. Rules followed here:

  * time order is respected: the last 15% of the (chronological) normal training
    period is validation; the labelled test files are only scored, never tuned on;
  * detectors are unsupervised (trained on normal data only): a window
    autoencoder, a forecasting MLP (torch, CPU) and an IsolationForest baseline;
  * validation has no attacks, so model choice and the alarm threshold use a FIXED
    copy of the validation data with synthetic faults injected (offset, drift and
    freeze on 1-3 continuous sensors) - never the test attacks;
  * reported on test: point-wise precision/recall/F1, attack episodes detected,
    detection delay (s) and false alarms per day of normal operation.

Blind final test (owner 2026-10-15): only HAI 22.04 is used for training,
validation and test. HAI 21.03 is the untouched blind version - never downloaded,
opened or tuned on here. ``blind_test(path)`` scores it with the frozen model and
threshold; the lead runs it once, at the very end.

SOC API: ``score_window(df)`` -> anomaly score, alarm, numbered plain-English "why"
(which sensors drifted, in what order, how far from normal, what to check);
``improve(new_data)`` champion/challenger update; ``blind_test(path)``; ``load()``.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn

from src.mlareas import common

AREA = "ics"
LOOKBACK = 30  # seconds of history per window
STEP = 3  # use every 3rd second inside the window -> 10 frames
SMOOTH = 10  # causal moving average of the score, seconds
MERGE_GAP = 60  # alarms closer than this are one alarm event (s)
POST_ATTACK_GRACE = 300  # alarms within 5 min after an attack are not "false" (s)
FAMILIES = ("autoencoder", "forecaster", "isolation_forest")
PERCENTILES = (95.0, 97.0, 98.0, 99.0, 99.5, 99.9, 99.95, 99.99, 99.995, 99.999, 100.0)
ALARM_BUDGET = 5.0  # false alarm events per day on clean validation (owner 2026-10-15)
INJECT_SEED = 1234

SENSOR_HINTS = (
    ("FCV", "flow control valve"), ("PCV", "pressure control valve"),
    ("LCV", "level control valve"), ("FT", "flow sensor"), ("FIT", "flow sensor"),
    ("PIT", "pressure sensor"), ("LIT", "water-level sensor"),
    ("TIT", "temperature sensor"), ("PP", "pump"), ("SIT", "turbine speed sensor"),
    ("VIB", "vibration sensor"), ("VT", "vibration sensor"), ("ST_", "steam-turbine signal"),
    ("HT_", "hydro-turbine signal"), ("SOL", "solenoid valve"), ("B", "control setpoint"),
)  # fmt: skip
PROCESS = {
    "P1": "boiler",
    "P2": "turbine",
    "P3": "water treatment",
    "P4": "HIL power grid",
}


def describe_sensor(name: str) -> str:
    proc, _, tag = name.partition("_")
    kind = next((txt for key, txt in SENSOR_HINTS if tag.startswith(key)), "signal")
    return f"{name} ({PROCESS.get(proc, proc)} {kind})"


# ------------------------------------------------------------ data


@dataclass
class Prep:
    columns: list[str]
    lo: np.ndarray
    span: np.ndarray
    continuous: np.ndarray  # bool mask of non-binary columns (used for injection)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.columns].to_numpy(np.float32)
        return np.clip((x - self.lo) / self.span, -10.0, 10.0).astype(np.float32)


def fit_prep(df: pd.DataFrame) -> Prep:
    num = df.drop(columns=[c for c in ("timestamp", "Attack") if c in df.columns])
    lo, hi = num.min().to_numpy(np.float32), num.max().to_numpy(np.float32)
    keep = (hi - lo) > 1e-6
    cols = [c for c, k in zip(num.columns, keep, strict=True) if k]
    cont = (num[cols].nunique() > 10).to_numpy()
    return Prep(cols, lo[keep], (hi - lo)[keep], cont)


def load_hai(
    data_dir: Path | None = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Return (train files, test files) in chronological order."""
    d = data_dir or common.raw_data_dir("hai")
    read = lambda p: pd.read_csv(p, dtype={"timestamp": str}).astype(  # noqa: E731
        {
            c: np.float32
            for c in pd.read_csv(p, nrows=1).columns
            if c not in ("timestamp", "Attack")
        }
    )
    train = [read(p) for p in sorted(d.glob("hai-22.04_train*.csv"))]
    test = [read(p) for p in sorted(d.glob("hai-22.04_test*.csv"))]
    if not train or not test:
        raise FileNotFoundError(f"HAI 22.04 CSVs not found in {d}")
    return train, test


def offsets() -> np.ndarray:
    return np.arange(-LOOKBACK + STEP, 1, STEP)  # e.g. -27..0 -> 10 frames


def valid_index(lengths: Sequence[int]) -> np.ndarray:
    """Row indices with a full look-back window inside their own segment."""
    out, start = [], 0
    for n in lengths:
        out.append(np.arange(start + LOOKBACK, start + n))
        start += n
    return np.concatenate(out) if out else np.zeros(0, int)


def gather(arr: np.ndarray, idx: np.ndarray, offs: np.ndarray) -> np.ndarray:
    return arr[idx[:, None] + offs[None, :]]  # (n, frames, d)


# ------------------------------------------------------------ detectors


class WindowAE(nn.Module):
    def __init__(self, d: int, frames: int, bottleneck: int = 32) -> None:
        super().__init__()
        n = d * frames
        self.net = nn.Sequential(
            nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, bottleneck), nn.ReLU(),
            nn.Linear(bottleneck, 256), nn.ReLU(), nn.Linear(256, n),
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1)).view_as(x)


class Forecaster(nn.Module):
    def __init__(self, d: int, frames: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d * frames, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, d),
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


def _net_io(
    kind: str, arr: np.ndarray, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if kind == "autoencoder":
        w = gather(arr, idx, offsets())
        return w, w
    # forecaster: predict x[t] from frames ending STEP seconds earlier
    return gather(arr, idx, offsets() - STEP), arr[idx]


def build_net(kind: str, d: int) -> nn.Module:
    frames = len(offsets())
    return WindowAE(d, frames) if kind == "autoencoder" else Forecaster(d, frames)


def train_net(
    kind: str, arr: np.ndarray, idx: np.ndarray, seed: int, epochs: int = 5,
    per_epoch: int = 200_000, net: nn.Module | None = None, lr: float = 1e-3,
) -> nn.Module:  # fmt: skip
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = net if net is not None else build_net(kind, arr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        pick = rng.choice(idx, size=min(per_epoch, len(idx)), replace=False)
        for b in range(0, len(pick), 1024):
            xi, yi = _net_io(kind, arr, pick[b : b + 1024])
            loss = nn.functional.mse_loss(
                model(torch.from_numpy(xi)), torch.from_numpy(yi)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def net_sensor_errors(
    kind: str, model: nn.Module, arr: np.ndarray, idx: np.ndarray
) -> np.ndarray:
    """Per-row, per-sensor squared error, shape (len(idx), d)."""
    out = []
    for b in range(0, len(idx), 20_000):
        xi, yi = _net_io(kind, arr, idx[b : b + 20_000])
        err = (model(torch.from_numpy(xi)).numpy() - yi) ** 2
        out.append(err.mean(axis=1) if kind == "autoencoder" else err)
    return np.concatenate(out) if out else np.zeros((0, arr.shape[1]), np.float32)


def if_features(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.hstack([arr[idx], arr[idx] - arr[idx - LOOKBACK + 1]])


@dataclass
class Detector:
    """One trained detector: produces raw per-row scores and per-sensor deviations."""

    kind: str
    model: Any
    norm: np.ndarray  # per-sensor normal error (nets) or variance (IF)
    mean: np.ndarray  # per-sensor normal mean (for IF deviation explanations)

    def sensor_dev(self, arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if self.kind == "isolation_forest":
            return np.asarray(((arr[idx] - self.mean) ** 2) / self.norm)
        return net_sensor_errors(self.kind, self.model, arr, idx) / self.norm

    def raw_scores(self, arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if self.kind == "isolation_forest":
            return np.asarray(-self.model.score_samples(if_features(arr, idx)))
        # max over sensors: single-sensor faults are not diluted by ~60 quiet
        # sensors (chosen on injected-fault validation F1: max > mean, top-3 mean)
        return np.asarray(self.sensor_dev(arr, idx).max(axis=1))


def fit_detector(
    kind: str, arr: np.ndarray, idx: np.ndarray, seed: int, **kw: Any
) -> Detector:
    mean = arr[idx].mean(axis=0)
    var = arr[idx].var(axis=0) + 1e-4
    rng = np.random.default_rng(seed)
    sub = rng.choice(idx, size=min(100_000, len(idx)), replace=False)
    if kind == "isolation_forest":
        iso = IsolationForest(
            n_estimators=200, max_samples=4096, random_state=seed, n_jobs=4
        )
        iso.fit(if_features(arr, sub))
        return Detector(kind, iso, var, mean)
    net = train_net(kind, arr, idx, seed, **kw)
    norm = net_sensor_errors(kind, net, arr, sub).mean(axis=0) + 1e-6
    return Detector(kind, net, norm, mean)


def rescale_per_sensor(
    det: Detector, arr: np.ndarray, idx: np.ndarray, q: float, seed: int = 0
) -> Detector:
    """Per-sensor (per-channel) normalisation (owner 2026-10-16, ICS underfitting):
    divide each sensor's error by its own ``q`` quantile on normal rows ``idx``
    instead of its mean, so a few noisy-tailed sensors no longer set the alarm
    level for all ~80 (the max over sensors then compares like with like).
    Folded into ``norm``: scoring, explanations and saved bundles are unchanged."""
    if det.kind == "isolation_forest":
        return det
    rng = np.random.default_rng(seed)
    sub = rng.choice(idx, size=min(100_000, len(idx)), replace=False)
    dev = det.sensor_dev(arr, np.sort(sub))
    scale = np.quantile(dev, q, axis=0) + 1e-6
    return Detector(det.kind, det.model, det.norm * scale, det.mean)


def smooth(scores: np.ndarray, lengths: Sequence[int] | None = None) -> np.ndarray:
    """Causal moving average over SMOOTH rows, restarted at segment boundaries."""
    parts, start = [], 0
    for n in lengths or [len(scores)]:
        s = pd.Series(scores[start : start + n]).rolling(SMOOTH, min_periods=1).mean()
        parts.append(s.to_numpy())
        start += n
    return np.concatenate(parts) if parts else scores


# ------------------------------------------------------------ evaluation helpers


def inject_faults(arr: np.ndarray, continuous: np.ndarray, seed: int = INJECT_SEED,
                  n_events: int | None = None) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Copy of ``arr`` with synthetic offset/drift/freeze faults + 0/1 labels."""
    rng = np.random.default_rng(seed)
    out, labels = arr.copy(), np.zeros(len(arr), np.int8)
    cols = np.flatnonzero(continuous)
    n_events = n_events or max(1, len(arr) // 4000)
    for _ in range(n_events):
        length = int(rng.integers(60, 301))
        s = int(rng.integers(LOOKBACK + 1, max(LOOKBACK + 2, len(arr) - length)))
        sl = slice(s, s + length)
        for c in rng.choice(cols, size=int(rng.integers(1, 4)), replace=False):
            kind = rng.integers(3)
            mag = rng.uniform(0.1, 0.5) * rng.choice([-1, 1])
            if kind == 0:
                out[sl, c] += mag
            elif kind == 1:
                out[sl, c] += np.linspace(0, mag, length, dtype=np.float32)
            else:
                out[sl, c] = out[s - 1, c]
        labels[sl] = 1
    return out, labels


def episodes(labels: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) of each run of 1s."""
    d = np.diff(np.concatenate([[0], labels.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def alarm_events(alarm: np.ndarray) -> list[tuple[int, int]]:
    """Alarm runs, merging runs separated by < MERGE_GAP seconds."""
    merged: list[tuple[int, int]] = []
    for s, e in episodes(alarm):
        if merged and s - merged[-1][1] < MERGE_GAP:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def event_metrics(alarm: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Episode detection, delay and false alarms per normal day (1 row = 1 s)."""
    eps = episodes(labels)
    delays = []
    for s, e in eps:
        hit = np.flatnonzero(alarm[s:e])
        if len(hit):
            delays.append(int(hit[0]))
    excused = np.zeros(len(labels), bool)
    for s, e in eps:
        excused[s : min(len(labels), e + POST_ATTACK_GRACE)] = True
    false_events = sum(1 for s, _ in alarm_events(alarm) if not excused[s])
    normal_days = float((labels == 0).sum()) / 86_400
    return {
        "episodes": len(eps),
        "episodes_detected": len(delays),
        "episode_recall": round(len(delays) / max(len(eps), 1), 4),
        "delay_s_median": float(np.median(delays)) if delays else None,
        "delay_s_mean": round(float(np.mean(delays)), 1) if delays else None,
        "false_alarms": false_events,
        "false_alarms_per_day": round(false_events / max(normal_days, 1e-9), 2),
    }


def point_metrics(alarm: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "precision": float(precision_score(labels, alarm, zero_division=0)),
        "recall": float(recall_score(labels, alarm, zero_division=0)),
        "f1": float(f1_score(labels, alarm, zero_division=0)),
    }


def full_scores(det: Detector, arr: np.ndarray, lengths: Sequence[int]) -> np.ndarray:
    """Smoothed score for every row (rows without full history get 0)."""
    idx = valid_index(lengths)
    raw = np.zeros(len(arr), np.float64)
    raw[idx] = det.raw_scores(arr, idx)
    return smooth(raw, lengths)


def choose_threshold(
    clean: np.ndarray, injected: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """Pick the clean-validation percentile that maximises injected-val F1 among the
    thresholds that keep false alarm EVENTS on clean validation within
    ALARM_BUDGET per day (owner 2026-10-15: at most 5 alarms/day for an operator).
    If no percentile meets the budget, the strictest one is used."""
    days = max(len(clean) / 86_400, 1e-9)
    best = (-1.0, float(np.max(clean)) + 1e-9, 100.0)
    for pct in PERCENTILES:
        thr = float(np.percentile(clean, pct))
        if len(alarm_events((clean > thr).astype(int))) / days > ALARM_BUDGET:
            continue
        f1 = point_metrics((injected > thr).astype(int), labels)["f1"]
        if f1 > best[0]:
            best = (f1, thr, pct)
    return best[1], best[2]


# ------------------------------------------------------------ SOC API

_CACHE: dict[str, Any] = {}


def site_bundle_path(root: Path | None = None) -> Path | None:
    """The plant's own model when one was fitted and activated (scripts/ics_site_fit.py
    ``fit --activate``): artifacts/mlareas/ics/sites/ACTIVE names it."""
    sites = common.area_dir(AREA, root) / "sites"
    try:
        name = (sites / "ACTIVE").read_text("utf-8").strip()
    except OSError:
        return None
    p = sites / name / "bundle.joblib"
    return p if name and p.is_file() else None


def load(root: Path | None = None, refresh: bool = False) -> dict[str, Any]:
    """The active plant model if one is fitted (per-plant fit, owner 2026-10-24), else
    the shipped HAI model."""
    key = str(root)
    if refresh or key not in _CACHE:
        site = site_bundle_path(root)
        if site is not None:
            import joblib

            _CACHE[key] = joblib.load(site)
        else:
            _CACHE[key] = common.load_bundle(AREA, root)
    bundle: dict[str, Any] = _CACHE[key]
    return bundle


def _detector(bundle: dict[str, Any]) -> Detector:
    if "_det" not in bundle:
        kind = "autoencoder" if bundle["family"] == HYBRID else bundle["family"]
        if kind == "isolation_forest":
            model: Any = bundle["model"]
        else:
            model = build_net(kind, len(bundle["prep"].columns))
            model.load_state_dict(
                {k: torch.from_numpy(v) for k, v in bundle["model"].items()}
            )
            model.eval()
        bundle["_det"] = Detector(kind, model, bundle["norm"], bundle["mean"])
    det: Detector = bundle["_det"]
    return det


DRIFT_FRACTION = 0.5  # nets: a sensor "drifts" above half the alarm threshold
DEVIATES = 9.0  # IsolationForest explanations: squared z-score > 9 (|z| > 3)
CHECKS = (
    ("CV", "compare the valve's reported position with its command and the controller output"),
    ("SOL", "confirm the solenoid valve's state locally"),
    ("PP", "check locally that the pump is really running/stopped as reported"),
    ("B", "confirm on the HMI / engineering-station log who changed this setpoint and why"),
    ("VIB", "check turbine vibration and speed locally"),
    ("SIT", "check turbine speed locally"),
)  # fmt: skip


def operator_check(name: str) -> str:
    tag = name.partition("_")[2]
    hit = next((txt for key, txt in CHECKS if key in tag[:4]), None)
    return hit or "compare the reading with a local gauge or redundant sensor"


def _range_text(raw: float, lo: float, hi: float) -> str:
    span = max(hi - lo, 1e-9)
    if raw > hi:
        return f"reads {raw:.3g}, {(raw - hi) / span:.0%} of its normal span above the normal maximum {hi:.3g}"
    if raw < lo:
        return f"reads {raw:.3g}, {(lo - raw) / span:.0%} of its normal span below the normal minimum {lo:.3g}"
    return f"reads {raw:.3g}, inside its normal range {lo:.3g}-{hi:.3g} (the deviation is in how it moves relative to the other sensors)"


def _drift_order(
    dev_rows: np.ndarray, top: np.ndarray, level: float
) -> list[tuple[int, int | None]]:
    """(sensor, first row index over ``level``) sorted by onset; None = never."""
    onsets = []
    for i in top:
        over = np.flatnonzero(dev_rows[:, i] > level)
        onsets.append((int(i), int(over[0]) if len(over) else None))
    return sorted(onsets, key=lambda t: (t[1] is None, t[1] if t[1] is not None else 0))


def explain_window(
    df: pd.DataFrame, b: dict[str, Any], dev_rows: np.ndarray, score: float, alarm: bool
) -> list[str]:
    """Numbered steps: which sensors drifted, in what order, how far, what to check."""
    prep: Prep = b["prep"]
    det = _detector(b)
    recent = dev_rows[-SMOOTH:].mean(axis=0)
    top = np.argsort(recent)[::-1][:3]
    unit = (
        "normal deviation"
        if det.kind == "isolation_forest"
        else "normal prediction error"
    )
    n = len(dev_rows)
    level = (DEVIATES if det.kind == "isolation_forest"
             else DRIFT_FRACTION * float(b.get("ae_threshold", b["threshold"])))  # fmt: skip
    steps = []
    for i, onset in _drift_order(dev_rows, top, level):
        name = prep.columns[i]
        raw = float(df[name].iloc[-1])
        lo, hi = float(prep.lo[i]), float(prep.lo[i] + prep.span[i])
        if not alarm:
            when = "is among the largest deviations"
        elif onset is None:
            when = "is elevated but below the drift level"
        elif onset == 0:
            when = f"was already drifting when this {n}s window began"
        else:
            when = f"started drifting {n - onset}s ago"
        steps.append(f"{describe_sensor(name)} {when}; it is now {recent[i]:.1f}x its {unit} "
                     f"and {_range_text(raw, lo, hi)}.")  # fmt: skip
    if alarm:
        steps.append("Operator check: " + "; ".join(
            f"{prep.columns[i]}: {operator_check(prep.columns[i])}" for i in top) + ".")  # fmt: skip
        verdict = (f"Verdict: ANOMALY (score {score:.3g} > alarm threshold "
                   f"{b['threshold']:.3g}; {_kind(b)}).")  # fmt: skip
        action = ("What to do: verify the listed readings in the field before trusting the "
                  "HMI; if they disagree, suspect spoofed data or an unauthorised change and "
                  "escalate to OT security.")  # fmt: skip
    else:
        steps.insert(0, "No alarm: the strongest deviation is below the alarm "
                        "threshold. The largest deviations are listed for context.")  # fmt: skip
        verdict = (
            f"Verdict: NORMAL (score {score:.3g} <= threshold {b['threshold']:.3g})."
        )
        action = "What to do: nothing; keep monitoring."
    return [f"{k}. {s}" for k, s in enumerate(steps, 1)] + [verdict, action]


def _kind(b: dict[str, Any]) -> str:
    """Model name for messages: the hybrid reports itself, not its autoencoder."""
    return "hybrid (autoencoder + LightGBM)" if b.get("family") == HYBRID else _detector(b).kind


HYBRID = "hybrid_gbm"  # ICS v3 (src/mlareas/ics_hybrid.py)


def bundle_scores(b: dict[str, Any], arr: np.ndarray, lengths: Sequence[int]) -> np.ndarray:
    """Smoothed alarm score for every row, for any deployed family."""
    if b.get("family") == HYBRID:
        from src.mlareas import ics_hybrid

        return ics_hybrid.scores(b, arr, lengths)
    return full_scores(_detector(b), arr, lengths)


def score_window(
    df: pd.DataFrame, bundle: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Score the latest second of a 1 Hz sensor window (>= 31 rows; ~300 is ideal)."""
    b = bundle or load()
    prep: Prep = b["prep"]
    df = df.rename(columns=b.get("column_map", {}))  # newer HAI names -> the model's
    missing = [c for c in prep.columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"window lacks {len(missing)} sensor columns, e.g. {missing[:3]}"
        )
    if len(df) <= LOOKBACK:
        raise ValueError(f"need more than {LOOKBACK} rows (1 Hz) of history")
    df = df.iloc[-(LOOKBACK + 300) :]
    arr = prep.transform(df)
    det = _detector(b)
    idx = np.arange(LOOKBACK, len(arr))
    dev_rows = det.sensor_dev(arr, idx)
    if b.get("family") == HYBRID:  # attack probability of the last second (smoothed)
        score = float(bundle_scores(b, arr, [len(arr)])[-1])
    else:
        score = float(det.raw_scores(arr, idx[-SMOOTH:]).mean())
    alarm = score > b["threshold"]
    top = np.argsort(dev_rows[-SMOOTH:].mean(axis=0))[::-1][:3]
    return {
        "anomaly_score": round(score, 4),
        "threshold": round(float(b["threshold"]), 4),
        "alarm": bool(alarm),
        "top_sensors": [prep.columns[i] for i in top],
        "why": explain_window(df, b, dev_rows, score, bool(alarm)),
        "model": _kind(b),
    }


# ------------------------------------------------------------ blind final test


def _labels_for(path: Path, df: pd.DataFrame) -> np.ndarray:
    for col in ("Attack", "attack", "label"):
        if col in df.columns:
            return (df[col].to_numpy() > 0).astype(np.int8)
    label_file = path.with_name(path.name.replace("hai-test", "label-test"))
    if label_file.exists():
        lab = pd.read_csv(label_file)
        return (lab[lab.columns[-1]].to_numpy() > 0).astype(np.int8)
    raise ValueError(f"no attack labels found for {path.name}")


def blind_test(
    path: str | Path, bundle: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Score an unseen HAI version with the FROZEN model + threshold (no refitting).

    ``path`` is a HAI version folder (its ``*test*.csv`` files are scored) or one
    CSV. Sensors the model needs but the file lacks are filled with the normal
    training midpoint and listed in the result. Run once, at the very end.
    """
    b = bundle or load()
    prep: Prep = b["prep"]
    p = Path(path)
    files = [p] if p.is_file() else sorted(f for f in p.glob("*test*.csv*")
                                          if not f.name.startswith("label")
                                          and f.name.endswith((".csv", ".csv.gz")))  # fmt: skip
    if not files:
        raise FileNotFoundError(f"no *test*.csv or *test*.csv.gz files in {p}")
    rename = b.get("column_map", {})  # newer HAI names -> the model's names
    frames, labels, missing = [], [], set()
    for f in files:
        df = pd.read_csv(f).rename(columns=rename)
        labels.append(_labels_for(f, df))
        for i, c in enumerate(prep.columns):
            if c not in df.columns:
                missing.add(c)
                df[c] = prep.lo[i] + prep.span[i] / 2
        frames.append(prep.transform(df))
    arr, y = np.vstack(frames), np.concatenate(labels)
    lengths = [len(f) for f in frames]
    t0 = time.perf_counter()
    scores = bundle_scores(b, arr, lengths)
    speed = len(arr) / max(time.perf_counter() - t0, 1e-9)
    alarm = (scores > b["threshold"]).astype(int)
    return {
        "files": [f.name for f in files], "rows": len(arr),
        "point": point_metrics(alarm, y), "events": event_metrics(alarm, y),
        "missing_sensors_filled": sorted(missing), "rows_per_s": round(speed, 1),
        "model": f"{_kind(b)} v{b.get('version', 1)}", "threshold": b["threshold"],
    }  # fmt: skip


# ------------------------------------------------------------ champion/challenger


def _val_detection(det: Detector, b: dict[str, Any]) -> tuple[float, float, float]:
    """(injected-val F1, threshold, rows/s) for a detector on the fixed val set."""
    lengths = [len(b["val_clean"])]
    clean = full_scores(det, b["val_clean"], lengths)
    t0 = time.perf_counter()
    inj = full_scores(det, b["val_injected"], lengths)
    speed = len(inj) / max(time.perf_counter() - t0, 1e-9)
    thr = float(np.percentile(clean, b["percentile"]))
    f1 = point_metrics((inj > thr).astype(int), b["val_labels"])["f1"]
    return f1, thr, round(speed, 1)


def improve(
    new_labelled_data: pd.DataFrame, root: Path | None = None
) -> dict[str, Any]:
    """Champion/challenger update with new 1 Hz plant data.

    Rows with ``Attack == 1`` are dropped (we only learn normal behaviour). The
    challenger fine-tunes the champion network on new normal data mixed with the
    stored replay sample (IsolationForest is refitted on both). It is promoted only
    if injected-fault F1 on the fixed validation set is not worse.
    """
    champ = load(root, refresh=True)
    if champ.get("family") == HYBRID:
        # the classifier was trained on labelled attacks of the HAI plants; new normal
        # data alone cannot re-check it, so a hybrid is retrained by its script
        return {"promoted": False, "reason": "hybrid ICS model: retrain with "
                "python -m scripts.train_ics_v3 (the inbox only updates normal-only models)"}
    df = new_labelled_data
    if "Attack" in df.columns:
        df = df[df["Attack"] == 0]
    new = champ["prep"].transform(df)
    data = np.vstack([champ["replay"], new]).astype(np.float32)
    idx = valid_index([len(champ["replay"]), len(new)])
    det_c = _detector(champ)
    if det_c.kind == "isolation_forest":
        det_n = fit_detector("isolation_forest", data, idx, seed=champ["seed"])
    else:
        net = train_net(det_c.kind, data, idx, champ["seed"], epochs=2, per_epoch=50_000,
                        net=copy.deepcopy(det_c.model), lr=3e-4)  # fmt: skip
        norm = net_sensor_errors(det_c.kind, net, data, idx).mean(axis=0) + 1e-6
        det_n = Detector(det_c.kind, net, norm, det_c.mean)
    f_c, _, sp_c = _val_detection(det_c, champ)
    f_n, thr_n, sp_n = _val_detection(det_n, champ)
    result = common.champion_challenger(f_c, f_n, sp_c, sp_n, "val_injected_f1")
    result["new_normal_rows"] = len(new)
    if result["promoted"]:
        cand = {k: v for k, v in champ.items() if k != "_det"}
        cand.update(
            threshold=thr_n, norm=det_n.norm, version=int(champ.get("version", 1)) + 1
        )
        cand["model"] = _export_model(det_n)
        common.save_bundle(AREA, cand, root)
        load(root, refresh=True)
    return result


def _export_model(det: Detector) -> Any:
    if det.kind == "isolation_forest":
        return det.model
    return {k: v.detach().numpy().copy() for k, v in det.model.state_dict().items()}


# ------------------------------------------------------------ research training


@dataclass
class Splits:
    prep: Prep
    train: np.ndarray
    train_lengths: list[int]
    val: np.ndarray
    val_injected: np.ndarray
    val_labels: np.ndarray
    fit_injected: np.ndarray  # same-size train slice with the same kind of faults
    fit_labels: np.ndarray
    test: np.ndarray
    test_labels: np.ndarray
    test_lengths: list[int]


def _drop_meta(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in ("timestamp", "Attack") if c in df.columns])


def make_splits(train_dfs: list[pd.DataFrame], test_dfs: list[pd.DataFrame]) -> Splits:
    """Chronological split: last 15% of normal time = validation."""
    normal = pd.concat([_drop_meta(d) for d in train_dfs], ignore_index=True)
    cut = int(0.85 * len(normal))
    lengths, start = [], 0
    for d in train_dfs:
        lengths.append(max(0, min(len(d), cut - start)))
        start += len(d)
    lengths = [n for n in lengths if n > LOOKBACK]
    prep = fit_prep(normal.iloc[:cut])
    tr, va = prep.transform(normal.iloc[:cut]), prep.transform(normal.iloc[cut:])
    del normal
    va_inj, va_lab = inject_faults(va, prep.continuous)
    big = int(np.argmax(lengths))
    s0 = sum(lengths[:big])
    fit_slice = tr[s0 : s0 + min(len(va), lengths[big])]
    fit_inj, fit_lab = inject_faults(fit_slice, prep.continuous)
    te = np.vstack([prep.transform(d) for d in test_dfs])
    te_lab = np.concatenate([d["Attack"].to_numpy() > 0 for d in test_dfs]).astype(
        np.int8
    )
    return Splits(prep, tr, lengths, va, va_inj, va_lab, fit_inj, fit_lab, te, te_lab,
                  [len(d) for d in test_dfs])  # fmt: skip


def _evaluate(
    det: Detector, sp: Splits
) -> tuple[dict[str, Any], float, float, np.ndarray]:
    one = lambda a: [len(a)]  # noqa: E731
    clean = full_scores(det, sp.val, one(sp.val))
    val_inj = full_scores(det, sp.val_injected, one(sp.val))
    thr, pct = choose_threshold(clean, val_inj, sp.val_labels)
    fit_inj = full_scores(det, sp.fit_injected, one(sp.fit_injected))
    test = full_scores(det, sp.test, sp.test_lengths)
    alarm = lambda s: (s > thr).astype(int)  # noqa: E731
    clean_days = len(clean) / 86_400
    res = {
        "train": point_metrics(alarm(fit_inj), sp.fit_labels)
        | {"episode_recall": event_metrics(alarm(fit_inj), sp.fit_labels)["episode_recall"]},
        "val": point_metrics(alarm(val_inj), sp.val_labels)
        | {"episode_recall": event_metrics(alarm(val_inj), sp.val_labels)["episode_recall"],
           "clean_false_alarms_per_day":
               len(alarm_events(alarm(clean))) / max(clean_days, 1e-9)},
        "test": point_metrics(alarm(test), sp.test_labels)
        | event_metrics(alarm(test), sp.test_labels),
    }  # fmt: skip
    return res, thr, pct, test


def _summary(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        out[split] = {}
        for m, v in per_seed[0][split].items():
            vals = [r[split][m] for r in per_seed]
            if isinstance(v, int | float) and all(x is not None for x in vals):
                out[split][m] = common.mean_sd(vals)
    out["fit"] = common.fit_verdict(
        out["train"]["f1"]["mean"], out["val"]["f1"]["mean"]
    )
    out["fit"]["note"] = "F1 on synthetic injected faults (train slice vs validation)"
    return out


def _bundle(
    det: Detector, sp: Splits, thr: float, pct: float, seed: int
) -> dict[str, Any]:
    return {
        "area": AREA, "dataset": "HAI 22.04", "family": det.kind, "seed": seed,
        "version": 1, "model": _export_model(det), "prep": sp.prep, "norm": det.norm,
        "mean": det.mean, "threshold": thr, "percentile": pct,
        "val_clean": sp.val, "val_injected": sp.val_injected, "val_labels": sp.val_labels,
        "replay": sp.train[-60_000:],
    }  # fmt: skip


def _api_speed(bundle: dict[str, Any], test_df: pd.DataFrame, n: int = 100) -> float:
    rows = LOOKBACK + 300
    starts = np.linspace(0, len(test_df) - rows - 1, n).astype(int)
    windows = [test_df.iloc[s : s + rows] for s in starts]
    return common.items_per_second(
        lambda: [score_window(w, bundle) for w in windows], n, 1
    )


def train(
    data_dir: Path | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    families: Sequence[str] = FAMILIES,
    epochs: int = 5,
    per_epoch: int = 200_000,
    save: bool = True,
    root: Path | None = None,
    frames: tuple[list[pd.DataFrame], list[pd.DataFrame]] | None = None,
) -> dict[str, Any]:
    """Every family x seed on HAI 22.04; choose family + threshold on validation."""
    torch.set_num_threads(4)
    train_dfs, test_dfs = frames or load_hai(data_dir)
    sp = make_splits(train_dfs, test_dfs)
    results: dict[str, Any] = {
        "dataset": "HAI 22.04 (train1-6 normal, test1-4 labelled)",
        "blind_holdout_version": "hai-21.03 (never downloaded or opened; lead runs blind_test)",
        "sensors_used": len(sp.prep.columns),
        "rows": {"train": len(sp.train), "val": len(sp.val), "test": len(sp.test)},
        "test_attack_rows": int(sp.test_labels.sum()),
        "seeds": list(seeds),
        "selection": "mean validation F1 on injected synthetic faults; threshold = "
                     "clean-val percentile maximising that F1",
        "families": {},
    }  # fmt: skip
    seed0: dict[str, tuple[Detector, float, float, np.ndarray]] = {}
    for fam in families:
        per_seed = []
        for seed in seeds:
            det = fit_detector(fam, sp.train, valid_index(sp.train_lengths), seed,
                               **({} if fam == "isolation_forest"
                                  else {"epochs": epochs, "per_epoch": per_epoch}))  # fmt: skip
            res, thr, pct, test_scores = _evaluate(det, sp)
            res["threshold_percentile"] = pct
            per_seed.append(res)
            if seed == seeds[0]:
                seed0[fam] = (det, thr, pct, test_scores)
        results["families"][fam] = _summary(per_seed)
        results["families"][fam]["threshold_percentiles"] = [
            r["threshold_percentile"] for r in per_seed
        ]
    best = max(families, key=lambda f: results["families"][f]["val"]["f1"]["mean"])
    det, thr, pct, test_scores = seed0[best]
    results["selected"] = best
    bundle = _bundle(det, sp, thr, pct, int(seeds[0]))
    t0 = time.perf_counter()
    full_scores(det, sp.test, sp.test_lengths)
    results["speed_rows_per_s_batch"] = round(
        len(sp.test) / (time.perf_counter() - t0), 1
    )
    results["speed_score_window_per_s"] = _api_speed(bundle, test_dfs[0])
    if save:
        common.save_bundle(AREA, bundle, root)
        load(root, refresh=True)
    results["_plot"] = {"scores": test_scores, "labels": sp.test_labels, "threshold": thr,
                        "lengths": sp.test_lengths, "fit": results["families"]}  # fmt: skip
    return results
