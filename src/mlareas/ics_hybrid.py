"""ICS v3 hybrid detector (owner 2026-10-22: "train a model that responds" on more ICS data).

The deployed ICS models learn normal behaviour only and miss most real HAI attacks
(recall ~0.17). The hybrid adds what the attacks look like:

  * an autoencoder trained on normal plant data (HAI 22.04 + 23.05) gives every
    sensor's prediction error ("how unusual is this sensor right now");
  * per-sensor trend features over the last 10 s / 60 s / 300 s (change, spread,
    distance from the recent level) show slow manipulations that the autoencoder
    window (30 s) is too short to see;
  * a LightGBM classifier, trained on normal rows plus LABELLED attack files, turns
    those into an attack probability, smoothed over SMOOTH seconds.

The alarm threshold is chosen on a separate labelled validation file with the
operator budget (<= 5 false alarm events per day on normal operation). Explanations
still come from the autoencoder errors (which sensors drifted, in what order).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

SHORT, MID, LONG = 10, 60, 300  # seconds
SMOOTH = 10  # causal moving average of the probability (s)


WINDOWS = (SHORT, MID, LONG)


def feature_names(columns: Sequence[str], windows: Sequence[int] = WINDOWS) -> list[str]:
    short, mid, long = windows
    out = []
    for c in columns:
        out += [f"{c}|dev", f"{c}|d{short}", f"{c}|d{mid}", f"{c}|std{mid}", f"{c}|z{long}"]
    return out + ["dev_max", "dev_mean", "dev_top3", "z_over3"]


def _segment_features(x: np.ndarray, dev: np.ndarray, windows: Sequence[int] = WINDOWS) -> np.ndarray:
    """Features of one continuous segment; every value uses the past only."""
    short, mid, long = windows
    xs, ds = pd.DataFrame(x), pd.DataFrame(dev)
    dev_s = ds.rolling(SMOOTH, min_periods=1).mean().to_numpy(np.float32)
    d_short = (xs - xs.shift(short)).fillna(0.0).to_numpy(np.float32)
    d_mid = (xs - xs.shift(mid)).fillna(0.0).to_numpy(np.float32)
    std_mid = xs.rolling(mid, min_periods=2).std().fillna(0.0).to_numpy(np.float32)
    roll = xs.rolling(long, min_periods=2)
    z_long = ((xs - roll.mean()) / (roll.std() + 0.01)).fillna(0.0).to_numpy(np.float32)
    z_long = np.clip(z_long, -50, 50)
    n, d = x.shape
    per = np.empty((n, d, 5), np.float32)
    per[:, :, 0], per[:, :, 1], per[:, :, 2] = dev_s, d_short, d_mid
    per[:, :, 3], per[:, :, 4] = std_mid, z_long
    srt = np.sort(dev_s, axis=1)
    glob = np.column_stack([srt[:, -1], dev_s.mean(axis=1), srt[:, -3:].mean(axis=1),
                            (np.abs(z_long) > 3).sum(axis=1)]).astype(np.float32)  # fmt: skip
    return np.hstack([per.reshape(n, d * 5), glob])


def features(arr: np.ndarray, dev: np.ndarray, lengths: Sequence[int],
             windows: Sequence[int] = WINDOWS) -> np.ndarray:
    """(rows, features) for consecutive segments of 1 Hz data (arr = normalised
    sensors, dev = autoencoder per-sensor error, same shape)."""
    parts, start = [], 0
    for n in lengths:
        parts.append(_segment_features(arr[start : start + n], dev[start : start + n], windows))
        start += n
    return np.vstack(parts) if parts else np.zeros((0, arr.shape[1] * 5 + 4), np.float32)


TOP = 5  # sensor-agnostic mode: the 5 largest values of each per-sensor feature


def aggregate(X: np.ndarray, d: int) -> np.ndarray:
    """Sensor-agnostic features from ``features()`` output: for each of the 5 per-sensor
    kinds, the TOP largest absolute values across sensors and their mean, plus the 4
    global columns. A classifier on these learns HOW an attack looks (how many sensors
    move, how far, how fast), not WHICH sensor a training attack happened to hit."""
    per = np.abs(X[:, : d * 5].reshape(len(X), d, 5))
    top = -np.sort(-per, axis=1)[:, :TOP, :]  # (n, TOP, 5)
    return np.hstack([top.reshape(len(X), TOP * 5), per.mean(axis=1), X[:, d * 5 :]]).astype(np.float32)


def agg_names() -> list[str]:
    kinds = ("dev", "d_short", "d_mid", "std_mid", "z_long")
    return ([f"top{i + 1}_{k}" for i in range(TOP) for k in kinds] + [f"mean_{k}" for k in kinds]
            + ["dev_max", "dev_mean", "dev_top3", "z_over3"])  # fmt: skip


def smooth(p: np.ndarray, lengths: Sequence[int]) -> np.ndarray:
    parts, start = [], 0
    for n in lengths:
        parts.append(pd.Series(p[start : start + n]).rolling(SMOOTH, min_periods=1).mean().to_numpy())
        start += n
    return np.concatenate(parts) if parts else p


def sensor_dev(det: Any, arr: np.ndarray, lengths: Sequence[int]) -> np.ndarray:
    """Autoencoder per-sensor error for every row (0 where the window is incomplete)."""
    from src.mlareas import ics

    dev = np.zeros(arr.shape, np.float32)
    idx = ics.valid_index(lengths)
    if len(idx):
        for s in range(0, len(idx), 200_000):
            part = idx[s : s + 200_000]
            dev[part] = det.sensor_dev(arr, part)
    return dev


def scores(b: dict[str, Any], arr: np.ndarray, lengths: Sequence[int]) -> np.ndarray:
    """Smoothed attack probability per row for a hybrid bundle."""
    import lightgbm as lgb

    from src.mlareas import ics

    det = ics._detector(b)
    dev = sensor_dev(det, arr, lengths)
    X = features(arr, dev, lengths)
    if b.get("feature_mode") == "aggregate":
        X = aggregate(X, arr.shape[1])
    booster = b.get("_booster")
    if booster is None:
        booster = b["_booster"] = lgb.Booster(model_str=b["gbm"])
    return smooth(np.asarray(booster.predict(X, num_threads=4)), lengths)
