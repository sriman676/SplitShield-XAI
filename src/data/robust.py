"""Robust preprocessing for the DEPLOYED detector (owner decision 2026-10-14).

Research runs keep the registered StandardScaler. Network-flow counts and rates are
heavy-tailed (a few flows carry 10^9 bytes/s), which squeezes ordinary traffic into
a tiny slice of the standardized range. ``RobustFlowScaler``:

  1. clips every feature to its training [0.1 %, 99.9 %] quantiles (outliers no
     longer dominate; live flows far outside training are bounded, not extrapolated);
  2. log1p-transforms non-negative heavy-tailed features (skewness > ``SKEW``);
  3. standardizes (mean 0, std 1; constant columns -> 0).

It also keeps per-feature training medians and ranges, used to fill missing live
values and to detect live input in different units (``input_check``). Deploy picks
it over the plain scaler only if validation macro-F1 is higher.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SKEW = 2.0
Q_LO, Q_HI = 0.001, 0.999


class RobustFlowScaler:
    """clip -> log1p(heavy-tailed) -> standardize, fitted on the training split."""

    def fit(self, X: np.ndarray) -> RobustFlowScaler:
        x = np.asarray(X, dtype=np.float64)
        self.lo_, self.hi_ = np.quantile(x, [Q_LO, Q_HI], axis=0)
        xc = np.clip(x, self.lo_, self.hi_)
        sd = xc.std(axis=0)
        mu = xc.mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            skew = np.where(sd > 0, (((xc - mu) / sd) ** 3).mean(axis=0), 0.0)
        self.log_cols_ = (self.lo_ >= 0) & (skew > SKEW)
        xt = self._shape(xc)
        self.mean_ = xt.mean(axis=0)
        self.scale_ = xt.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        self.median_ = np.median(x, axis=0)
        self.min_, self.max_ = x.min(axis=0), x.max(axis=0)
        return self

    def _shape(self, xc: np.ndarray) -> np.ndarray:
        out = xc.copy()
        out[:, self.log_cols_] = np.log1p(out[:, self.log_cols_])
        return out

    def transform(self, X: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(X, dtype=np.float64), self.lo_, self.hi_)
        return (self._shape(x) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Xs: np.ndarray) -> np.ndarray:
        """Back to raw units (clipping is not undone: values stay in range)."""
        x = np.asarray(Xs, dtype=np.float64) * self.scale_ + self.mean_
        x[:, self.log_cols_] = np.expm1(x[:, self.log_cols_])
        return x

    @property
    def n_features_in_(self) -> int:
        return len(self.mean_)


def input_stats(scaler: Any, X_train_raw: np.ndarray) -> dict[str, list[float]]:
    """Per-feature training median / min / max for live-input checks (columns.json)."""
    x = np.asarray(X_train_raw, dtype=np.float64)
    return {
        # deciles per feature: the reference distribution for live drift (PSI)
        "deciles": np.quantile(x, np.linspace(0.1, 0.9, 9), axis=0).T.tolist(),
        "median": np.median(x, axis=0).tolist(),
        "min": x.min(axis=0).tolist(),
        "max": x.max(axis=0).tolist(),
    }


def input_check(
    X: np.ndarray, stats: dict[str, list[float]], missing: int = 0, n_features: int = 0
) -> dict[str, Any]:
    """Does live input look like the training data? Flags different units (e.g. raw
    bytes against a model trained on [0,1]-normalized data, or the reverse) and
    missing columns. ``ok`` is False when the input should not be trusted."""
    x = np.asarray(X, dtype=np.float64)
    lo, hi = np.asarray(stats["min"]), np.asarray(stats["max"])
    span = np.maximum(hi - lo, 1e-12)
    out = (x < lo - 0.5 * span) | (x > hi + 0.5 * span)
    share = float(out.mean()) if out.size else 0.0
    rows_bad = float((out.mean(axis=1) > 0.2).mean()) if out.size else 0.0
    msgs = []
    if rows_bad > 0.5:
        msgs.append(
            f"{rows_bad:.0%} of flows have >20% of features far outside the training "
            "range - different units than the training data (raw vs normalized?)"
        )
    if n_features and missing > n_features // 4:
        msgs.append(
            f"{missing} of {n_features} model features missing (filled with medians)"
        )
    return {"ok": not msgs, "out_of_range_share": share, "warnings": msgs}
