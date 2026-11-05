"""Deployment preprocessing (src/data/robust.py) and the live-input check."""

from __future__ import annotations

import numpy as np

from src.data.robust import RobustFlowScaler, input_check, input_stats


def _flows(n: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(0)
    heavy = rng.lognormal(3, 2.5, n)  # bytes/s-like, heavy tail
    normal = rng.normal(10, 2, n)
    const = np.full(n, 7.0)
    return np.c_[heavy, normal, const]


def test_robust_scaler_logs_heavy_tails_and_bounds_outliers() -> None:
    x = _flows()
    r = RobustFlowScaler().fit(x)
    assert r.log_cols_.tolist() == [True, False, False]
    z = r.transform(x)
    assert np.isfinite(z).all() and np.allclose(z[:, 2], 0)  # constant -> 0
    assert abs(z[:, 0].mean()) < 1e-6 and abs(z[:, 0].std() - 1) < 1e-6
    far = r.transform(np.array([[1e15, 10.0, 7.0]]))
    assert far[0, 0] <= z[:, 0].max() + 1e-9  # clipped, not extrapolated
    back = r.inverse_transform(r.transform(x[:5]))
    np.testing.assert_allclose(back[:, 1], x[:5, 1])


def test_input_check_flags_unit_mismatch_and_missing() -> None:
    x = _flows()
    stats = input_stats(None, x)
    assert input_check(x[:100], stats)["ok"]
    scaled = (x - x.min(0)) / (x.max(0) - x.min(0) + 1e-9)  # [0,1] data vs raw model
    raw_model_stats = input_stats(None, x * 1e6)
    assert not input_check(scaled[:100], raw_model_stats)["ok"]
    chk = input_check(x[:10], stats, missing=3, n_features=4)
    assert not chk["ok"] and "missing" in chk["warnings"][0]
