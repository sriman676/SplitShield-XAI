"""Tests for src/explain/ensemble_why.py on tiny synthetic models (owner 2026-10-15)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.explain import ensemble_why as ew
from src.models.ensemble import EnsembleModel
from src.models.mlp import SharedMLP

lightgbm = pytest.importorskip("lightgbm")


def _data(n: int = 600, d: int = 5, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.digitize(X[:, 0], [-0.5, 0.5]) if k == 3 else (X[:, 0] > 0).astype(int)
    return X, y.astype(np.int64)


def _mlp(d: int, k: int) -> SharedMLP:
    torch.manual_seed(0)
    return SharedMLP(n_features=d, n_classes=k, hidden_layers=2, hidden_width=8)


def _det(k: int = 3, weights: dict[str, float] | None = None) -> Any:
    X, y = _data(k=k)
    gbm = lightgbm.LGBMClassifier(n_estimators=20, num_leaves=7, verbose=-1)
    gbm.fit(X, y)
    mlp = _mlp(X.shape[1], k)
    ens = EnsembleModel(mlp, {"lgbm": gbm}, weights or {"mlp": 1.0, "lgbm": 1.0})
    return SimpleNamespace(
        model=mlp,
        ensemble=ens.eval(),
        feature_names=[f"f{i}" for i in range(X.shape[1])],
        label_names=["BENIGN", "DoS", "PortScan"][:k],
        scaler=None,
        input_stats={"median": [1.0] * X.shape[1]},
        temperature=1.0,
    ), X


def test_lgbm_contrib_sums_to_raw_score() -> None:
    det, X = _det()
    gbm = det.ensemble.trees["lgbm"]
    cls = np.array([0, 1, 2, 1])
    a = ew.lgbm_attributions(gbm, X[:4], cls)
    raw = gbm.booster_.predict(X[:4], raw_score=True)
    full = gbm.booster_.predict(X[:4], pred_contrib=True).reshape(4, 3, -1)
    for i, c in enumerate(cls):  # TreeSHAP efficiency: features + bias = raw score
        assert a[i].sum() + full[i, c, -1] == pytest.approx(raw[i, c], abs=1e-6)
    assert np.abs(a[:, 0]).mean() > np.abs(a[:, 1:]).mean()  # f0 drives the label


def test_binary_lgbm_sign_flips_for_class_zero() -> None:
    det, X = _det(k=2)
    gbm = det.ensemble.trees["lgbm"]
    a1 = ew.lgbm_attributions(gbm, X[:3], np.array([1, 1, 1]))
    a0 = ew.lgbm_attributions(gbm, X[:3], np.array([0, 0, 0]))
    np.testing.assert_allclose(a0, -a1)


def test_combine_weights_normalised_members() -> None:
    attrs = {"mlp": np.array([[2.0, 0.0]]), "lgbm": np.array([[0.0, -10.0]])}
    out = ew.combine(attrs, {"mlp": 1.0, "lgbm": 3.0})
    np.testing.assert_allclose(out, [[0.25, -0.75]])
    with pytest.raises(ValueError):
        ew.combine(attrs, {"rf": 1.0})


def test_explain_flows_structure_and_text() -> None:
    det, X = _det()
    out = ew.explain_flows(det, X, [0, 5, 9], k=3)
    assert [o["row"] for o in out] == [0, 5, 9]
    for o in out:
        assert o["members"] == ["lgbm", "mlp"]
        assert 1 <= len(o["top"]) <= 3
        assert o["text"].startswith("1. '")
        assert 0.0 <= o["probability"] <= 1.0
        assert sum(abs(t["contribution"]) for t in o["top"]) <= 1.0 + 1e-9
    assert ew.explain_flows(det, X, []) == []


def test_explain_flows_forced_label_and_mlp_only() -> None:
    det, X = _det()
    out = ew.explain_flows(det, X, [1], labels=["PortScan"])
    assert out[0]["label"] == "PortScan"
    det.ensemble = None
    only = ew.explain_flows(det, X, [1])
    assert only[0]["members"] == ["mlp"]


def test_describe_phrases() -> None:
    assert "40x higher" in ew.describe("Flow Packets/s", 12000.0, 300.0)
    assert "4x lower" in ew.describe("Flow Duration", 25.0, 100.0)
    assert "typical traffic has 0" in ew.describe("SYN Flag Count", 3.0, 0.0)
    assert "close to typical" in ew.describe("x", 1.1, 1.0)
    assert ew.describe("x", 2.0, None) == "'x' is 2"
