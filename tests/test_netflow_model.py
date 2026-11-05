"""Tests for the NetFlow detector loader/scorer and its deploy helpers (owner 2026-10-15)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from scripts import deploy_netflow as dn
from src.models.mlp import SharedMLP
from src.soc import netflow_model as nm

FEATS = ["IN_BYTES", "OUT_BYTES", "L4_DST_PORT", "FLOW_DURATION_MILLISECONDS"]


def _artifact(tmp: Path, weights: dict[str, float]) -> Path:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, len(FEATS))) * 100 + 500
    y = (X[:, 0] > 500).astype(int)
    sc = StandardScaler().fit(X)
    rf = RandomForestClassifier(n_estimators=5, random_state=0).fit(sc.transform(X), y)
    mlp = SharedMLP(n_features=len(FEATS), n_classes=2, hidden_layers=2, hidden_width=8)
    meta = {
        "feature_names": FEATS,
        "label_names": ["BENIGN", "ATTACK"],
        "hidden_layers": 2,
        "hidden_width": 8,
        "dropout": 0.2,
        "ensemble": weights,
        "temperature": 1.0,
        "alert_threshold": 0.5,
        "input_stats": {"median": np.median(X, axis=0).tolist()},
    }
    path = tmp / "deploy_netflow"
    nm.save_netflow(path, mlp, {"rf": rf}, sc, meta)
    return path


def test_load_missing_returns_none(tmp_path: Path) -> None:
    assert nm.load_netflow(tmp_path / "nothing") is None
    with pytest.raises(FileNotFoundError):
        nm.score_netflow(pd.DataFrame({"IN_BYTES": [1]}), tmp_path / "nothing")


def test_score_matches_columns_by_name(tmp_path: Path) -> None:
    path = _artifact(tmp_path, {"mlp": 1.0, "rf": 1.0})
    assert (path / "trees.joblib").exists()
    df = pd.DataFrame(
        {
            "out_bytes": [400, 600],
            "In Bytes": [900, 100],  # different case/spacing, different order
            "l4_dst_port": ["443", "bad"],  # unparsable -> training median
            "EXTRA": [1, 2],
        }
    )  # FLOW_DURATION_MILLISECONDS missing -> median
    det = nm.load_netflow(path)
    assert det is not None
    X, missing = det.align(df)
    assert missing == ["FLOW_DURATION_MILLISECONDS"]
    assert X[0, 0] == 900 and X[1, 1] == 600 and X[0, 2] == 443
    assert X[1, 2] == pytest.approx(det.medians[2])
    p, alerts = nm.score_netflow(df, path)
    assert p.shape == (2, 2) and np.allclose(p.sum(1), 1)
    assert alerts.dtype == bool and alerts.tolist() == list(p[:, 1] >= 0.5)


def test_mlp_only_artifact_has_no_trees(tmp_path: Path) -> None:
    path = _artifact(tmp_path, {"mlp": 1.0})
    assert not (path / "trees.joblib").exists()
    det = nm.NetflowDetector(path)
    p, _ = det.score(pd.DataFrame({f: [1.0] for f in FEATS}))
    assert p.shape == (1, 2)


def test_threshold_and_metrics_helpers() -> None:
    benign_attack_prob = np.linspace(0, 1, 1001)
    assert dn.alert_threshold(benign_attack_prob) == pytest.approx(0.99)
    assert dn.alert_threshold(np.zeros(10)) == 0.5  # floor
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]])
    y = np.array([0, 1, 0, 1])
    m = dn.split_metrics(p, y, 0.5)
    assert m["attacks_caught"] == 1 and m["false_alarms"] == 1
    assert m["recall_at_threshold"] == 0.5 and m["false_alarm_rate"] == 0.5
    assert "overfitting" in dn.fit_verdict(1.0, 0.9)
    assert "good fit" in dn.fit_verdict(0.99, 0.98)


def test_sample_netflow_caps_per_class(tmp_path: Path) -> None:
    df = pd.DataFrame({"IN_BYTES": np.arange(1000), "Label": [0] * 900 + [1] * 100})
    path = tmp_path / "nf.parquet"
    df.to_parquet(path)
    out = dn.sample_netflow(path, per_class=100)
    counts = out["Label"].value_counts()
    assert counts[1] == 100  # rare class kept whole
    assert 50 < counts[0] < 150  # ~100 of 900
