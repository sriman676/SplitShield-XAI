"""Deploy end to end on the fixture into a temp folder (never the real artifacts):
ensemble choice, per-class thresholds, robust preprocessing choice, neuron growth,
then the SOC Detector loads it and scores flows."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import deploy
import utils.paths
from config.base import build_config, experiment_for_dataset
from src.data.preprocess import preprocess_frame
from src.data.sources import get_spec, load_raw
from src.models.ensemble import EnsembleModel, candidates, fit_trees
from src.models.grow import grow
from src.training.centralized import train_centralized
from src.training.common import eval_macro_f1


@pytest.fixture()
def data_cfg():  # noqa: ANN201
    cfg = build_config(experiment_for_dataset("fixture", fast=True), seed=0, fast=True)
    spec = get_spec("fixture")
    return preprocess_frame(load_raw(spec), spec, cfg, seed=0), cfg


def test_deploy_fast_end_to_end(tmp_path: Path, monkeypatch, data_cfg) -> None:  # noqa: ANN001
    monkeypatch.setattr(deploy, "DEPLOY", tmp_path / "deploy")
    monkeypatch.setattr(utils.paths, "RESULTS_DIR", tmp_path / "results")
    deploy.deploy("fixture", fast=True)
    cols = json.loads((tmp_path / "deploy" / "columns.json").read_text())
    assert "alert_threshold_by_class" in cols and "ensemble" in cols
    assert "input_stats" in cols and "conformal_qhat" in cols
    rep = json.loads((tmp_path / "results" / "deploy_report.json").read_text())
    assert rep["ensemble"] and rep["ensemble"][0]["members"] == "mlp"
    assert 0 <= rep["conformal"]["test_coverage"] <= 1

    from src.soc import engine
    from src.soc.model import Detector

    det = Detector(tmp_path / "deploy")
    assert (det.ensemble is not None) == bool(cols["ensemble"])
    data, _ = data_cfg
    flows = data.frame.iloc[data.idx_test[:50]][det.feature_names]
    logits = engine.flow_logits(det, flows)
    assert logits.shape == (len(flows), len(det.label_names))
    assert np.isfinite(logits).all()
    assert det.last_mlp_logits is not None


def test_ensemble_is_a_valid_scorer(data_cfg) -> None:  # noqa: ANN001
    data, cfg = data_cfg
    mlp, _, _ = train_centralized(data, cfg, "cpu")
    trees = fit_trees(data, 0, fast=True)
    assert {"mlp": 1.0} in candidates(trees)
    ens = EnsembleModel(mlp, trees, {"mlp": 1.0, "rf": 1.0}).eval()
    with torch.no_grad():
        lp = ens(torch.as_tensor(data.X_val[:20], dtype=torch.float32))
    np.testing.assert_allclose(lp.exp().sum(1).numpy(), 1.0, atol=1e-4)
    th = deploy._class_thresholds(ens, data, 1.0, "cpu", 0.01, min_count=1)
    assert all(0.5 <= v <= 1.0 for v in th.values())


def test_growth_and_preprocessing_choice_run(data_cfg) -> None:  # noqa: ANN001
    data, cfg = data_cfg
    n = len(data.label_names)
    d2, info = deploy._choose_preprocessing(data, cfg, 0, "cpu", n)
    assert info["chosen"] in ("standard", "robust") and set(info["val_macro_f1"]) == {
        "standard",
        "robust",
    }
    small = replace(cfg, model=replace(cfg.model, hidden_width=8, hidden_layers=2))
    m, _, _ = train_centralized(d2, small, "cpu")
    c2, m2, log = grow(
        d2, small, m, "cpu", n, train=train_centralized,
        evaluate=lambda mm: eval_macro_f1(mm, d2.X_val, d2.y_val, n, "cpu"),
    )  # fmt: skip
    assert log and m2(torch.as_tensor(d2.X_val[:3], dtype=torch.float32)).shape == (
        3,
        n,
    )


def test_per_flow_why_names_features(tmp_path: Path, monkeypatch, data_cfg) -> None:  # noqa: ANN001
    monkeypatch.setattr(deploy, "DEPLOY", tmp_path / "deploy")
    monkeypatch.setattr(utils.paths, "RESULTS_DIR", tmp_path / "results")
    deploy.deploy("fixture", fast=True)
    from src.soc import engine
    from src.soc.model import Detector

    det = Detector(tmp_path / "deploy")
    data, _ = data_cfg
    flows = data.frame.iloc[data.idx_test[:10]][det.feature_names]
    engine.flow_logits(det, flows)
    alerts = [{"label": det.label_names[1], "reason": "r"} for _ in range(3)]
    engine._why(det, alerts, [0, 4, 9])
    for a in alerts:
        assert 0 < len(a["why"]) <= 3
        assert all(w.rsplit(" ", 1)[0] in det.feature_names for w in a["why"])
        assert "[this flow:" in a["reason"]


def test_conformal_sets_cover_and_flag_unsure() -> None:
    torch.manual_seed(0)
    n, c = 2000, 4
    logits = torch.randn(n, c) * 2
    y = torch.distributions.Categorical(logits=logits).sample().numpy()

    class Fixed(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return logits[x[:, 0].long()]

    X = np.c_[np.arange(n), np.zeros(n)].astype("float32")
    q = deploy.conformal_qhat(Fixed(), X[:1000], y[:1000], 1.0, "cpu", alpha=0.1)
    p = torch.softmax(logits[1000:], 1).numpy()
    sets = deploy.conformal_sets(p, q)
    cover = sets[np.arange(1000), y[1000:]].mean()
    assert 0.86 <= cover <= 0.95  # ~90 % guaranteed coverage on exchangeable data
    assert (sets.sum(1) > 1).any()  # ambiguous flows get several classes
