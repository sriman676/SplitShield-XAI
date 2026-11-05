"""Explainability: SHAP shapes, cross-val bounds, LIME top-k sets."""

from __future__ import annotations

from config.base import Config, ModelConfig, SubsampleConfig
from src.data.preprocess import preprocess_frame
from src.data.sources import get_spec, load_raw
from src.explain.explainer import compute_shap
from src.explain.lime_utils import lime_topk
from src.explain.shap_utils import global_mean_abs, jaccard, topk_features
from src.models.mlp import SharedMLP


def _data():
    cfg = Config(
        experiment_id="t",
        dataset="fixture",
        subsample=SubsampleConfig(rare_floor=50, class_cap=200),
        model=ModelConfig(hidden_width=16),
    )
    spec = get_spec("fixture")
    return preprocess_frame(load_raw(spec), spec, cfg, seed=0)


def test_compute_shap_shapes_and_crossval() -> None:
    data = _data()
    model = SharedMLP(n_features=len(data.feature_names), hidden_width=16)
    res = compute_shap(
        model, data.X_test[:4], data.X_train[:10], 9, device="cpu", kernel_instances=3
    )
    assert res.attributions.shape == (4, len(data.feature_names), 9)
    assert -1.0 <= res.kernel_crossval_spearman <= 1.0
    assert res.explainer in {"GradientExplainer", "DeepExplainer", "KernelExplainer"}


def test_shap_aggregation_helpers() -> None:
    import numpy as np

    attr = np.random.default_rng(0).normal(size=(5, 6, 9))
    imp = global_mean_abs(attr)
    assert imp.shape == (6,)
    assert jaccard(topk_features(imp, 3), topk_features(imp, 3)) == 1.0


def test_lime_topk_returns_sets() -> None:
    data = _data()
    model = SharedMLP(n_features=len(data.feature_names), hidden_width=16)
    sets = lime_topk(model, data.X_test[:2], data.X_train[:10], data.feature_names, k=3)
    assert len(sets) == 2
    assert all(isinstance(s, set) for s in sets)
