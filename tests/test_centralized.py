"""Centralized training smoke: the loop runs and returns valid metric shapes.

The fixture has no real class signal (random features), so this asserts the
machinery works, not that it learns. The learning / shuffled-label collapse check
is a real-data assertion run in Task 2's driver.
"""

from __future__ import annotations

from config.base import Config, ModelConfig, SubsampleConfig, TrainingConfig
from src.data.preprocess import preprocess_frame
from src.data.sources import get_spec, load_raw
from src.models.mlp import SharedMLP, param_count
from src.training.centralized import train_centralized


def _fast_cfg() -> Config:
    return Config(
        experiment_id="t",
        dataset="fixture",
        subsample=SubsampleConfig(rare_floor=50, class_cap=200),
        model=ModelConfig(hidden_layers=4, hidden_width=16, dropout=0.2),
        training=TrainingConfig(max_epochs=5, patience=2),
    )


def test_param_count_deterministic() -> None:
    a = param_count(SharedMLP(n_features=9, hidden_width=16))
    b = param_count(SharedMLP(n_features=9, hidden_width=16))
    assert a == b and a > 0


def test_centralized_runs_and_returns_metrics() -> None:
    cfg = _fast_cfg()
    spec = get_spec("fixture")
    data = preprocess_frame(load_raw(spec), spec, cfg, seed=0)
    model, history, metrics = train_centralized(data, cfg, device="cpu")
    assert len(history) >= 1
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert set(metrics["per_class"]) == set(data.label_names)
