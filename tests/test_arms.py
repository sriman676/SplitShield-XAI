"""Distributed arms run end to end and return valid metric shapes on the fixture."""

from __future__ import annotations

import pytest

from config.base import Config, ModelConfig, SubsampleConfig, TrainingConfig
from src.data.preprocess import preprocess_frame
from src.data.sources import get_spec, load_raw
from src.training.fedavg import train_fedavg
from src.training.split_sequential import train_split_sequential
from src.training.splitfed import train_splitfed


def _cfg() -> Config:
    return Config(
        experiment_id="t",
        dataset="fixture",
        subsample=SubsampleConfig(rare_floor=50, class_cap=200),
        model=ModelConfig(hidden_layers=4, hidden_width=16, cut_depth=2),
        training=TrainingConfig(sl_cycles=2, local_epochs=1, patience=1, clients=4),
    )


@pytest.fixture(scope="module")
def data():
    cfg = _cfg()
    spec = get_spec("fixture")
    return preprocess_frame(load_raw(spec), spec, cfg, seed=0)


@pytest.mark.parametrize(
    "train_fn", [train_split_sequential, train_fedavg, train_splitfed]
)
def test_arm_runs(train_fn, data) -> None:
    _model, result = train_fn(data, _cfg(), device="cpu")
    assert 0.0 <= result["test"]["macro_f1"] <= 1.0
    hist_key = next(k for k in result if k.endswith("macro_f1"))
    assert len(result[hist_key]) >= 1
