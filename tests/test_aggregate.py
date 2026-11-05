"""Aggregation guardrails (06_EXPERIMENTS §5): comparability + summary."""

from __future__ import annotations

import pytest

from src.eval.aggregate import (
    IncompatibleConfigError,
    assert_comparable,
    summarize,
)


def _run(cut_depth: int, hidden_width: int = 128, seed: int = 0, prelim: bool = False):
    return {
        "experiment_id": "E4_cut_depth",
        "seed": seed,
        "preliminary": prelim,
        "config_resolved": {
            "dataset": "cicids2017-improved",
            "model": {"cut_depth": cut_depth, "hidden_width": hidden_width},
            "training": {"lr": 1e-3},
            "runtime": {"batch_size": 256, "device": "cuda"},  # excluded from compare
        },
        "_metrics": {"stage5_cutdepth": {}},
    }


def test_comparable_when_only_iv_differs() -> None:
    runs = [_run(1), _run(2), _run(3)]
    assert_comparable(runs, iv_fields={"model.cut_depth"})  # must not raise


def test_raises_when_non_iv_field_differs() -> None:
    # hidden_width differs but is NOT declared an IV -> incomparable.
    runs = [_run(1, hidden_width=128), _run(2, hidden_width=64)]
    with pytest.raises(IncompatibleConfigError):
        assert_comparable(runs, iv_fields={"model.cut_depth"})


def test_excluded_runtime_fields_never_block() -> None:
    a = _run(1)
    b = _run(1)
    b["config_resolved"]["runtime"]["device"] = "cpu"  # hardware knob, excluded
    assert_comparable([a, b], iv_fields=set())  # must not raise


def test_summarize_flags_preliminary_and_seed_shortfall() -> None:
    runs = [_run(1, seed=0, prelim=True), _run(2, seed=1, prelim=True)]
    s = summarize(runs)
    cell = s["experiments"]["E4_cut_depth"]
    assert cell["n_runs"] == 2
    assert cell["n_preliminary"] == 2
    assert cell["headline_ready"] is False  # preliminary + <5 seeds
