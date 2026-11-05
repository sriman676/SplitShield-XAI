"""Metrics correctness and the control-enforcement gate (08_ENGINEERING §6)."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval import baselines as B
from src.eval.metrics import (
    MissingControlError,
    classification_metrics,
    leakage_ratio,
    metric_entry,
)


def test_macro_f1_matches_hand_computation() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    # f1: class0=0.6667, class1=0.8, class2=1.0 -> macro=0.8222
    m = classification_metrics(y_true, y_pred, ("A", "B", "C"))
    assert m["macro_f1"] == pytest.approx(0.82222, abs=1e-4)
    assert m["per_class"]["C"]["f1"] == pytest.approx(1.0)
    assert m["per_class"]["A"]["recall"] == pytest.approx(0.5)


def test_missing_control_raises() -> None:
    with pytest.raises(MissingControlError):
        metric_entry("leakage_ratio", 0.5)  # control required
    # a non-controlled metric is fine
    assert metric_entry("macro_f1", 0.9)["control"] is None


def test_interpretable_flag_on_low_support() -> None:
    y_true = np.array([0] * 40 + [1] * 5)
    y_pred = y_true.copy()
    m = classification_metrics(y_true, y_pred, ("A", "B"))
    assert m["per_class"]["A"]["interpretable"] is True
    assert m["per_class"]["B"]["interpretable"] is False  # support 5 < 30


def test_leakage_ratio_control_and_value() -> None:
    good = leakage_ratio(mse_attack=0.2, mse_mean_predictor=1.0)
    assert good["value"] == pytest.approx(0.8)
    assert good["control"]["name"] == "mean_predictor"
    learned_nothing = leakage_ratio(mse_attack=1.0, mse_mean_predictor=1.0)
    assert learned_nothing["value"] == pytest.approx(0.0)


def test_mean_predictor_and_prior_baselines() -> None:
    X = np.array([[0.0], [2.0], [4.0]])  # mean 2.0
    assert B.mean_predictor_mse(X, np.array([[2.0]])) == pytest.approx(0.0)
    y = np.array([0, 0, 0, 0, 1])  # majority class 0
    # constant predictor => balanced accuracy ~ 1/n_classes
    assert B.class_prior_balanced_accuracy(y, 2) == pytest.approx(0.5)
