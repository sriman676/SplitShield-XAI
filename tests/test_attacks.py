"""T1 (FGSM/PGD), the random-matched control, and attack metrics (Task 5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.attack.activation_inference import activation_inference
from src.attack.client_attribution import client_attribution, cut_activations
from src.attack.constraints import FeatureConstraints
from src.attack.fgsm_pgd import fgsm, pgd
from src.attack.gan_attack import gan_evasion
from src.attack.random_matched import perturbation_norm, random_matched
from src.eval.metrics import (
    MissingControlError,
    achieved_perturbation,
    attack_success_rate,
    clean_correct_index,
)
from src.models.mlp import SharedMLP
from src.models.splittable import SplittableMLP

FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Header Length",
    "Flow IAT Mean",
    "Idle Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
]


def _setup() -> tuple[FeatureConstraints, np.ndarray]:
    rng = np.random.default_rng(0)
    fwd = rng.integers(1, 500, size=80).astype(float)
    bwd = rng.integers(0, 500, size=80).astype(float)
    dur = rng.integers(1, 1_000_000, size=80).astype(float)
    raw = np.column_stack(
        [
            rng.choice([80, 443, 22], size=80).astype(float),
            dur,
            fwd,
            bwd,
            fwd * 20,
            rng.uniform(0, 1e5, size=80),
            rng.uniform(0, 1e5, size=80),
            (fwd + bwd) * 1e6 / dur,
            (fwd + bwd) * 1e6 / dur,
        ]
    )
    scaler = StandardScaler().fit(raw)
    fc = FeatureConstraints.from_run(FEATURES, scaler, raw)
    x_scaled = scaler.transform(raw)
    return fc, x_scaled


def test_perturbation_norm_linf_and_l2() -> None:
    x = np.zeros((1, 9))
    adv = np.zeros((1, 9))
    adv[0, 1] = 3.0
    adv[0, 2] = 4.0  # both mutable
    mask = np.ones(9, dtype=bool)
    assert perturbation_norm(x, adv, mask, np.inf)[0] == pytest.approx(4.0)
    assert perturbation_norm(x, adv, mask, 2)[0] == pytest.approx(5.0)


def test_random_matched_confined_to_mask_and_constrained() -> None:
    fc, x = _setup()
    rng = np.random.default_rng(1)
    adv = pgd(SharedMLP(9), x, eps=0.5, iters=5, constraints=fc, device="cpu")
    rnd = random_matched(x, adv, fc.mask, fc, p=np.inf, rng=rng)
    # Destination Port (IMMUTABLE) and the two rates (DERIVED) are not attacker-set:
    x_raw, rnd_raw = fc.to_raw(x), fc.to_raw(rnd)
    assert np.allclose(rnd_raw[:, 0], x_raw[:, 0])  # port frozen
    # random-matched actually perturbed something on mutable features
    assert not np.allclose(rnd, x)


def test_pgd_respects_linf_ball_up_to_rounding() -> None:
    fc, x = _setup()
    eps = 0.3
    adv = pgd(SharedMLP(9), x, eps=eps, iters=10, constraints=fc, device="cpu")
    # Mutable features stay within the scaled eps ball, plus the <=0.5 raw-unit
    # slack that integer rounding (a required constraint) can add (02_THREAT §4.2).
    slack = np.where(fc.integer_cols, 0.5 / fc.scaler.scale_, 0.0)
    within = np.abs(adv - x)[:, fc.mask] <= (eps + slack[fc.mask] + 1e-6)
    assert within.all()


def test_fgsm_low_violation_rate_after_projection() -> None:
    fc, x = _setup()
    adv = fgsm(SharedMLP(9), x, eps=0.4, constraints=fc, device="cpu")
    # Its own output, re-projected, should already satisfy the constraints.
    assert fc.violation_rate(fc.to_raw(adv), fc.to_raw(x)) == pytest.approx(0.0)


def test_pgd_deterministic_under_eval() -> None:
    fc, x = _setup()
    torch.manual_seed(0)
    m = SharedMLP(9)
    a = pgd(m, x, eps=0.3, iters=4, constraints=fc, device="cpu")
    b = pgd(m, x, eps=0.3, iters=4, constraints=fc, device="cpu")
    assert np.allclose(a, b)  # eval() => dropout off => reproducible


def test_asr_over_clean_correct_filter() -> None:
    y_true = np.array([0, 1, 2, 1, 2])  # 0 = BENIGN
    y_clean = np.array([0, 1, 2, 0, 2])  # idx1,2,4 correct & non-benign; idx3 wrong
    C = clean_correct_index(y_true, y_clean, benign_index=0)
    assert list(C) == [1, 2, 4]
    y_adv_on_C = np.array([0, 0, 2])  # two flipped to BENIGN of three
    entry = attack_success_rate(y_adv_on_C, len(C), control={"name": "random_matched"})
    assert entry["value"] == pytest.approx(2 / 3)
    assert entry["denominator"] == 3


def test_asr_requires_control() -> None:
    with pytest.raises(MissingControlError):
        attack_success_rate(np.array([0]), 1, control=None)  # type: ignore[arg-type]


def test_client_attribution_controlled_and_bounded() -> None:
    fc, x = _setup()
    client, _ = SplittableMLP(9).split(2)
    rng = np.random.default_rng(3)
    idx = rng.permutation(len(x))
    X_by_client = [x[idx[:30]], x[idx[30:55]], x[idx[55:]]]
    entry = client_attribution(client, X_by_client, device="cpu", seed=0)
    assert 0.0 <= entry["value"] <= 1.0
    assert entry["control"]["name"] == "class_prior"
    assert entry["control"]["balanced_accuracy"] == pytest.approx(1 / 3, abs=0.05)


def test_client_attribution_single_client_degenerate() -> None:
    fc, x = _setup()
    client, _ = SplittableMLP(9).split(2)
    entry = client_attribution(client, [x, np.empty((0, 9))], device="cpu", seed=0)
    assert entry["control"] is not None  # still controlled, never bare
    assert cut_activations(client, x, "cpu").shape[0] == len(x)


def test_achieved_perturbation_grouped_by_class() -> None:
    fc, x = _setup()
    adv = pgd(SharedMLP(9), x, eps=0.3, iters=3, constraints=fc, device="cpu")
    rep = achieved_perturbation(fc.to_raw(x), fc.to_raw(adv), fc.classes, FEATURES)
    assert set(rep) <= {"IMMUTABLE", "INCREASE_ONLY", "FREE_BOUNDED", "DERIVED"}
    assert rep["IMMUTABLE"]["max"] == pytest.approx(0.0)  # port never moves
    assert "mean" in rep["INCREASE_ONLY"]


def test_activation_inference_leakage_structure_and_control() -> None:
    fc, x = _setup()
    client, _ = SplittableMLP(9).split(2)
    entry = activation_inference(
        client, x[:50], x[50:], device="cpu", seed=0, epochs=20
    )
    assert entry["control"]["name"] == "mean_predictor"
    assert entry["value"] <= 1.0 + 1e-9  # leakage ratio is capped at 1
    assert len(entry["per_feature_leakage_ratio"]) == 9
    assert entry["control"]["mse_baseline"] > 0


def test_gan_evasion_reports_asr_with_realism_control() -> None:
    fc, x = _setup()
    target = SharedMLP(9)
    gen, entry = gan_evasion(
        target, x, fc, device="cpu", seed=0, steps=15, n_generate=40
    )
    assert gen.shape == (40, 9)
    assert 0.0 <= entry["value"] <= 1.0
    ctrl = entry["control"]
    assert ctrl["name"] == "gan_realism"
    assert len(ctrl["per_feature_wasserstein"]) == 9
    assert {"mean_wasserstein", "violation_rate_pre_repair", "mean_nn_distance"} <= set(
        ctrl
    )
    # Repaired output is in-bounds: repairing again changes nothing.
    assert fc.absolute_violation_rate(fc.to_raw(gen)) == pytest.approx(0.0)
