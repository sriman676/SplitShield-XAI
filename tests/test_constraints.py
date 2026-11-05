"""Feature-constraint projection (02_THREAT_MODEL §4) — the control for realism.

Written before any real attack (09_TASKS Task 5). Verifies the perturbation mask
and each per-class rule on the fixture mini-schema, and that DERIVED rates are
recomputed rather than perturbed.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.attack.constraints import FeatureConstraints, resolve_classes

# The fixture mini-schema (config/constraints.MUTABILITY_MAP), same column order.
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


def _build() -> tuple[FeatureConstraints, np.ndarray]:
    rng = np.random.default_rng(0)
    fwd = rng.integers(1, 500, size=50).astype(float)
    bwd = rng.integers(0, 500, size=50).astype(float)
    dur = rng.integers(1, 1_000_000, size=50).astype(float)
    raw = np.column_stack(
        [
            rng.choice([80, 443, 22], size=50).astype(float),
            dur,
            fwd,
            bwd,
            fwd * 20,
            rng.uniform(0, 1e5, size=50),
            rng.uniform(0, 1e5, size=50),
            (fwd + bwd) * 1e6 / dur,
            (fwd + bwd) * 1e6 / dur,
        ]
    )
    scaler = StandardScaler().fit(raw)
    return FeatureConstraints.from_run(FEATURES, scaler, raw), raw


def test_all_features_classified() -> None:
    classes = resolve_classes(FEATURES)
    assert set(classes) == set(FEATURES)


def test_unclassified_feature_fails_loud() -> None:
    with pytest.raises(ValueError, match="Unclassified"):
        resolve_classes(FEATURES + ["Some New Column"])


def test_mask_excludes_immutable_and_derived() -> None:
    fc, _ = _build()
    m = dict(zip(FEATURES, fc.mask, strict=True))
    assert m["Destination Port"] is np.False_ or not m["Destination Port"]  # IMMUTABLE
    assert not m["Flow Bytes/s"] and not m["Flow Packets/s"]  # DERIVED
    assert m["Flow Duration"] and m["Total Fwd Packets"]  # INCREASE_ONLY
    assert m["Flow IAT Mean"] and m["Idle Mean"]  # FREE_BOUNDED


def test_immutable_frozen_and_increase_only_cannot_decrease() -> None:
    fc, raw = _build()
    orig = raw[:1].copy()  # a real integer-valued flow
    adv = orig.copy()
    adv[0, 0] = 9999  # try to change Destination Port (IMMUTABLE)
    adv[0, 2] = orig[0, 2] - 50  # try to REDUCE Total Fwd Packets (INCREASE_ONLY)
    proj = fc.project_raw(adv, orig)
    assert proj[0, 0] == pytest.approx(orig[0, 0])  # port unchanged
    assert proj[0, 2] >= orig[0, 2]  # cannot go below original


def test_free_bounded_clipped_to_training_range() -> None:
    fc, raw = _build()
    orig = raw[:1].copy()
    adv = orig.copy()
    adv[0, 5] = 1e12  # Flow IAT Mean far above observed max
    proj = fc.project_raw(adv, orig)
    assert proj[0, 5] <= fc.hi[5] + 1e-6


def test_integer_features_rounded() -> None:
    fc, raw = _build()
    orig = raw[:1].copy()
    adv = orig.copy()
    adv[0, 2] = orig[0, 2] + 3.7  # add a fractional packet count
    proj = fc.project_raw(adv, orig)
    assert proj[0, 2] == pytest.approx(round(proj[0, 2]))  # integer-valued


def test_derived_recomputed_from_inputs() -> None:
    fc, raw = _build()
    orig = raw[:1].copy()
    adv = orig.copy()
    adv[0, 2] = orig[0, 2] + 100  # pad Total Fwd Packets
    adv[0, 7] = 0.0  # attacker tries to fix Flow Bytes/s directly
    proj = fc.project_raw(adv, orig)
    fwd, bwd, dur = proj[0, 2], proj[0, 3], proj[0, 1]
    expected = (fwd + bwd) * 1e6 / dur if dur > 0 else 0.0
    assert proj[0, 7] == pytest.approx(expected)  # recomputed, not left at 0
    assert proj[0, 7] != 0.0


def test_violation_rate_zero_when_already_valid() -> None:
    fc, raw = _build()
    assert fc.violation_rate(raw, raw) == pytest.approx(0.0)


# --- CIC-IoT-2023 (recency arm) 39-feature schema ---------------------------

from config.constraints_ciciot import MUTABILITY_MAP_CICIOT  # noqa: E402

CIOT_FEATURES = list(MUTABILITY_MAP_CICIOT)


def _build_ciot() -> tuple[FeatureConstraints, np.ndarray]:
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 100, size=(50, len(CIOT_FEATURES)))
    scaler = StandardScaler().fit(raw)
    return FeatureConstraints.from_run(CIOT_FEATURES, scaler, raw), raw


def test_ciot_all_features_classified() -> None:
    classes = resolve_classes(CIOT_FEATURES)
    assert set(classes) == set(CIOT_FEATURES)


def test_ciot_mask_by_class() -> None:
    fc, _ = _build_ciot()
    m = dict(zip(CIOT_FEATURES, fc.mask, strict=True))
    assert not m["Protocol Type"] and not m["TCP"]  # IMMUTABLE protocol one-hots
    assert not m["Rate"] and not m["Variance"]  # DERIVED
    assert m["syn_count"] and m["Number"]  # INCREASE_ONLY
    assert m["Min"] and m["IAT"]  # FREE_BOUNDED


def test_ciot_variance_recomputed_from_std() -> None:
    fc, raw = _build_ciot()
    orig = raw[:1].copy()
    adv = orig.copy()
    col = {n: i for i, n in enumerate(CIOT_FEATURES)}
    adv[0, col["Variance"]] = 0.0  # attacker tries to fix Variance directly
    proj = fc.project_raw(adv, orig)
    assert proj[0, col["Variance"]] == pytest.approx(proj[0, col["Std"]] ** 2)


def test_ciot_rate_is_unknown_derived_and_frozen() -> None:
    fc, raw = _build_ciot()
    assert "Rate" in fc.unknown_derived  # DERIVED with no formula -> frozen+flagged
    orig = raw[:1].copy()
    adv = orig.copy()
    col = {n: i for i, n in enumerate(CIOT_FEATURES)}
    adv[0, col["Rate"]] = 9999.0
    proj = fc.project_raw(adv, orig)
    assert proj[0, col["Rate"]] == pytest.approx(orig[0, col["Rate"]])  # frozen


# --- CIC IoT-DIAD 2024 (84-col CICFlowMeter) schema -------------------------

from config.constraints_diad import MUTABILITY_MAP_DIAD  # noqa: E402

DIAD_FEATURES = list(MUTABILITY_MAP_DIAD)


def _build_diad() -> tuple[FeatureConstraints, np.ndarray]:
    rng = np.random.default_rng(2)
    raw = rng.uniform(0, 100, size=(50, len(DIAD_FEATURES)))
    scaler = StandardScaler().fit(raw)
    return FeatureConstraints.from_run(DIAD_FEATURES, scaler, raw), raw


def test_diad_all_features_classified() -> None:
    assert set(resolve_classes(DIAD_FEATURES)) == set(DIAD_FEATURES)


def test_diad_mask_by_class() -> None:
    fc, _ = _build_diad()
    m = dict(zip(DIAD_FEATURES, fc.mask, strict=True))
    assert not m["Dst Port"]  # IMMUTABLE
    assert not m["Flow Bytes/s"] and not m["Down/Up Ratio"]  # DERIVED
    assert m["Total Fwd Packet"] and m["Subflow Fwd Bytes"]  # INCREASE_ONLY
    assert m["Flow IAT Mean"] and m["Active Mean"]  # FREE_BOUNDED


def test_diad_shared_derived_name_freezes_not_crashes() -> None:
    # "Flow Bytes/s" shares a fixture formula name but DIAD lacks the fixture's
    # input columns -> the guard must freeze it, not raise KeyError.
    fc, raw = _build_diad()
    orig = raw[:1].copy()
    adv = orig.copy()
    col = {n: i for i, n in enumerate(DIAD_FEATURES)}
    adv[0, col["Flow Bytes/s"]] = 12345.0
    proj = fc.project_raw(adv, orig)
    assert proj[0, col["Flow Bytes/s"]] == pytest.approx(orig[0, col["Flow Bytes/s"]])
