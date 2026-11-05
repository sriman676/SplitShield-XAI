"""Corrected follow-up (owner 2026-10-14): registered behaviour unchanged, verified
DERIVED formulas, pre-normalized guard, self-increasing probe iterations, report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from config.base import build_config, config_hash, load_experiment
from scripts.backup_deploy import backup
from src.attack.client_attribution import fit_converged
from src.attack.constraints import FeatureConstraints, canonical, is_prenormalized
from src.eval.final_report import _corrected

ORIG = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Down/Up Ratio",
]


def _raw(n: int = 50) -> np.ndarray:
    rng = np.random.default_rng(0)
    dur = rng.integers(1, 10_000_000, n).astype(float)
    fp = rng.integers(1, 50, n).astype(float)
    bp = rng.integers(0, 50, n).astype(float)
    fb, bb = fp * 60, bp * 900
    return np.c_[dur, fp, bp, fb, bb, (fb + bb) * 1e6 / dur, np.floor(bp / fp)]


def test_registered_hash_unchanged_and_followup_distinct() -> None:
    reg = build_config(load_experiment("E4_cut_depth.yaml"), seed=0)
    assert reg.corrections == ()
    assert config_hash(reg) == "b11831f13482"  # existing E4 run ids
    c = build_config(load_experiment("followup/C_E4_cut_depth.yaml"), seed=0)
    assert c.experiment_id == "C_E4_cut_depth" and "derived_v2" in c.corrections
    assert config_hash(c) != config_hash(reg)


def test_canonical_folds_schemas() -> None:
    assert canonical(" Total Fwd Packets") == canonical("Total_Fwd_Packets")
    assert canonical("Total Bwd packets") == "total_backward_packets"
    assert canonical("Fwd Segment Size Avg") == "avg_fwd_segment_size"


def test_corrected_recomputes_bytes_rate_from_bytes() -> None:
    x = _raw()
    sc = StandardScaler().fit(x)
    adv = x.copy()
    adv[:, 3] += 600  # attacker pads forward bytes
    reg = FeatureConstraints.from_run(ORIG, sc, x)
    cor = FeatureConstraints.from_run(ORIG, sc, x, corrected=True)
    want = (adv[:, 3] + adv[:, 4]) * 1e6 / adv[:, 0]
    got = cor.project_raw(adv, x)
    np.testing.assert_allclose(got[:, 5], want)  # bytes/s from BYTES
    np.testing.assert_allclose(got[:, 6], np.floor(x[:, 2] / x[:, 1]))
    # the registered (fixture) formula used packets here; kept for registered runs
    assert not np.allclose(reg.project_raw(adv, x)[:, 5], want)
    np.testing.assert_allclose(cor.project_raw(x, x)[:, 5], x[:, 5])  # consistent


def test_prenormalized_guard_skips_rounding_and_recompute() -> None:
    x = _raw()
    xn = (x - x.min(0)) / (x.max(0) - x.min(0))
    assert is_prenormalized(xn) and not is_prenormalized(x)
    sc = StandardScaler().fit(xn)
    cor = FeatureConstraints.from_run(ORIG, sc, xn, corrected=True)
    assert cor.normalized and not cor.integer_cols.any()
    adv = xn.copy()
    adv[:, 3] = np.clip(adv[:, 3] + 0.01, 0, 1)
    out = cor.project_raw(adv, xn)
    assert out.max() <= 1 + 1e-9  # no out-of-range "recomputed" rates
    assert not np.allclose(out[:, 3], np.rint(out[:, 3]))  # tiny values kept
    reg = FeatureConstraints.from_run(ORIG, sc, xn)
    assert not reg.normalized  # registered behaviour untouched


def test_probe_iterations_increase_until_converged() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 20)) * rng.uniform(1, 1000, 20)
    y = (a[:, 0] + a[:, 1] > 0).astype(int) + (a[:, 2] > 0)
    clf = fit_converged(a, y, seed=0, start=1)
    assert clf.max_iter_used > 1  # grew automatically from a too-small start
    assert clf.score(a, y) > 0.7


def test_backup_never_overwrites(tmp_path: Path) -> None:
    dep = tmp_path / "artifacts" / "deploy"
    dep.mkdir(parents=True)
    (dep / "columns.json").write_text(json.dumps({"dataset": "d"}))
    (dep / "model.pt").write_text("v1")
    b = backup(dep, tmp_path / "bk")
    assert b is not None and (b / "deploy" / "model.pt").read_text() == "v1"
    (dep / "model.pt").write_text("v2")
    backup(dep, tmp_path / "bk")
    assert (b / "deploy" / "model.pt").read_text() == "v1"


def test_report_side_by_side() -> None:
    def run(exp: str, asr: float) -> dict:
        m = {"pgd_asr": {"value": asr}}
        return {
            "experiment_id": exp,
            "_metrics": {"stage4_attacks": {"metrics": m}},
        }

    lines = _corrected([run("E4_cut_depth", 0.0), run("C_E4_cut_depth", 0.25)])
    row = next(ln for ln in lines if "PGD ASR" in ln)
    assert "0.000" in row and "0.250" in row and "+0.250" in row
    assert _corrected([run("E4_cut_depth", 0.0)]) == []
