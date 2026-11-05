"""E3 matrix driver: runs a small sweep on the fixture, then resumes it.

Kept tiny (2 schemes × 1 α × 1 seed, --fast) so it stays a smoke, not a real run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import run_matrix as rm
import utils.paths
from run_matrix import run_matrix

_OVERRIDE = {
    "scheme": ["centralized", "split_sequential"],
    "dirichlet_alpha": [1.0],
    "seeds": [0],
}


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # never write the fixture smoke into the real results/ or runs/ (it once
    # overwrote results/E3_alpha_skew_summary.json during the engine's pre-checks)
    monkeypatch.setattr(rm, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(rm, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(utils.paths, "RUNS_DIR", tmp_path / "runs")


def test_matrix_runs_then_resumes() -> None:
    r1 = run_matrix(
        "E3_alpha_skew",
        fast=True,
        resume=False,
        matrix_override=_OVERRIDE,
        dataset="fixture",
    )
    assert r1["n_run"] == 2
    assert "centralized|a1.0" in r1["summary"]
    assert r1["summary"]["centralized|a1.0"]["n_seeds"] == 1

    # second pass with --resume must skip both finished cells and still report them
    r2 = run_matrix(
        "E3_alpha_skew",
        fast=True,
        resume=True,
        matrix_override=_OVERRIDE,
        dataset="fixture",
    )
    assert r2["n_run"] == 0
    assert r2["n_skipped"] == 2
    assert "split_sequential|a1.0" in r2["summary"]


def test_matrix_never_touches_real_results() -> None:
    run_matrix(
        "E3_alpha_skew",
        fast=True,
        resume=False,
        matrix_override=_OVERRIDE,
        dataset="fixture",
    )
    assert rm.RESULTS_DIR != utils.paths.PROJECT_ROOT / "results"
