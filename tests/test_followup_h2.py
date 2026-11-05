"""H2 follow-up (owner 2026-10-15): same per-seed rule and majority as registered."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.followup_h2 import build, main, seed_rule


def _run(root: Path, exp: str, seed: int, p: float, md: float, pgd: float) -> None:
    d = root / "runs" / f"{exp}__seed{seed}__x"
    (d / "metrics").mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"experiment_id": exp, "seed": seed}), encoding="utf-8"
    )
    m = {
        "metrics": {
            "explainer": {"value": "GradientExplainer"},
            "drift_spearman": {"value": pgd, "control": {"value": 0.1}},
            "H2_wilcoxon_pgd_vs_random": {
                "statistic": 1.0,
                "p": p,
                "median_diff": md,
                "n": 16,
                "alternative": "greater",
            },
            "H2_verdict": {"value": seed_rule(p, md)},
        }
    }
    (d / "metrics" / "stage6_drift.json").write_text(json.dumps(m), encoding="utf-8")


def test_seed_rule_matches_stage6() -> None:
    assert seed_rule(0.01, 0.02) == "supported"
    assert seed_rule(0.01, -0.02) == "not_supported"
    assert seed_rule(0.05, 0.02) == "not_supported"


def test_majority_and_outputs(tmp_path: Path) -> None:
    for s, p in enumerate([0.01, 0.02, 0.9]):
        _run(tmp_path, "C_E4_cut_depth", s, p, 0.03 if p < 0.5 else -0.01, 0.2)
    _run(tmp_path, "E4_cut_depth", 0, 0.9, -0.1, 0.8)
    res = build(tmp_path / "runs")
    assert res["followup"]["summary"]["verdict"] == "supported (2/3 seeds)"
    assert res["registered"]["summary"]["verdict"] == "not supported (0/1 seeds)"
    assert all(r["verdict_recomputed_matches"] for r in res["followup"]["rows"])
    assert main(["--root", str(tmp_path)]) == 0
    md = (tmp_path / "results" / "extra" / "followup_H2.md").read_text("utf-8")
    assert "follow-up (owner 2026-10-15)" in md


def test_preregistered_h2c_outputs(tmp_path: Path) -> None:
    for s, p in zip(range(10, 15), [0.01, 0.02, 0.03, 0.2, 0.001], strict=True):
        _run(tmp_path, "H2c_E4_cut_depth", s, p, 0.03, 0.2)
    assert main(["--root", str(tmp_path), "--experiment", "H2c_E4_cut_depth"]) == 0
    res = json.loads((tmp_path / "results" / "extra" / "H2c.json").read_text("utf-8"))
    assert res["verdict"] == "supported (4 / 5 seeds)"
    assert res["registration"].startswith("docs/PREREG_H2c.md")
    assert res["stouffer_p"] < 0.001
    assert (tmp_path / "results" / "extra" / "H2c.md").read_text("utf-8").startswith("# H2c")
