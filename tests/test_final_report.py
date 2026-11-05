"""Final report: verdicts come from stored metrics; missing evidence => not evaluated."""

from __future__ import annotations

import json
from pathlib import Path

from run_pipeline import _parse_stage
from src.eval.final_report import NOT_EVALUATED, build_final_report


def _run(runs: Path, rid: str, exp: str, metrics: dict) -> None:
    d = runs / rid
    (d / "metrics").mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": rid,
                "experiment_id": exp,
                "seed": 0,
                "preliminary": True,
                "dataset": {"id": "x"},
                "config_resolved": {},
            }
        )
    )
    for stage, m in metrics.items():
        (d / "metrics" / f"{stage}.json").write_text(json.dumps({"metrics": m}))


def test_verdicts_from_stored_metrics(tmp_path: Path) -> None:
    runs, res = tmp_path / "runs", tmp_path / "results"
    f1 = lambda v: {"stage1_baseline": {"mlp_macro_f1": {"value": v}}}  # noqa: E731
    _run(runs, "a", "H5_original", f1(0.95))
    _run(
        runs,
        "b",
        "E4_cut_depth",
        {
            **f1(0.90),
            "stage6_drift": {
                "drift_spearman": {"value": 0.4, "control": {"value": 0.1}},
                "H2_wilcoxon_pgd_vs_random": {"p": 0.01},
                "H2_verdict": {"value": "supported"},
            },
        },
    )
    res.mkdir()
    (res / "e3.json").write_text(
        json.dumps(
            {
                f"{s}|a{a}": {"scheme": s, "dirichlet_alpha": a, "macro_f1_mean": v}
                for s, a, v in [
                    ("split_sequential", 10.0, 0.9),
                    ("split_sequential", 0.1, 0.5),
                    ("splitfed", 10.0, 0.9),
                    ("splitfed", 0.1, 0.85),
                ]
            }
        )
    )

    v = build_final_report(e3_summary=res / "e3.json", runs_dir=runs, results_dir=res)

    assert v["H5"] == "supported"  # 0.95 original > 0.90 improved
    assert v["H4"] == "supported"  # seq drops 0.4, splitfed 0.05
    assert v["H2"].startswith("supported (1/1")
    assert v["H1"] == NOT_EVALUATED and v["H3"] == NOT_EVALUATED  # no runs => no guess
    assert "Adversarial attacks explained" in (res / "FINAL_REPORT.md").read_text(
        "utf-8"
    )


def test_parse_stage_accepts_lists() -> None:
    assert _parse_stage("all") == list(range(10))
    assert _parse_stage("1,2,8") == [1, 2, 8]


def test_wilson_interval_and_reproducibility_table(tmp_path) -> None:
    import json

    from src.eval.final_report import _reproducibility, _wilson

    lo, hi = _wilson(9, 10)
    assert 0.55 < lo < 0.6 and 0.98 < hi <= 1.0  # textbook 9/10 -> [0.596, 0.982]
    for name, rf, sup in (
        ("E4_cut_depth__seed0__20261013T1407Z__h", 0.964, True),
        ("E4_cut_depth__seed0__20261014T0300Z__h", 0.964, False),
    ):
        m = tmp_path / name / "metrics"
        m.mkdir(parents=True)
        f = m / ("stage1_baseline.json.superseded" if sup else "stage1_baseline.json")
        f.write_text(json.dumps({"metrics": {"rf_macro_f1": {"value": rf}}}), "utf-8")
    out = "\n".join(_reproducibility(tmp_path))
    assert "| E4_cut_depth | seed0 | rf_macro_f1 | 0.9640 | 0.9640 | 0.0000 |" in out
