"""Labelled follow-up of H2 on the corrected runs (owner 2026-10-15).

H2 (registered): adversarial (PGD) perturbations change a flow's SHAP explanation
more than random noise of the same per-flow L-inf size (the random-matched
control). Stage 6 of ``run_pipeline.py`` already computes, per seed, a one-sided
paired Wilcoxon over 16 instances (PGD drift - control drift, Spearman measure)
and calls the seed "supported" iff ``median_diff > 0 and p < 0.05``; the headline
verdict is the seed majority (``src.eval.final_report._majority``).

The corrected follow-up (``config/experiments/followup/C_*.yaml``,
``corrections: [derived_v2, attribution_converge]``) re-ran stages 4/6/7, so its
Stage-6 blocks already hold the H2 test. This script only READS run metrics and
applies the SAME rule; it never touches registered results or FINAL_REPORT.

    python -m scripts.followup_h2 --root "<project>"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.eval.final_report import _majority
from utils.paths import PROJECT_ROOT

ALPHA = 0.05  # run_pipeline.stage6: supported iff median_diff > 0 and p < 0.05
LABEL = "follow-up (owner 2026-10-15)"


def seed_rule(p: float, median_diff: float) -> str:
    """The registered per-seed H2 decision (copied from ``run_pipeline.stage6``)."""
    return "supported" if median_diff > 0 and p < ALPHA else "not_supported"


def collect(runs_dir: Path, exp: str) -> list[dict[str, Any]]:
    """Per-seed H2 rows for ``exp`` (runs whose Stage-6 block has the H2 test)."""
    rows: list[dict[str, Any]] = []
    for d in sorted(runs_dir.iterdir()):
        man, mf = d / "manifest.json", d / "metrics" / "stage6_drift.json"
        if not (man.is_file() and mf.is_file()):
            continue
        manifest = json.loads(man.read_text(encoding="utf-8"))
        if manifest.get("experiment_id") != exp:
            continue
        m = json.loads(mf.read_text(encoding="utf-8")).get("metrics", {})
        if "skipped" in m or "H2_wilcoxon_pgd_vs_random" not in m:
            continue
        w, ds = m["H2_wilcoxon_pgd_vs_random"], m["drift_spearman"]
        stored = m["H2_verdict"]["value"]
        recomputed = seed_rule(float(w["p"]), float(w["median_diff"]))
        rows.append(
            {
                "run_id": d.name,
                "seed": manifest.get("seed"),
                "pgd_drift": ds["value"],
                "control_drift": ds["control"]["value"],
                "difference": ds["value"] - ds["control"]["value"],
                "median_paired_diff": w["median_diff"],
                "wilcoxon_stat": w["statistic"],
                "p": w["p"],
                "n_instances": w["n"],
                "test": f"paired Wilcoxon signed-rank, one-sided ({w['alternative']})",
                "verdict": stored,
                "verdict_recomputed_matches": stored == recomputed,
                "explainer": m["explainer"]["value"],
            }
        )
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    mean = (lambda k: sum(r[k] for r in rows) / n) if n else (lambda k: None)
    return {
        "n_seeds": n,
        "mean_pgd_drift": mean("pgd_drift"),
        "mean_control_drift": mean("control_drift"),
        "seeds_pgd_above_control": sum(r["difference"] > 0 for r in rows),
        "verdict": _majority([r["verdict"] for r in rows]),
    }


def _f(x: Any, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if isinstance(x, int | float) else "-"


def _table(reg: list[dict[str, Any]], fol: list[dict[str, Any]]) -> list[str]:
    out = [
        "| Seed | Run | PGD drift | Random-matched drift | Difference | Median paired diff | p (one-sided Wilcoxon, n=16) | Seed verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag, rows in (("registered", reg), ("follow-up", fol)):
        for r in sorted(rows, key=lambda r: r["seed"]):
            v = "✅ supported" if r["verdict"] == "supported" else "❌ not supported"
            out.append(
                f"| {r['seed']} | {tag} | {_f(r['pgd_drift'])} | {_f(r['control_drift'])} "
                f"| {r['difference']:+.3f} | {r['median_paired_diff']:+.3f} | "
                f"{_f(r['p'], 4)} | {v} |"
            )
    return out


def render_md(res: dict[str, Any]) -> str:
    reg, fol = res["registered"], res["followup"]
    sr, sf = reg["summary"], fol["summary"]
    xr, xf = (
        res["raw_unit_crosscheck"]["registered"],
        res["raw_unit_crosscheck"]["followup"],
    )
    lines = [
        f"# 🔁 Follow-up: H2 explanation drift on the corrected runs ({LABEL.split('(')[1][:-1]})",
        "",
        "_Registered verdict kept unchanged in FINAL_REPORT; this is a separately "
        f"labelled {LABEL}. No new training was needed: the corrected follow-up runs "
        "(C_E4_cut_depth, 5 seeds) already contain Stage 6 with the H2 test._",
        "",
        "## 🧪 What H2 asks",
        "",
        "When an attacker nudges a malicious flow with a PGD attack, does the model's "
        "explanation (which features SHAP says matter for the true class) change *more* "
        "than when the same flow is nudged by random noise of exactly the same size on "
        "the same features? If yes, the attack specifically targets what the explanation "
        "relies on; if no, any explanation change is just a size-of-perturbation effect.",
        "",
        "Drift = 1 - Spearman rank correlation between the clean and perturbed "
        "attributions (0 = identical ranking, 1 = unrelated). Decision rule, identical "
        "for registered and follow-up: per seed, one-sided paired Wilcoxon over 16 "
        "clean-correct malicious flows; a seed supports H2 iff median(PGD - control) > 0 "
        f"and p < {ALPHA}; H2 is supported if a majority of seeds support it.",
        "",
        "## 📊 Verdicts",
        "",
        "| | Registered (E4_cut_depth) | Follow-up (C_E4_cut_depth) |",
        "|---|---|---|",
        f"| Verdict | **{sr['verdict']}** | **{sf['verdict']}** |",
        f"| Mean PGD drift | {_f(sr['mean_pgd_drift'])} | {_f(sf['mean_pgd_drift'])} |",
        f"| Mean random-matched drift | {_f(sr['mean_control_drift'])} | {_f(sf['mean_control_drift'])} |",
        f"| Seeds with PGD drift > control | {sr['seeds_pgd_above_control']}/{sr['n_seeds']} | {sf['seeds_pgd_above_control']}/{sf['n_seeds']} |",
        "",
        "## 🔬 Per seed",
        "",
        *_table(reg["rows"], fol["rows"]),
        "",
        "## 🧭 Raw-unit cross-check (supplementary, not the H2 experiment)",
        "",
        "The primary dataset (CIC-IDS2017-improved) ships min-max normalised, so attack "
        "realism there is approximate. The same Stage-6 test on original CIC-IDS2017 in "
        "raw units (H5_original):",
        "",
        "| | Registered (H5_original) | Follow-up (C_H5_original) |",
        "|---|---|---|",
        f"| Verdict | {xr['summary']['verdict']} | {xf['summary']['verdict']} |",
        f"| Mean PGD / control drift | {_f(xr['summary']['mean_pgd_drift'])} / {_f(xr['summary']['mean_control_drift'])} | {_f(xf['summary']['mean_pgd_drift'])} / {_f(xf['summary']['mean_control_drift'])} |",
        f"| Per-seed p | {', '.join(_f(r['p'], 3) for r in xr['rows'])} | {', '.join(_f(r['p'], 3) for r in xf['rows'])} |",
        "",
        "## 💡 What this means",
        "",
        *[f"- {s}" for s in res["interpretation"]],
        "",
        "## ⚠️ Caveats",
        "",
        *[f"- {s}" for s in res["caveats"]],
        "",
    ]
    return "\n".join(lines)


INTERPRETATION = [
    "Registered runs: PGD and random noise both scrambled the explanation almost "
    "completely (drift ~0.9-1.0, close to an unrelated ranking), and noise scrambled it "
    "MORE than PGD on average (4 of 5 seeds), so H2 failed. Such large drift from "
    "noise most likely reflects the registered constraint projection recomputing "
    "DERIVED features on pre-normalised data, which distorts any perturbed flow "
    "regardless of direction (inference, not separately tested).",
    "Corrected follow-up (verified formulas + pre-normalisation guard): explanations "
    "are far more stable (drift ~0.15-0.2) and PGD now moves them a little more than "
    "equal-size noise in 4 of 5 seeds (p < 0.05), so by the same rule H2 is supported "
    "on the primary experiment.",
    "The effect is small in absolute terms (median paired difference ~0.02-0.05 on a "
    "0-1 scale): the attack does steer the explanation beyond what its size explains, "
    "but only modestly. It does not replicate in the raw-unit cross-check "
    "(C_H5_original, 0/5 seeds), so it should be reported as a dataset-dependent, "
    "modest effect, not a general law.",
]

CAVEATS = [
    "Follow-up only; the registered H2 verdict (not supported, 0/5) stands as the "
    "pre-registered result. Corrections changed the attack's feature-constraint "
    "projection, so the follow-up is a different (corrected) measurement, not a re-test "
    "of the same one.",
    "n = 16 instances per seed; per-seed p-values are not multiplicity-corrected "
    "(same as the registered rule). With Bonferroni over 5 seeds (alpha 0.01) only "
    "seeds 1 and 2 would pass (2/5).",
    "SHAP via GradientExplainer; its cross-check against KernelSHAP is weak "
    "(Spearman 0.16-0.35 in both registered and follow-up), so attributions are "
    "themselves noisy.",
    "Corrected PGD succeeds far less often (mean ASR 0.127 vs 0.478 registered), so "
    "most attacked flows keep their class; drift is measured at the fixed true class.",
    "GAN drift is not part of H2 (no paired clean/control counterpart), as registered.",
]


def build(runs_dir: Path) -> dict[str, Any]:
    def block(exp: str) -> dict[str, Any]:
        rows = collect(runs_dir, exp)
        return {"experiment": exp, "rows": rows, "summary": summarise(rows)}

    return {
        "label": LABEL,
        "hypothesis": "H2: adversarial (PGD) explanation drift exceeds equal-norm "
        "random-matched drift",
        "decision_rule": {
            "statistic": "fixed-class SHAP drift, 1 - Spearman, per instance",
            "control": "random_matched (same per-instance L-inf norm as PGD, same "
            "mutable features)",
            "test": "paired Wilcoxon signed-rank, one-sided greater, PGD - control",
            "seed_supported_if": f"median_diff > 0 and p < {ALPHA}",
            "aggregate": "majority of seeds (src.eval.final_report._majority)",
            "source": "run_pipeline.stage6 (values read from runs/*/metrics/"
            "stage6_drift.json; per-seed verdict re-derived and checked)",
        },
        "registered": block("E4_cut_depth"),
        "followup": block("C_E4_cut_depth"),
        "raw_unit_crosscheck": {
            "registered": block("H5_original"),
            "followup": block("C_H5_original"),
        },
        "interpretation": INTERPRETATION,
        "caveats": CAVEATS,
    }


def h2b(runs_dir: Path, exp: str = "H2b_E4_cut_depth") -> dict[str, Any]:
    """Pre-registered H2b / H2c (docs/PREREG_<name>.md, name = the experiment id's prefix):
    the registered per-seed rule, seed majority (>= 3 of 5); Bonferroni (p < 0.01), drifts
    and a Stouffer combination over seeds reported alongside, not deciding."""
    from scipy.stats import norm

    rows = collect(runs_dir, exp)
    name = exp.split("_")[0]
    n_sup = sum(1 for r in rows if seed_rule(r["p"], r["median_paired_diff"]) == "supported")
    z = sum(float(norm.isf(r["p"])) for r in rows) / max(1, len(rows)) ** 0.5
    return {"registration": f"docs/PREREG_{name}.md (committed before any {name} run)", "experiment": exp,
            "stouffer_z": round(z, 3), "stouffer_p": float(norm.sf(z)),
            "seeds": [r["seed"] for r in rows], "rows": rows,
            "seeds_supported": n_sup, "seeds_supported_bonferroni": sum(1 for r in rows
                if r["median_paired_diff"] > 0 and r["p"] < ALPHA / 5),
            "verdict": ("supported" if n_sup >= 3 else "not supported") + f" ({n_sup} / {len(rows)} seeds)",
            "complete": len(rows) == 5}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--experiment", default=None, help="H2b_E4_cut_depth / H2c_E4_cut_depth: pre-registered")
    args = ap.parse_args(argv)
    if args.experiment:
        res2 = h2b(args.root / "runs", args.experiment)
        out = args.root / "results" / "extra"
        out.mkdir(parents=True, exist_ok=True)
        name = args.experiment.split("_")[0]
        (out / f"{name}.json").write_text(json.dumps(res2, indent=2), encoding="utf-8")
        lines = [f"# {name} (pre-registered, docs/PREREG_{name}.md)", "", f"**Verdict: {res2['verdict']}**; "
                 f"Bonferroni p < 0.01: {res2['seeds_supported_bonferroni']} / {len(res2['rows'])}.", "",
                 "| Seed | PGD drift | Control drift | Median paired diff | p | Seed verdict |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['seed']} | {r['pgd_drift']:.3f} | {r['control_drift']:.3f} | {r['median_paired_diff']:.3f} | "
                  f"{r['p']:.4f} | {seed_rule(r['p'], r['median_paired_diff'])} |" for r in res2["rows"]]  # fmt: skip
        (out / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
        print(res2["verdict"])
        return 0
    res = build(args.root / "runs")
    out = args.root / "results" / "extra"
    out.mkdir(parents=True, exist_ok=True)
    (out / "followup_H2.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "followup_H2.md").write_text(render_md(res), encoding="utf-8")
    print(
        f"registered: {res['registered']['summary']['verdict']} | "
        f"follow-up: {res['followup']['summary']['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
