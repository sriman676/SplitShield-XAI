# Pre-registration: H2b

[← Back to RESEARCH](RESEARCH.md)

**Written and committed on 2026-10-23, before any H2b run.** The commit that adds this file
is the registration; the analysis below is fixed by it. Nothing in this file may change
after the runs start. Results are reported whatever they are.

## Why H2b exists

- **H2 (registered, 2026-10-13):** adversarial (PGD) perturbations change a flow's SHAP
  explanation more than random noise of the same per-flow L∞ size. Result: **not
  supported** (0 / 5 seeds); reproduced exactly by both full re-runs. **This verdict stands.**
- After that run, a measurement flaw was found: the registered pipeline recomputed derived
  features on already-normalised data, which scrambled every perturbed explanation (drift
  ≈ 0.9–1.0 for both PGD and noise), so the test could not detect a difference.
- A corrected, exploratory follow-up (`C_E4_cut_depth`, seeds 0–4) was supported in 3–4 of
  5 seeds. Because those seeds were analysed **after** the flaw was known, they cannot
  confirm the effect. H2b tests it on **new seeds that nobody has looked at**.

## Hypothesis

On the corrected pipeline, PGD perturbations change SHAP explanations more than random
noise of the same per-flow L∞ size.

## Design (fixed)

| Item | Value |
|---|---|
| Config | `config/experiments/followup/H2b_E4_cut_depth.yaml`: identical to `C_E4_cut_depth.yaml` except `id`, `description`, `seeds` |
| Data | CIC-IDS2017 (improved release), random split, family9 labels, the same subsampling |
| Corrections | `derived_v2` (verified derived-feature formulas, pre-normalisation guard), `attribution_converge` |
| Seeds | **5, 6, 7, 8, 9** (never run before) |
| Code | the frozen code at this commit; `run_pipeline.py` stage 6 computes the test |
| Stages | 4, 6, 7 as for the follow-up, plus any earlier stage that code needs for a new seed |

## Test and decision rule (fixed)

- **Per seed:** one-sided paired Wilcoxon signed-rank test over the 16 stage-6 instances,
  PGD drift minus random-matched control drift (1 − Spearman). The seed is **supported**
  iff the median paired difference is > 0 **and** p < 0.05 (the registered H2 rule,
  `scripts/followup_h2.seed_rule`).
- **H2b verdict:** **supported** if at least **3 of the 5** seeds are supported (seed
  majority, as for every registered hypothesis); otherwise **not supported**.
- **Reported alongside (not deciding):** the number of seeds supported with a Bonferroni
  correction (p < 0.01), the mean PGD and control drift, and the median paired difference
  per seed.
- A seed whose run fails is re-run once with the same seed; if it fails again it counts
  as **not supported**.

## What will not happen

- No change of statistic, threshold, instance count, seeds, data or code after the first
  H2b run starts.
- No dropping of seeds, no extra seeds, no re-runs to "improve" a result.
- The registered H2 verdict is not replaced: the report shows H2 (registered, not
  supported), the exploratory follow-up, and H2b, each labelled.

## Where the result will be

`results/extra/H2b.json` and `results/extra/H2b.md`, built by
`python -m scripts.followup_h2 --experiment H2b_E4_cut_depth` on the H2b runs only.
