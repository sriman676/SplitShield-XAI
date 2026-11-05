# Pre-registration: H2c

[← Back to RESEARCH](RESEARCH.md)

**Written and committed on 2026-10-24, before any H2c run.** The commit that adds this file
is the registration; the analysis below is fixed by it. Nothing in this file may change
after the runs start. Results are reported whatever they are.

## Why H2c exists

- **H2 (registered, 2026-10-13):** not supported (0 / 5 seeds). **This verdict stands.**
- **H2b (pre-registered, [PREREG_H2b](PREREG_H2b.md)):** not supported (2 / 5 seeds). **This
  verdict stands.** In all 5 H2b seeds the mean PGD drift was above the mean control drift
  (+0.012 to +0.036); the median paired difference was positive in 4 of 5 (−0.009 to
  +0.033). With 16 instances per seed only 2 seeds reached p < 0.05.
- **Power analysis** (`scripts/power_h2c.py`, `results/extra/H2c_power.json`, using H2b
  as a pilot only): the per-seed effect size estimated from H2b is d ≈ 0.39 (z / √n). At 16
  instances the registered rule has only ≈ 35 % power to give "supported" even if the
  effect is real, so H2b could not answer the question. At **128 instances** the power of
  the verdict is ≈ 1.00 at d = 0.39 and ≈ 0.98 at a pessimistic d = 0.25.

H2c asks the same question with enough instances to answer it, on seeds nobody has looked at.

## Hypothesis

On the corrected pipeline, PGD perturbations change SHAP explanations more than random
noise of the same per-flow L∞ size.

## Design (fixed)

| Item | Value |
|---|---|
| Config | `config/experiments/followup/H2c_E4_cut_depth.yaml`: identical to `H2b_E4_cut_depth.yaml` except `id`, `description`, `seeds` and `training.drift_instances` |
| Instances | **128** per seed (`training.drift_instances`; H2 / H2b used 16), chosen the same way as before: the first clean-correct, non-benign test flows |
| Data | CIC-IDS2017 (improved release), random split, family9 labels, the same subsampling |
| Corrections | `derived_v2`, `attribution_converge` (as H2b) |
| Seeds | **10, 11, 12, 13, 14** (never run before) |
| Code | the frozen code at this commit; `run_pipeline.py` stage 6 computes the test |
| Stages | 4, 6, 7 as for H2b, plus any earlier stage that code needs for a new seed |

## Test and decision rule (fixed, the same as H2 and H2b)

- **Per seed:** one-sided paired Wilcoxon signed-rank test over the stage-6 instances,
  PGD drift minus random-matched control drift (1 − Spearman). The seed is **supported**
  iff the median paired difference is > 0 **and** p < 0.05 (`scripts/followup_h2.seed_rule`).
- **H2c verdict:** **supported** if at least **3 of the 5** seeds are supported; otherwise
  **not supported**.
- **Reported alongside (not deciding):** seeds supported with Bonferroni (p < 0.01); a
  Stouffer combination of the 5 per-seed p-values; mean PGD and control drift and the
  median paired difference per seed; the instance count actually used per seed.
- A seed whose run fails is re-run once with the same seed; if it fails again it counts
  as **not supported**. A seed with fewer than 128 eligible instances uses all it has and
  the count is reported.

## What will not happen

- No change of statistic, threshold, instance count, seeds, data or code after the first
  H2c run starts.
- No dropping of seeds, no extra seeds, no re-runs to "improve" a result.
- H2 and H2b are not replaced: the report shows H2 (registered), the exploratory follow-up,
  H2b and H2c, each labelled. If H2c is supported, the statement is: *with adequate power,
  a small effect (median paired difference of a few hundredths on a 0–1 scale) was found
  on CIC-IDS2017 with the corrected pipeline*; it does not turn H2 or H2b into "supported".

## Where the result will be

`results/extra/H2c.json` and `results/extra/H2c.md`, built by
`python -m scripts.followup_h2 --experiment H2c_E4_cut_depth` on the H2c runs only.
