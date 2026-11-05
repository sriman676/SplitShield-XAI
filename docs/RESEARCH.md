# 🔬 Research: split learning, explanations and attacks

[← Back to the main README](../README.md)

The registered research half of the project: hypotheses H1–H5, 5 seeds per headline number, every metric paired with a control. The full registered report is `results/FINAL_REPORT.md`.

## 🧪 Research hypotheses (5 seeds, controls applied)

| | Prediction | Registered verdict | Evidence |
|---|---|---|---|
| **H1** | Activation leakage falls as the split cut deepens; accuracy roughly flat | ✅ `SUPPORTED` | leakage 0.981 → 0.958 → 0.911 for cut depth 1→3 (Spearman ρ = −0.64, BH-adjusted p = 0.020); macro-F1 0.85–0.87 |
| **H2** | Adversarial attacks change explanations more than random noise of the same size | ❌ `NOT SUPPORTED` (0 / 5 seeds) | PGD drift 0.865 vs random-matched control 0.967 |
| **H2c** (pre-registered, powered) | Same question, corrected pipeline, 128 instances per seed, new seeds 10–14 | ✅ `SUPPORTED` (5 / 5) | median paired difference +0.020 … +0.051 (small); every seed p < 0.0001 |
| **H3** | Adversarial training cuts attack success at bounded clean cost | ✅ `SUPPORTED` (3 / 5) | PGD success → 0 in every seed; clean-F1 change −0.014 … +0.095 |
| **H4** | Sequential split learning degrades fastest under label skew; SplitFed does not | ✅ `SUPPORTED` | macro-F1 drop from α=10 to α=0.1: split-sequential −0.083, SplitFed −0.007 |
| **H5** | The original CIC-IDS2017 scores higher than its corrected release | ✅ `SUPPORTED` | MLP 0.875 vs 0.853, RF 0.962 vs 0.956 |

**H2, labelled follow-up** (`results/extra/followup_H2.md`; the registered verdict
above stands). The registered runs recomputed derived features on pre-normalised
data, which scrambled every perturbed explanation (drift about 0.9–1.0 for both PGD
and noise). On the corrected runs (verified feature formulas plus a
pre-normalisation guard), explanations are far more stable, and PGD moves them more
than equal-size noise in **4 of 5 seeds** (p < 0.05; PGD drift 0.178 vs control
0.149). The effect is small (median paired difference 0.02–0.05 on a 0–1 scale). Only
2 of 5 seeds pass with a Bonferroni correction, and it does not replicate on raw-unit
CIC-IDS2017 (0 / 5). Read it as a modest, dataset-dependent effect, not a general
law.

**H2b, pre-registered confirmatory test** ([PREREG_H2b](PREREG_H2b.md), committed
`af2555d` before any run; `results/extra/H2b.md`). Same corrected pipeline, **new seeds
5–9**, the registered per-seed rule, verdict = at least 3 of 5 seeds. Result: **NOT
SUPPORTED (2 / 5 seeds)**; Bonferroni (p < 0.01) 1 / 5. PGD drift exceeded the
random-matched control in 4 of 5 seeds (median paired difference −0.009 to +0.033; PGD
0.16–0.22 vs control 0.13–0.20), but only seeds 6 and 7 reached p < 0.05 (seeds 5 and 9:
p = 0.065). Conclusion: on corrected data the effect is at most small and not reliably
detectable with 16 instances per seed. The registered H2 verdict (not supported) and this
confirmatory result agree; the exploratory follow-up (3–4 / 5) did not replicate.

**H2c, pre-registered with a power analysis** ([PREREG_H2c](PREREG_H2c.md), committed before
any run). H2b had only ≈ 35 % power at 16 instances per seed (`scripts/power_h2c.py`), so
H2c repeats the same test and rule on **new seeds 10–14 with 128 instances per seed**
(power ≈ 0.98 even at an effect ⅔ the size H2b suggests). Result (`results/extra/H2c.md`):
**SUPPORTED (5 / 5 seeds)**, Bonferroni (p < 0.01) 5 / 5, every seed p < 0.0001
(Stouffer z = 12.3). PGD drift 0.18–0.25 vs random-matched control 0.16–0.17; median
paired difference +0.020 to +0.051 on the 0–1 scale. So, with enough instances, PGD
changes SHAP explanations more than random noise of the same size, but the effect is
**small** (a few hundredths). H2 (registered, not supported) and H2b (not supported, too
few instances) stay as they are; H2c is the adequately powered test of the same question
on the corrected pipeline.

## 🌐 Detection across datasets (macro-F1, test set, mean over seeds)

| Dataset | RF | GBM | MLP | Split-seq. | FedAvg | SplitFed |
|---|---|---|---|---|---|---|
| CIC-IDS2017 corrected (E4) | 0.956 | 0.846 | 0.853 | 0.827 | 0.711 | 0.701 |
| CIC-IDS2017 original (H5) | 0.962 | 0.985 | 0.875 | 0.843 | 0.726 | 0.726 |
| NF-CSE-CIC-IDS2018-v2 (E8) | 0.980 | 0.980 | 0.979 | 0.978 | 0.977 | 0.977 |
| CIC-IoT-2023 (R1) | 0.690 | 0.684 | 0.612 | 0.577 | 0.593 | 0.595 |
| CIC-IoT-DIAD-2024 (R2) | 0.611 | 0.616 | 0.483 | 0.462 | 0.451 | 0.455 |

Extra datasets, same recipe (model chosen on validation, test reported once, 3 seeds):

| Dataset (native classes) | Ensemble test macro-F1 | Attacks caught | False alarms |
|---|---|---|---|
| NF-ToN-IoT-v2 (10) | **0.916** | 99.90 % | 1.60 % |
| CIC-DDoS2019, day 03-11 (8 present) | **0.880** | 99.99 % | 0.01 % |
| NF-UNSW-NB15-v2 (10) | **0.732** | 99.95 % | 0.52 % |

- ⏱️ **Time-ordered test (T1)**: train on earlier days, test on later traffic with
  attack families never seen in training. The MLP still flags **99.9 %** of the new
  attacks as attacks (vs 62 % for RF), at about 9–10 % false positives. Registered
  macro-F1 0.104 counts every class absent from the test days as 0; over the classes
  actually present it is 0.469 (labelled follow-up).
- 📐 **Fit:** ✅ no real overfitting anywhere. The fixed-size research MLP mildly
  underfits on some datasets (by design of the study); the deployed detector fixes
  that with the ensemble. CIC-IoT-2023 is capped by its data: 17 % of rows carry
  contradictory labels.

## 🔬 Research pipeline (`run_pipeline.py`, stages 0–9)

| Stage | What | Output |
|---|---|---|
| 1 | Preprocess (dedupe, NaN/Inf, class-capped sampling, 70/15/15 split, scaling) + RF/GBM/MLP baselines, automatic fit check | `stage1_baseline.json` |
| 2 | Training schemes: centralized, split-sequential, FedAvg, SplitFed | `stage2_arms.json` |
| 3 | Per-class SHAP, LIME cross-check | `stage3_explain.json` |
| 4 | Attacks with controls: random-matched → FGSM/PGD → GAN → activation inference → client attribution | `stage4_attacks.json` |
| 5 | Cut-depth ablation (H1) | `stage5_cutdepth.json` |
| 6 | Explanation drift vs random-matched control (H2) | `stage6_drift.json` |
| 7 | Defences: adversarial training, activation noise (H3) | `stage7_defense.json` |
| 8 | Cross-dataset transfer (NetFlow) | `stage8_transfer.json` |
| 9 | Aggregation → `results/` | `results/summary.*` |

`run_matrix.py` runs the E3 scheme × label-skew matrix (H4, 60 cells).

## 📏 Rules the code enforces

- ⚖️ **No metric without its control**: leakage, attack success and drift raise
  `MissingControlError` without one.
- 🔗 **Comparable configs only**: runs that differ outside the declared variables never
  share a table (config hash).
- 🎯 **Selection on validation, test reported only.** Blind sets are scored once.
- 🎲 **5 seeds for headline numbers**; `--fast` runs are marked preliminary.
- 📝 **Registered design kept**: corrections are separate, labelled follow-ups, shown
  side by side with the registered numbers (for example, E4 attack success fell from
  about 48 % to 8–13 % once the normalisation artefact was removed).
- 🧪 **Fair rule tests**: rules come from a dev split only; Sigma rules written by a
  test set's authors, or citing it, are excluded; Sysmon `RuleName` is ignored.
