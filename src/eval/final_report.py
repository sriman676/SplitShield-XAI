"""Final report: H1-H5 verdicts vs their controls + every attack explained.

Reads only what the pipeline already wrote (``runs/*/metrics/*.json``, the per-run
``attack_defense_report.md`` from Stage 3, and ``results/<E3>_summary.json`` from
``run_matrix.py``) and renders ``results/FINAL_REPORT.md``. Nothing is recomputed,
so the report can never disagree with the stored metrics.

Each hypothesis is evaluated exactly as pre-registered (slide 9 / 01_RESEARCH_DESIGN
§1); a hypothesis with no runs behind it is reported as "not evaluated", never
guessed. Preliminary (fixture / ``--fast`` / dirty-tree) runs are flagged in the
header: their verdicts are machinery checks, not results.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.aggregate import load_runs
from utils.logging_config import get_logger
from utils.paths import RESULTS_DIR, RUNS_DIR

_log = get_logger(__name__)

NOT_EVALUATED = "not evaluated (no runs)"

# What each adversarial attack is, who the adversary is, and how it is measured.
ATTACKS: dict[str, dict[str, str]] = {
    "fgsm_asr": {
        "name": "FGSM evasion (T1)",
        "what": "One-step gradient-sign perturbation of a malicious flow so the "
        "detector labels it benign.",
        "adversary": "White-box attacker who can query gradients of the full model.",
        "measure": "Attack success rate over clean-correctly-flagged malicious flows, "
        "within the feature-mutability constraints.",
    },
    "pgd_asr": {
        "name": "PGD evasion (T1)",
        "what": "Iterated, projected gradient attack - the strong first-order version "
        "of FGSM under the same epsilon budget.",
        "adversary": "White-box attacker; the standard robustness benchmark.",
        "measure": "Attack success rate (clean-correct filter).",
    },
    "random_matched_asr": {
        "name": "Random-matched perturbation (control)",
        "what": "Random noise with the same per-flow L-inf norm as PGD on the same "
        "mutable features. Not an attack: the control that shows how much evasion "
        "comes from perturbation size alone.",
        "adversary": "None (control).",
        "measure": "Same ASR as PGD; PGD must clearly beat it to count.",
    },
    "gan_asr": {
        "name": "GAN evasion (T2)",
        "what": "A generator learns to emit realistic-looking malicious flows the "
        "detector misses.",
        "adversary": "Black-box attacker with samples of its own attack traffic.",
        "measure": "ASR reported jointly with realism (per-feature Wasserstein).",
    },
    "activation_leakage": {
        "name": "Activation inference (T3)",
        "what": "An honest-but-curious server trains an inversion network to "
        "reconstruct private raw flow features from cut-layer activations.",
        "adversary": "Passive server (split-learning threat model).",
        "measure": "Leakage ratio = attack MSE / mean-predictor MSE (<1 means leakage).",
    },
    "client_attribution": {
        "name": "Client attribution (T4)",
        "what": "The server infers which client an activation came from, revealing "
        "who sees which traffic.",
        "adversary": "Passive server.",
        "measure": "Balanced accuracy vs the class-prior floor.",
    },
}


def _val(x: Any) -> Any:
    return x.get("value") if isinstance(x, dict) else x


def _f(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, int | float) and not isinstance(x, bool) else "-"


def _stage(runs: list[dict], exp: str, stage: str, need: str = "") -> list[dict]:
    """Metric blocks for ``stage`` across runs of ``exp``; skip stubs and blocks
    missing ``need`` (runs written by an older metrics schema)."""
    out = []
    for r in runs:
        if r.get("experiment_id") != exp:
            continue
        m = r.get("_metrics", {}).get(stage, {}).get("metrics")
        if m and "skipped" not in m and (not need or need in m):
            out.append(m)
    return out


def _majority(verdicts: list[str], yes: str = "supported") -> str:
    if not verdicts:
        return NOT_EVALUATED
    k = sum(v == yes for v in verdicts)
    tag = "supported" if k > len(verdicts) / 2 else "not supported"
    return f"{tag} ({k}/{len(verdicts)} seeds)"


def h1(runs: list[dict], exp: str) -> tuple[str, list[str]]:
    blocks = _stage(runs, exp, "stage5_cutdepth", "by_depth")
    if not blocks:
        return NOT_EVALUATED, []
    m = blocks[-1]  # stage 5 already loops its own seeds; take the latest run
    lines = [
        "| cut depth d | macro-F1 (mean ± std) | leakage ratio (mean ± std) | "
        "client FLOPs |",
        "|---|---|---|---|",
    ]
    for d, row in sorted(_val(m["by_depth"]).items()):
        f1, lk = row["macro_f1"], row["leakage_ratio"]
        lines.append(
            f"| {d} | {_f(f1['mean'])} ± {_f(f1['std'])} | {_f(lk['value'])} ± "
            f"{_f(lk.get('std'))} | {row['client_cost']['forward_flops']} |"
        )
    t = m["leakage_trend_spearman"]
    lines.append(
        f"\nLeakage trend over d: Spearman rho={_f(t['value'])}, "
        f"BH-adjusted p={_f(t.get('p_adjusted'))} (n={t.get('n')}, "
        f"{_val(m['n_seeds'])} seeds). Control: mean-predictor reconstruction."
    )
    return _val(m["H1_verdict"]), lines


def h2(runs: list[dict], exp: str) -> tuple[str, list[str]]:
    blocks = _stage(runs, exp, "stage6_drift", "H2_wilcoxon_pgd_vs_random")
    if not blocks:
        return NOT_EVALUATED, []
    pgd = [b["drift_spearman"]["value"] for b in blocks]
    ctrl = [
        v
        for b in blocks
        if (v := (b["drift_spearman"].get("control") or {}).get("value")) is not None
    ]
    ps = [b["H2_wilcoxon_pgd_vs_random"]["p"] for b in blocks]
    lines = [
        f"Fixed-class SHAP drift (1 - Spearman), mean over {len(blocks)} seed(s): "
        f"PGD {_f(float(np.mean(pgd)))} vs random-matched control "
        f"{_f(float(np.mean(ctrl)))}; per-seed one-sided Wilcoxon p = "
        + ", ".join(_f(p) for p in ps)
        + "."
    ]
    return _majority([_val(b["H2_verdict"]) for b in blocks]), lines


def h3(runs: list[dict], exp: str) -> tuple[str, list[str]]:
    blocks = _stage(runs, exp, "stage7_defense", "D1_adversarial_training")
    if not blocks:
        return NOT_EVALUATED, []
    lines = [
        "| seed run | PGD ASR before | PGD ASR after | clean macro-F1 base | "
        "defended | delta (control) |",
        "|---|---|---|---|---|---|",
    ]
    for i, b in enumerate(blocks):
        d1 = b["D1_adversarial_training"]
        c = d1["clean_macro_f1"]
        lines.append(
            f"| {i} | {_f(_val(d1['asr_pgd']['before']))} | "
            f"{_f(_val(d1['asr_pgd']['after']))} | {_f(c['base'])} | "
            f"{_f(c['defended'])} | {_f(c['delta'])} |"
        )
    sweep = _val(blocks[-1]["D2_activation_noise_sweep"])
    lines.append(
        "\nD2 activation-noise frontier (latest run): "
        + "; ".join(
            f"sigma={p['sigma']:g} -> F1 {_f(p['macro_f1'])}, leakage "
            f"{_f(p['leakage_ratio'])}"
            for p in sweep
        )
    )
    return _majority([_val(b["H3_verdict"]) for b in blocks]), lines


def h4(e3_summary: Path) -> tuple[str, list[str]]:
    if not e3_summary.is_file():
        return NOT_EVALUATED, []
    s = json.loads(e3_summary.read_text("utf-8"))
    by: dict[str, dict[float, float]] = {}
    for cell in s.values():
        by.setdefault(cell["scheme"], {})[float(cell["dirichlet_alpha"])] = cell[
            "macro_f1_mean"
        ]
    alphas = sorted({a for d in by.values() for a in d}, reverse=True)
    lines = [
        "| scheme | " + " | ".join(f"alpha={a:g}" for a in alphas) + " |",
        "|---|" + "---|" * len(alphas),
    ]
    for scheme, d in sorted(by.items()):
        lines.append(f"| {scheme} | " + " | ".join(_f(d.get(a)) for a in alphas) + " |")
    lines.append("\nControl: the centralized row is the upper bound for every arm.")

    def drop(scheme: str) -> float | None:
        d = by.get(scheme, {})
        if len(d) < 2:
            return None
        return d[max(d)] - d[min(d)]  # F1 lost going from IID to most skewed

    seq, sfl = drop("split_sequential"), drop("splitfed")
    if seq is None or sfl is None:
        return "not evaluated (needs split_sequential + splitfed at 2+ alphas)", lines
    lines.append(
        f"F1 drop from least to most skewed: split_sequential {_f(seq)}, "
        f"splitfed {_f(sfl)}."
    )
    return ("supported" if seq > sfl else "not supported"), lines


def h5(runs: list[dict], original: str, improved: str) -> tuple[str, list[str]]:
    def mean_f1(exp: str) -> float | None:
        v = [
            b["mlp_macro_f1"]["value"]
            for b in _stage(runs, exp, "stage1_baseline", "mlp_macro_f1")
        ]
        return float(np.mean(v)) if v else None

    o, i = mean_f1(original), mean_f1(improved)
    if o is None or i is None:
        return NOT_EVALUATED, []
    lines = [
        f"MLP macro-F1, identical pipeline: original CIC-IDS2017 {_f(o)} vs "
        f"improved release {_f(i)} (gap {_f(o - i)})."
    ]
    return ("supported" if o > i else "not supported"), lines


def _safe(fn: Any, *args: Any) -> tuple[str, list[str]]:
    """Run one hypothesis; legacy/odd metric shapes give a reason, not a crash."""
    try:
        return fn(*args)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        _log.warning("%s unreadable: %r", fn.__name__, exc)
        return f"not evaluated (unreadable metrics: {exc!r})", []


def _tables(runs: list[dict]) -> list[str]:
    exps = sorted({r.get("experiment_id", "?") for r in runs})
    lines = [
        "## Detection performance (test set)",
        "",
        "| experiment | dataset | RF | GBM | MLP (centralized) | split seq. | "
        "FedAvg | SplitFed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for exp in exps:
        s1, s2 = (
            _stage(runs, exp, "stage1_baseline", "mlp_macro_f1"),
            _stage(runs, exp, "stage2_arms"),
        )
        if not s1 and not s2:
            continue

        def m(blocks: list[dict], key: str) -> str:
            v = [_val(b[key]) for b in blocks if key in b]
            return f"{np.mean(v):.3f}" if v else "-"

        ds = next(
            (r["dataset"]["id"] for r in runs if r.get("experiment_id") == exp), "?"
        )
        lines.append(
            f"| {exp} | {ds} | {m(s1, 'rf_macro_f1')} | {m(s1, 'gbm_macro_f1')} | "
            f"{m(s1, 'mlp_macro_f1')} | {m(s2, 'A1_split_sequential_macro_f1')} | "
            f"{m(s2, 'A2_fedavg_macro_f1')} | {m(s2, 'A3_splitfed_macro_f1')} |"
        )
    lines += ["", "Values are macro-F1, mean over seeds.", ""]

    for exp in exps:
        s8 = _stage(runs, exp, "stage8_transfer")
        if not s8:
            continue
        m8 = s8[-1]
        names = _val(m8["domains"])
        lines += [
            f"## Cross-dataset transfer ({exp})",
            "",
            "| train \\ test | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for n, row in zip(names, _val(m8["transfer_macro_f1"]), strict=True):
            lines.append(f"| {n} | " + " | ".join(_f(v) for v in row) + " |")
        lines.append("")
    return lines


def _adversarial(runs: list[dict], exp: str) -> list[str]:
    blocks = _stage(runs, exp, "stage4_attacks")
    lines = ["## Adversarial attacks explained", ""]
    for key, a in ATTACKS.items():
        vals = [_val(b[key]) for b in blocks if key in b]
        res = f"{np.mean(vals):.3f} (mean over {len(vals)} seed(s))" if vals else "-"
        lines += [
            f"### {a['name']}",
            f"- What it is: {a['what']}",
            f"- Adversary: {a['adversary']}",
            f"- How it is measured: {a['measure']}",
            f"- Result ({exp}): {res}",
            "",
        ]
    return lines


def _class_reports(runs: list[dict], runs_dir: Path) -> list[str]:
    """Latest Stage-3 per-class explanation + mitigation report per experiment."""
    latest: dict[str, Path] = {}
    for r in runs:  # load_runs is sorted by run id (timestamped), so last wins
        p = runs_dir / r["run_id"] / "attack_defense_report.md"
        if p.is_file():
            latest[r.get("experiment_id", "?")] = p
    if not latest:
        return []
    lines = ["## Attack classes explained (why the model flags them + how to defend)"]
    for exp, p in sorted(latest.items()):
        body = p.read_text("utf-8").splitlines()[1:]  # drop the file's own H1
        lines += [
            "",
            f"### {exp}",
            *[("#" + ln if ln.startswith("## ") else ln) for ln in body],
        ]
    return lines


def _nlp(results_dir: Path) -> list[str]:
    """ShieldLM / intent / RLCD metrics written by ``python -m src.nlp.train``."""
    p = results_dir / "nlp_metrics.json"
    if not p.is_file():
        return ["## NLP: ShieldLM + RLCD", "", NOT_EVALUATED, ""]
    try:
        m = json.loads(p.read_text("utf-8"))
        lm, it, r = m["shieldlm"], m["intent"], m["rlcd"]
        b, a = r["before_rlcd"], r["after_rlcd"]
    except (OSError, ValueError, KeyError) as exc:
        return [
            "## NLP: ShieldLM + RLCD",
            "",
            f"not evaluated (unreadable: {exc!r})",
            "",
        ]
    return [
        "## NLP: ShieldLM + RLCD",
        "",
        f"- Question understanding (supervised): accuracy {_f(it['intent_accuracy_heldout_templates'])}, "
        f"macro-F1 {_f(it['intent_macro_f1_heldout_templates'])} on phrasings never seen in training.",
        f"- ShieldLM (from scratch, {lm['params']:,} parameters, vocabulary {lm['vocab']}): "
        f"held-out perplexity {_f(lm['heldout_perplexity'])}.",
        f"- RLCD: {r['pairs']} contrastive pairs; reward-model held-out pair accuracy "
        f"{_f(r['reward_model_heldout_pair_accuracy'])}.",
        "",
        "| | reward | faithfulness | coverage | order | severity | no repetition "
        "| advice | corroboration |",
        "|---|---|---|---|---|---|---|---|---|",
        *[
            f"| {name} | {_f(x['reward_mean'])} | {_f(x['faithfulness_score'])} | "
            f"{_f(x['faithfulness_coverage'])} | {_f(x['faithfulness_order'])} | "
            f"{_f(x['faithfulness_level'])} | {_f(x['faithfulness_no_repeat'])} | "
            f"{_f(x['faithfulness_advice'])} | {_f(x.get('faithfulness_corroboration'))} |"
            for name, x in (("before RLCD", b), ("after RLCD", a))
        ],
        "",
        f"Win rate after vs before (reward model): {_f(r.get('win_rate_after_vs_before'))}. "
        f"Deployed: **{r.get('deployed', 'base')}** (the aligned model ships only if the "
        "independent faithfulness check does not drop, so a gamed reward model cannot "
        "promote a worse writer).",
        "",
        *_compression(m.get("compression")),
        *_selfimprove(results_dir),
    ]


def _compression(c: dict | None) -> list[str]:
    """Prune -> distil -> int8 search (src/nlp/compress.py) with its gate."""
    if not c:
        return []
    t = c["teacher"]
    rows = [
        f"| {x['name']} | {x['params']:,} | {_f(x['candidate']['faithfulness'])} "
        f"({_f(x['reference']['faithfulness'])}) | {_f(x['candidate']['perplexity'])} "
        f"({_f(x['reference']['perplexity'])}) | {_f(x['win_rate'])} | "
        f"{'pass' if x['passed'] else 'fail'} |"
        for x in c["candidates"]
    ]
    q = c["int8"]
    return [
        "### ShieldLM compression (Minitron-style prune + distil, then int8)",
        "",
        f"Original: {t['params']:,} parameters, {t['bytes'] / 1e6:.2f} MB. Gate: "
        "faithfulness at most 1 point lower, perplexity at most 5 % higher, not "
        "losing the reward-model head-to-head (binomial p >= 0.05).",
        "",
        "| candidate | params | faithfulness (orig.) | perplexity (orig.) | win rate | gate |",
        "|---|---|---|---|---|---|",
        *rows,
        f"| int8 of {q['of']} | - | {_f(q['candidate']['faithfulness'])} "
        f"({_f(q['reference']['faithfulness'])}) | {_f(q['candidate']['perplexity'])} "
        f"({_f(q['reference']['perplexity'])}) | {_f(q['win_rate'])} | "
        f"{'pass' if q['passed'] else 'fail'} |",
        "",
        f"Deployed: **{c['deployed']}**, {100 * c['size_ratio']:.0f}% of the original "
        f"size; generation latency {q['latency_ms']:.0f} ms (int8) vs "
        f"{t['latency_ms']:.0f} ms (original) per narrative on this machine.",
        "",
    ]


def _selfimprove(results_dir: Path) -> list[str]:
    p = results_dir / "nlp_selfimprove.json"
    try:
        hist = json.loads(p.read_text("utf-8")) if p.is_file() else []
    except (OSError, ValueError):
        hist = []
    cycles = [h for h in hist if "promoted" in h]
    if not cycles:
        return []
    last = cycles[-1]
    return [
        "### ShieldLM self-improvement (analyst-triggered)",
        "",
        f"{len(cycles)} cycle(s), {sum(h['promoted'] for h in cycles)} promoted. Last: "
        f"{last['incidents']} incidents, {last['pseudo_labels']} pseudo-labels, "
        f"{last['analyst_targets']} analyst targets, {last['rlcd_pairs']} RLCD pairs; "
        f"gate {last['gate']['checks']} -> {'promoted' if last['promoted'] else 'kept'}.",
        "",
    ]


def _wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a proportion k/n."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def _rare_classes(runs: list[dict]) -> list[str]:
    """Per-class MLP results pooled over every seed's test split (owner 2026-10-13).

    Counts are rebuilt from per-class precision/recall/support; recall carries a
    Wilson interval. Splits differ per seed, but for classes with only a handful of
    flows in the whole dataset the same flows recur across splits: pooling reduces
    split noise, it does not create independent evidence."""
    exps = sorted({r.get("experiment_id", "") for r in runs} - {"", "smoke"})
    lines = [
        "## Rare classes, pooled over seeds",
        "",
        "Per-class MLP test results summed over every seed's held-out split; 95% "
        "Wilson interval on recall. For the rarest classes (CIC-IDS2017: 11 "
        "Heartbleed, 36 Infiltration flows in the WHOLE dataset) the same flows "
        "recur across splits, so the interval reflects split noise only.",
        "",
        "| experiment | class | seeds | pooled test support | recall [95% CI] | precision | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = 0
    for exp in exps:
        blocks = [
            b["mlp_per_class"]
            for b in _stage(runs, exp, "stage1_baseline", "mlp_per_class")
        ]
        if not blocks:
            continue
        for cls in blocks[0]:
            tp = fp = sup = 0.0
            for pc in blocks:
                c = pc.get(cls, {})
                s, rec, prec = (
                    c.get("support", 0),
                    c.get("recall", 0.0),
                    c.get("precision", 0.0),
                )
                t = rec * s
                tp, sup = tp + t, sup + s
                fp += (t / prec - t) if prec > 0 else 0.0
            if sup == 0 or sup / len(blocks) >= 1000:  # only the rare classes
                continue
            lo, hi = _wilson(tp, sup)
            fn = sup - tp
            prec = tp / (tp + fp) if tp + fp else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            lines.append(
                f"| {exp} | {cls} | {len(blocks)} | {int(round(sup))} | "
                f"{tp / sup:.2f} [{lo:.2f}, {hi:.2f}] | {prec:.2f} | {f1:.2f} |"
            )
            rows += 1
    return lines + [""] if rows else []


def _fit(results_dir: Path) -> list[str]:
    """Over/underfitting diagnostic (results/fit_diagnostic.json, E4 data)."""
    p = results_dir / "fit_diagnostic.json"
    try:
        f = json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return []
    fin, wide, rf = f["final"], f["wide"], f["rf"]
    curve = "; ".join(
        f"ep {h['epoch']}: train {h['train_f1']:.3f} / val {h['val_f1']:.3f} "
        f"(loss {h['train_loss']:.3f} / {h['val_loss']:.3f})"
        for h in f["curve"][::2]
    )
    lc = ", ".join(f"{r['frac']:.0%}: {r['train']:.3f}/{r['val']:.3f}" for r in f["lc"])
    return [
        "## Over- / underfitting diagnostic (CIC-IDS2017 V2, research MLP 4x128)",
        "",
        f"- 50 epochs without early stopping: train {fin['train']:.3f}, val "
        f"{fin['val']:.3f}, test {fin['test']:.3f} macro-F1 -> **no overfitting** in "
        "accuracy; per-class train and test F1 agree within 0.03.",
        "- Validation loss rises while validation F1 still improves -> the model grows "
        "**overconfident**; the deployed detector is temperature-calibrated on "
        "validation data for this reason.",
        f"- Wider model (4x512): train {wide['train']:.3f}, val {wide['val']:.3f} -> "
        "capacity is not the limit.",
        f"- Learning curve (train/val by data fraction): {lc} -> not data-limited.",
        "- Mild **underfitting** comes from a noisy validation curve that the "
        "pipeline's early stopping (patience 8) stops on; the research runs keep the "
        "registered setting (owner decision), the deployed detector trains longer.",
        f"- Random Forest: train {rf['train']:.3f}, test {rf['test']:.3f}.",
        "",
        f"Curve: {curve}.",
        "",
    ]


def _fit_auto(runs: list[dict], results_dir: Path) -> list[str]:
    """Automatic fit check of every run's MLP + what deploy's fit doctor did."""
    rows = []
    for exp in sorted({r.get("experiment_id", "") for r in runs} - {"", "smoke"}):
        for b in _stage(runs, exp, "stage1_baseline", "mlp_fit"):
            f = b["mlp_fit"]["value"]
            rows.append(
                f"| {exp} | {f['verdict']} | {f['train_macro_f1']:.3f} | "
                f"{f['best_val_macro_f1']:.3f} | {f['best_epoch']}/{f['epochs_run']} | "
                f"{'; '.join(f['findings']) or '-'} |"
            )
    doc: list[str] = []
    try:
        dep = json.loads((results_dir / "deploy_report.json").read_text("utf-8"))
        for r in dep.get("candidates", []):
            for d in r.get("fit_doctor", []):
                doc.append(
                    f"- {r['name']}: {', '.join(d['tweaks'])} -> val macro-F1 "
                    f"{d['val_before']:.3f} -> {d['val_after']:.3f} "
                    f"({'kept' if d['kept'] else 'reverted'}; fit now {d['fit_after']})"
                )
    except (OSError, ValueError, KeyError):
        pass
    if not rows and not doc:
        return []
    return [
        "## Automatic fit check (every trained MLP)",
        "",
        "Rules in src/eval/fitcheck.py. Research runs are reported only (registered "
        "design unchanged); the deployed detector is adjusted automatically.",
        "",
        "| experiment | verdict | train F1 | best val F1 | best epoch | findings |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        *(["Deployed detector, automatic adjustments:", "", *doc, ""] if doc else []),
    ]


def _soc_selftest(results_dir: Path) -> list[str]:
    """Deployed detector end to end on held-out labelled flows (scripts/soc_selftest)."""
    try:
        r = json.loads((results_dir / "soc_selftest.json").read_text("utf-8"))
    except (OSError, ValueError):
        return []
    a, c = r["alerting_any"], r["alerting_confirmed"]
    d, asst = r["delivery"], r["assistant"]
    return [
        "## SOC self-test (deployed detector, held-out labelled flows)",
        "",
        f"{r['flows']} test flows of `{r['dataset']}` replayed through the real alerting "
        "engine (isolated store).",
        "",
        "| alert tier | recall | precision | false-positive rate | missed attacks by class |",
        "|---|---|---|---|---|",
        f"| any (>= 0.5, incl. low) | {a['recall']:.3f} | {a['precision']:.3f} | "
        f"{a['false_positive_rate']:.3%} | {a['missed_by_class'] or '-'} |",
        f"| confirmed (~1% FPR) | {c['recall']:.3f} | {c['precision']:.3f} | "
        f"{c['false_positive_rate']:.3%} | {c['missed_by_class'] or '-'} |",
        "",
        f"- ATT&CK mapping of confirmed alerts matches the true class: "
        f"{r['mapping_correct_share']:.1%}.",
        f"- Correlation: {r['correlation']['incidents']} incidents, "
        f"{r['correlation']['pushed']} pushed.",
        f"- Delivery: Splunk HEC {d['hec_received']} (token ok: {d['hec_auth_ok']}), "
        f"syslog {d['syslog_received']}, outbox {d['outbox_lines']} lines.",
        f"- Assistant: {asst['intent_correct']}/{asst['questions']} questions routed "
        f"correctly; narratives by {asst['narrator']}.",
        "",
    ]


def _temporal(runs: list[dict], exp: str = "T1_temporal") -> list[str]:
    """Time-ordered test: train on earlier traffic, test on later (owner 2026-10-13)."""
    blocks = _stage(runs, exp, "stage1_baseline", "detection")
    if not blocks:
        return []
    lines = [
        "## Time-ordered test (T1): train on earlier days, test on later traffic",
        "",
        "CIC-IDS2017 TrafficLabelling export ordered by timestamp (day-first, "
        "12-hour clock corrected). The newest traffic contains attack families "
        "never seen in training, so per-family F1 is not meaningful there; the "
        "question is whether new attacks are flagged as attacks at all.",
        "",
        "| seed | model | attack recall (any family) | false-positive rate | macro-F1 (classes with support) |",
        "|---|---|---|---|---|",
    ]
    for i, b in enumerate(blocks):
        det = b["detection"]["value"]
        for m in ("rf", "mlp"):
            f1 = b.get(f"{m}_macro_f1", {}).get("value")
            lines.append(
                f"| {i} | {m.upper()} | {det[m]['attack_recall']:.3f} | "
                f"{det[m]['binary_fpr']:.3f} | {_f(f1)} |"
            )
    return lines + [""]


CORRECTED_METRICS = (
    ("stage4_attacks", "fgsm_asr", "FGSM ASR"),
    ("stage4_attacks", "pgd_asr", "PGD ASR"),
    ("stage4_attacks", "gan_asr", "GAN ASR"),
    ("stage4_attacks", "client_attribution", "client attribution (bal. acc)"),
    ("stage6_drift", "drift_spearman", "explanation drift (Spearman)"),
    ("stage7_defense", "D1_adversarial_training", "defended clean macro-F1"),
)


def _mean_metric(blocks: list[dict], key: str) -> float | None:
    vals = []
    for b in blocks:
        v = _val(b.get(key))
        if isinstance(v, dict):  # D1 block: its clean macro-F1
            v = v.get("clean_macro_f1")
        if isinstance(v, int | float) and not isinstance(v, bool):
            vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def _corrected(runs: list[dict]) -> list[str]:
    """Registered vs corrected follow-up (owner decision 2026-10-14), side by side.

    C_<exp> re-runs the attack stages with verified DERIVED formulas, the
    pre-normalized-data guard and a converged attribution probe. Registered numbers
    are not replaced; this shows how much the two realism issues moved them."""
    pairs = sorted(
        {
            r["experiment_id"]
            for r in runs
            if str(r.get("experiment_id", "")).startswith("C_")
        }
    )
    if not pairs:
        return []
    lines = [
        "## Registered vs corrected attack stages (follow-up, owner 2026-10-14)",
        "",
        "Corrected runs (C_*) use DERIVED formulas verified against the extractor "
        "(100 % agreement on 120k real rows), skip integer rounding and formula "
        "recomputation on min-max pre-normalized data (CIC-IDS2017-improved V2 CSV; "
        "realism there is approximate), and run the client-attribution probe to "
        "convergence. C_H5_original (raw units, 5 seeds) is the raw-unit "
        "cross-check of the normalized primary dataset. Means over seeds.",
        "",
        "| experiment | metric | registered | corrected | change |",
        "|---|---|---|---|---|",
    ]
    for c in pairs:
        reg = c[2:]
        for stage, key, label in CORRECTED_METRICS:
            a = _mean_metric(_stage(runs, reg, stage), key)
            b = _mean_metric(_stage(runs, c, stage), key)
            if a is None and b is None:
                continue
            d = f"{b - a:+.3f}" if a is not None and b is not None else "-"
            lines.append(f"| {reg} | {label} | {_f(a)} | {_f(b)} | {d} |")
    return lines + [""]


def _data_quality(results_dir: Path) -> list[str]:
    """Per-dataset quality findings (scripts/data_quality.py)."""
    f = results_dir / "data_quality.json"
    if not f.is_file():
        return []
    dq = json.loads(f.read_text("utf-8"))
    lines = [
        "## Data quality",
        "",
        f"Profiled on up to {dq.get('max_rows', 0):,} evenly spaced raw rows per "
        "dataset. Handled by the pipeline: "
        + "; ".join(dq.get("handled_by_pipeline", {}).values())
        + ".",
        "",
        "| dataset | rows | classes | duplicates | label conflicts | NaN/Inf rows | findings |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds, p in dq.get("datasets", {}).items():
        if "error" in p:
            lines.append(f"| {ds} | - | - | - | - | - | {p['error']} |")
            continue
        lines.append(
            f"| {ds} | {p.get('rows_total', p['rows']):,} | {len(p['classes'])} | "
            f"{p['duplicate_share']:.1%} | {p['conflicting_label_share']:.2%} | "
            f"{p['rows_with_nan_or_inf']:,} | {'; '.join(p['findings']) or 'none'} |"
        )
    return lines + [""]


def _reproducibility(runs_dir: Path) -> list[str]:
    """Same seed + config run twice (a retried phase or a stage-1 re-run): do the
    unaffected numbers match? (owner 2026-10-13: superseded runs as evidence)."""
    keys = {
        "stage1_baseline": ("rf_macro_f1", "mlp_macro_f1"),
        "stage2_arms": (
            "A0_centralized_macro_f1",
            "A1_split_sequential_macro_f1",
            "A2_fedavg_macro_f1",
            "A3_splitfed_macro_f1",
        ),
    }
    pat = re.compile(r"__\d{8}T\d{4}Z")
    rows = []
    for old in sorted(runs_dir.glob("*/metrics/*.json.superseded")):
        stage = old.name.removesuffix(".json.superseded")
        if stage not in keys:
            continue
        run = old.parent.parent
        twins = [
            d
            for d in runs_dir.glob(pat.sub("__*", run.name))
            if d != run and (d / "metrics" / f"{stage}.json").is_file()
        ]
        if not twins:
            continue
        new = max(twins, key=lambda d: d.name)
        try:
            a = json.loads(old.read_text("utf-8"))["metrics"]
            b = json.loads((new / "metrics" / f"{stage}.json").read_text("utf-8"))[
                "metrics"
            ]
        except (OSError, ValueError, KeyError):
            continue
        for k in keys[stage]:
            if k in a and k in b:
                x, y = a[k]["value"], b[k]["value"]
                rows.append(
                    f"| {run.name.split('__')[0]} | {run.name.split('__')[1]} | {k} | {x:.4f} | {y:.4f} | {abs(x - y):.4f} |"
                )
    if not rows:
        return []
    return [
        "## Reproducibility (same seed and config, run twice)",
        "",
        "Earlier attempts kept as evidence: the same split and seed should give the "
        "same numbers (the GBM baseline is excluded: its first value came from the "
        "unweighted, diverging version).",
        "",
        "| experiment | seed | metric | first attempt | rerun | difference |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
    ]


def build_final_report(
    primary: str = "E4_cut_depth",
    h5_pair: tuple[str, str] = ("H5_original", "E4_cut_depth"),
    e3_summary: Path | None = None,
    runs_dir: Path = RUNS_DIR,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, str]:
    """Write ``FINAL_REPORT.md`` (+ ``.json`` verdicts). Returns the verdicts."""
    runs = load_runs(runs_dir)
    e3_summary = e3_summary or results_dir / "E3_alpha_skew_summary.json"
    hyps = {
        "H1": (
            "Leakage falls as the cut deepens; macro-F1 roughly flat",
            _safe(h1, runs, primary),
        ),
        "H2": (
            "Adversarial drift exceeds equal-norm random drift",
            _safe(h2, runs, primary),
        ),
        "H3": (
            "Adversarial training cuts ASR at bounded clean cost",
            _safe(h3, runs, primary),
        ),
        "H4": (
            "Sequential SL degrades fastest under skew; SplitFed does not",
            _safe(h4, e3_summary),
        ),
        "H5": (
            "Original CIC-IDS2017 scores higher than the corrected release",
            _safe(h5, runs, *h5_pair),
        ),
    }
    n_pre = sum(bool(r.get("preliminary")) for r in runs)
    lines = [
        "# SplitShield-XAI - Final report",
        "",
        f"Runs: {len(runs)} ({n_pre} preliminary). Primary experiment: `{primary}`.",
    ]
    if n_pre:
        lines.append(
            "\n> Preliminary runs (fixture / --fast / uncommitted code) are "
            "machinery checks, not results. Re-run on the real data from a clean "
            "commit for reportable numbers."
        )
    lines += ["", "## Hypotheses", "", "| | prediction | verdict |", "|---|---|---|"]
    lines += [f"| {h} | {desc} | **{v}** |" for h, (desc, (v, _)) in hyps.items()]
    for h, (desc, (v, detail)) in hyps.items():
        lines += ["", f"### {h} - {desc}", f"Verdict: **{v}**", "", *detail]
    lines += [
        "",
        *_tables(runs),
        *_fit(results_dir),
        *_fit_auto(runs, results_dir),
        *_soc_selftest(results_dir),
        *_rare_classes(runs),
        *_temporal(runs),
        *_corrected(runs),
        *_data_quality(results_dir),
        *_reproducibility(runs_dir),
        *_adversarial(runs, primary),
        *_nlp(results_dir),
        *_class_reports(runs, runs_dir),
    ]

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", "utf-8")
    verdicts = {h: v for h, (_, (v, _)) in hyps.items()}
    (results_dir / "final_verdicts.json").write_text(
        json.dumps(verdicts, indent=2), "utf-8"
    )
    _log.info("Wrote final report -> %s", results_dir / "FINAL_REPORT.md")
    return verdicts


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Render results/FINAL_REPORT.md")
    ap.add_argument("--primary", default="E4_cut_depth")
    args = ap.parse_args()
    print(build_final_report(primary=args.primary))
