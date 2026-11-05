"""Pipeline entrypoint.

Usage:
    python run_pipeline.py --fast --stage all           # smoke path
    python run_pipeline.py --stage 5 --experiment E4_cut_depth

Each stage (0-9) carries its real logic and writes a manifest + a schema-shaped
metrics JSON. When a dataset is absent the stage writes a skip stub (the
``except FileNotFoundError/KeyError`` branch) so the smoke path stays green.
``--stage`` is validated against an enum and never passed to ``eval`` or used to
build an import path (08_ENGINEERING §4).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

from config.base import Config, build_config, load_experiment
from utils.device import get_compute_tier, get_primary_device
from utils.logging_config import configure_logging, get_logger
from utils.manifest import make_run_id, write_manifest
from utils.paths import run_dir
from utils.seed import set_seed

_log = get_logger(__name__)

STAGES = tuple(range(10))


def _write_metrics(rid: str, name: str, metrics: dict[str, Any]) -> None:
    """Write a stage metrics JSON following the 06_EXPERIMENTS §4 shape."""
    payload = {
        "run_id": rid,
        "metrics": metrics,
        "deviations": [],
        "preliminary": True,  # stub output is always preliminary
    }
    path = run_dir(rid) / "metrics" / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


_RAW_CACHE: dict[str, Any] = {}


def _raw(spec: Any) -> Any:
    """``load_raw`` once per dataset per process. Every stage (and stage 5, 15 times)
    re-preprocesses the same raw frame; re-reading it (63 CSVs for CIC-IoT-2023)
    each time dominated wall-clock. ``preprocess_frame`` never mutates its input.
    Single entry, so at most one raw frame is held."""
    from src.data.sources import load_raw

    if spec.id not in _RAW_CACHE:
        _RAW_CACHE.clear()
        _RAW_CACHE[spec.id] = load_raw(spec)
    return _RAW_CACHE[spec.id]


# every Config field preprocess_frame reads (tests/test_prep_memo.py keeps this list
# in step with src/data/preprocess.py)
PREP_CFG_FIELDS = (
    "split",
    "subsample.rare_floor",
    "subsample.class_cap",
    "training.keep_dest_port",
    "training.scaler",
    "training.imbalance",
)
_PREP_CACHE: dict[tuple[Any, ...], Any] = {}
PREP_CACHE_MAX = 5  # stage 5 cycles 5 seeds per cut depth


def _prep_key(spec: Any, cfg: Config, seed: int) -> tuple[Any, ...]:
    def get(path: str) -> Any:
        obj: Any = cfg
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    return (spec.id, seed, *(get(f) for f in PREP_CFG_FIELDS))


def _prep(spec: Any, cfg: Config, seed: int) -> Any:
    """``preprocess_frame`` once per (dataset, seed, preprocessing settings) per
    process (2026-10-16 speed-up, identical results). Stages 1-4, 6 and 7 each
    re-preprocessed the same frame and stage 5 did so 15 times for 5 seeds;
    preprocessing is fully seeded and uses no global random state, so a stored copy
    is the same data. Its arrays are made read-only: a stage that tried to change
    them in place would fail loudly instead of silently changing a later stage."""
    import numpy as np

    from src.data.preprocess import preprocess_frame

    key = _prep_key(spec, cfg, seed)
    if key not in _PREP_CACHE:
        data = preprocess_frame(_raw(spec), spec, cfg, seed=seed)
        for name in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
            arr = getattr(data, name)
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)
        if len(_PREP_CACHE) >= PREP_CACHE_MAX:
            _PREP_CACHE.pop(next(iter(_PREP_CACHE)))  # oldest first
        _PREP_CACHE[key] = data
    return _PREP_CACHE[key]


def _map_deviation(feature_names: list[str]) -> list[dict[str, str]]:
    """Deviation entry when attack constraints use a not-yet-reviewed map."""
    from config.constraints import UNREVIEWED_MAP_FEATURES

    used = sorted(set(feature_names) & UNREVIEWED_MAP_FEATURES)
    if not used:
        return []
    return [
        {
            "id": "unreviewed_mutability_map",
            "what": f"{len(used)} feature(s) use the DRAFT NetFlow mutability map "
            "(config/constraints_netflow_draft.py), not yet human-reviewed "
            "(02_THREAT_MODEL 4.1). Owner decision 2026-10-13.",
            "effect": "Attack realism (which features may move, and how) is "
            "unverified; treat these attack results as provisional.",
        }
    ]


# --- stage stubs (replaced with real logic by later tasks) ------------------


def stage0(cfg: Config, rid: str) -> None:
    _log.info(
        "Stage 0 setup: device=%s tier=%s", get_primary_device(), get_compute_tier()
    )


def _gbm_baseline(data: Any, cfg: Config) -> tuple[str, Any] | None:
    """Gradient-boosting baseline (LightGBM, else XGBoost); None if neither installed.

    Returns ``(model_name, test_predictions)``. Same tree budget as the RF.
    """
    try:
        from lightgbm import LGBMClassifier

        # class-balanced like the RF and the class-weighted MLP. Unweighted, the
        # 7-25-row classes (Heartbleed, Infiltration) give near-zero hessians and
        # the multiclass boosting diverged (raw scores ~1e6, macro-F1 0.16-0.45).
        name, model = (
            "lightgbm",
            LGBMClassifier(
                n_estimators=cfg.training.rf_n_estimators,
                random_state=cfg.seed,
                class_weight="balanced",
                verbose=-1,
            ),
        )
    except ImportError:
        try:
            from xgboost import XGBClassifier

            name, model = (
                "xgboost",
                XGBClassifier(
                    n_estimators=cfg.training.rf_n_estimators, random_state=cfg.seed
                ),
            )
        except ImportError:
            _log.warning("Stage 1: no lightgbm/xgboost installed; GBM baseline skipped")
            return None
    import numpy as np

    # xgboost needs labels 0..n-1; encoded labels have gaps when a class is absent
    classes, y_enc = np.unique(data.y_train, return_inverse=True)
    if name == "xgboost":  # no class_weight option: balanced sample weights instead
        from sklearn.utils.class_weight import compute_sample_weight

        model.fit(
            data.X_train, y_enc, sample_weight=compute_sample_weight("balanced", y_enc)
        )
    else:
        model.fit(data.X_train, y_enc)
    return name, classes[model.predict(data.X_test)]


def stage1(cfg: Config, rid: str) -> None:
    """Preprocess the dataset and train the centralized baselines (RF + MLP).

    Real E1 / shuffled-label results need the actual datasets (Kaggle creds); on
    the fixture this exercises the full Task 1 + Task 2 path for the smoke run.
    """
    from src.data.sources import get_spec
    from src.eval.fitcheck import with_reference
    from src.eval.metrics import classification_metrics, metric_entry
    from src.models.baseline_rf import train_rf
    from src.training.centralized import train_centralized

    try:
        spec = get_spec(cfg.dataset)
        data = _prep(spec, cfg, cfg.seed)
    except (FileNotFoundError, KeyError) as exc:
        _log.warning("Stage 1 skipped (%s); writing stub", exc)
        _write_metrics(rid, "stage1_baseline.json", {"skipped": {"value": str(exc)}})
        return

    rf = train_rf(data.X_train, data.y_train, cfg.training.rf_n_estimators, cfg.seed)
    rf_m = classification_metrics(
        data.y_test, rf.predict(data.X_test), data.label_names
    )
    _, _, mlp_m = train_centralized(data, cfg, device=get_primary_device())
    gbm = _gbm_baseline(data, cfg)
    gbm_m = (
        {
            "gbm_macro_f1": metric_entry(
                "macro_f1",
                classification_metrics(data.y_test, gbm[1], data.label_names)[
                    "macro_f1"
                ],
                model=gbm[0],
            )
        }
        if gbm
        else {}
    )

    _write_metrics(
        rid,
        "stage1_baseline.json",
        {
            "rf_macro_f1": metric_entry("macro_f1", rf_m["macro_f1"]),
            # marks the fixed (class-balanced) GBM; scripts/rerun_baselines.py
            # re-runs stage 1 of any earlier run that lacks it
            "gbm_class_weight": {"value": "balanced"},
            **gbm_m,
            "mlp_macro_f1": metric_entry("macro_f1", mlp_m["macro_f1"]),
            "dataset_stats": {"n_train": int(len(data.y_train)), **data.stats},
            "mlp_per_class": mlp_m["per_class"],
            # automatic fit check (src/eval/fitcheck.py): reported, not acted on
            # in research runs (owner: registered design unchanged)
            "mlp_fit": {
                "value": with_reference(mlp_m["fit"], rf_m["macro_f1"], "Random Forest")
            },
            # any-attack detection + false alarms (key for the temporal test, where
            # test families can be absent from training)
            "detection": {
                "value": {
                    m: {
                        "attack_recall": x["attack_recall"],
                        "binary_fpr": x["binary_fpr"],
                    }
                    for m, x in (("rf", rf_m), ("mlp", mlp_m))
                }
            },
        },
    )


def stage2(cfg: Config, rid: str) -> None:
    """Train the distributed arms (A1/A2/A3) vs the centralized upper bound.

    Full E3 (arms x alpha x seeds) needs the real datasets; on the fixture this
    exercises all four arms for the smoke path.
    """
    from src.data.sources import get_spec
    from src.eval.metrics import metric_entry
    from src.training.centralized import train_centralized
    from src.training.fedavg import train_fedavg
    from src.training.split_sequential import train_split_sequential
    from src.training.splitfed import train_splitfed

    dev = get_primary_device()
    try:
        spec = get_spec(cfg.dataset)
        data = _prep(spec, cfg, cfg.seed)
    except (FileNotFoundError, KeyError) as exc:
        _log.warning("Stage 2 skipped (%s); writing stub", exc)
        _write_metrics(rid, "stage2_arms.json", {"skipped": {"value": str(exc)}})
        return

    _, _, cen = train_centralized(data, cfg, dev)
    _, a1 = train_split_sequential(data, cfg, dev)
    _, a2 = train_fedavg(data, cfg, dev)
    _, a3 = train_splitfed(data, cfg, dev)
    centralized_f1 = cen["macro_f1"]
    _write_metrics(
        rid,
        "stage2_arms.json",
        {
            # distributed arms carry their centralized upper-bound control (§3.4)
            "A0_centralized_macro_f1": metric_entry("macro_f1", centralized_f1),
            "A1_split_sequential_macro_f1": metric_entry(
                "macro_f1",
                a1["test"]["macro_f1"],
                control={"name": "centralized_upper_bound", "value": centralized_f1},
            ),
            "A2_fedavg_macro_f1": metric_entry(
                "macro_f1",
                a2["test"]["macro_f1"],
                control={"name": "centralized_upper_bound", "value": centralized_f1},
            ),
            "A3_splitfed_macro_f1": metric_entry(
                "macro_f1",
                a3["test"]["macro_f1"],
                control={"name": "centralized_upper_bound", "value": centralized_f1},
            ),
            "A1_per_turn_val_macro_f1": a1["per_turn_val_macro_f1"],
        },
    )


def stage3(cfg: Config, rid: str) -> None:
    """Explainability: per-class SHAP, LIME overlap, per-client comparison.

    Attributions are saved for ALL classes (Stage 6 needs fixed-class vectors).
    Per-client differences reflect INPUT distribution (all clients share one
    model), reported against the measured JS divergence.
    """
    import numpy as np

    from src.data.partition import client_distribution_report, partition_dirichlet
    from src.data.sources import get_spec
    from src.explain.explainer import compute_shap
    from src.explain.lime_utils import lime_topk
    from src.explain.shap_utils import (
        global_mean_abs,
        jaccard,
        save_attributions,
        topk_features,
    )
    from src.training.centralized import predict
    from src.training.split_sequential import train_split_sequential

    dev = get_primary_device()
    try:
        spec = get_spec(cfg.dataset)
        data = _prep(spec, cfg, cfg.seed)
    except (FileNotFoundError, KeyError) as exc:
        _log.warning("Stage 3 skipped (%s); writing stub", exc)
        _write_metrics(rid, "stage3_explain.json", {"skipped": {"value": str(exc)}})
        return

    chained, _ = train_split_sequential(data, cfg, dev)
    n_classes = len(data.label_names)
    n_feat = len(data.feature_names)
    k = min(10, n_feat)

    n = min(8, len(data.X_test))
    ids = np.arange(n)
    x_explain = data.X_test[:n]
    background = data.X_train[: cfg.training.shap_background]
    res = compute_shap(
        chained, x_explain, background, n_classes, dev, kernel_instances=min(5, n)
    )
    save_attributions(
        run_dir(rid) / "stage3_shap.npz",
        res.attributions,
        ids,
        data.label_names,
        data.feature_names,
    )

    # SHAP vs LIME top-k overlap, on the predicted class per instance.
    pred = predict(chained, x_explain, dev)
    shap_sets = [
        topk_features(np.abs(res.attributions[i, :, pred[i]]), k) for i in range(n)
    ]
    lime_sets = lime_topk(chained, x_explain, background, data.feature_names, k, dev)
    overlap = float(
        np.mean([jaccard(s, ls) for s, ls in zip(shap_sets, lime_sets, strict=False)])
    )

    # Per-client SHAP over each client's local training distribution.
    parts = partition_dirichlet(
        data.y_train, cfg.training.clients, cfg.training.dirichlet_alpha, cfg.seed
    )
    dist = client_distribution_report(data.y_train, parts, cfg.training.dirichlet_alpha)
    per_client = []
    for cidx in parts:
        m = min(4, len(cidx))
        if m == 0:
            per_client.append({"top_features": [], "n": 0})
            continue
        cres = compute_shap(
            chained,
            data.X_train[cidx][:m],
            background,
            n_classes,
            dev,
            kernel_instances=min(2, m),
        )
        top = sorted(topk_features(global_mean_abs(cres.attributions), k))
        per_client.append(
            {"top_features": [data.feature_names[i] for i in top], "n": int(len(cidx))}
        )

    _write_metrics(
        rid,
        "stage3_explain.json",
        {
            "explainer": {
                "value": res.explainer,
                "kernel_crossval_spearman": res.kernel_crossval_spearman,
            },
            "shap_lime_topk_jaccard_mean": {"value": overlap},
            "realised_mean_pairwise_js": {"value": dist["realised_mean_pairwise_js"]},
            "per_client_top_features": {"value": per_client},
        },
    )

    # Per-attack-class explanation (top SHAP features) + curated mitigations.
    from src.explain.attack_report import build_report

    build_report(chained, data, dev, run_dir(rid))


def stage4(cfg: Config, rid: str) -> None:
    """Attacks (Stage 4): control-first, each with its §3 control.

    random-matched (control) -> FGSM/PGD (T1) -> GAN (T2) -> activation inference
    (T3) -> client attribution (T4). ASR over the clean-correct filter (§3.5);
    GAN ASR jointly with realism (§4.5); T3 leakage vs the mean-predictor (§3.2);
    T4 balanced accuracy vs the class prior (§3.3). On the fixture this is
    plumbing (no class signal by design), not a result.
    """
    import numpy as np

    from src.attack.activation_inference import activation_inference
    from src.attack.client_attribution import client_attribution
    from src.attack.constraints import FeatureConstraints
    from src.attack.fgsm_pgd import fgsm, pgd
    from src.attack.gan_attack import gan_evasion
    from src.attack.random_matched import random_matched
    from src.data.partition import partition_dirichlet
    from src.data.sources import get_spec
    from src.eval.metrics import (
        achieved_perturbation,
        attack_success_rate,
        clean_correct_index,
        metric_entry,
    )
    from src.training.centralized import predict
    from src.training.split_sequential import train_split_sequential

    dev = get_primary_device()
    try:
        spec = get_spec(cfg.dataset)
        data = _prep(spec, cfg, cfg.seed)
    except (FileNotFoundError, KeyError) as exc:
        _log.warning("Stage 4 skipped (%s); writing stub", exc)
        _write_metrics(rid, "stage4_attacks.json", {"skipped": {"value": str(exc)}})
        return

    chained, _ = train_split_sequential(data, cfg, dev)
    fc = FeatureConstraints.from_run(
        data.feature_names,
        data.scaler,
        data.scaler.inverse_transform(data.X_train),
        corrected="derived_v2" in cfg.corrections,
    )
    eps, iters = cfg.training.epsilon, cfg.training.attack_iters
    inv_epochs = cfg.training.inversion_epochs
    rng = np.random.default_rng(cfg.seed)

    # T1: attack the clean-correct, non-benign test samples (§3.5).
    clean_pred = predict(chained, data.X_test, dev)
    C = clean_correct_index(data.y_test, clean_pred)
    x_c = data.X_test[C]
    x_fgsm = fgsm(chained, x_c, eps, fc, dev) if len(C) else x_c
    x_pgd = pgd(chained, x_c, eps, iters, fc, dev) if len(C) else x_c
    x_rand = random_matched(x_c, x_pgd, fc.mask, fc, np.inf, rng) if len(C) else x_c

    ctrl = {"name": "clean_correct_filter"}
    metrics: dict[str, Any] = {
        "fgsm_asr": (
            attack_success_rate(predict(chained, x_fgsm, dev), len(C), ctrl)
            if len(C)
            else metric_entry("attack_success_rate", 0.0, ctrl, denominator=0)
        ),
        "pgd_asr": (
            attack_success_rate(predict(chained, x_pgd, dev), len(C), ctrl)
            if len(C)
            else metric_entry("attack_success_rate", 0.0, ctrl, denominator=0)
        ),
        # the control's own ASR: random perturbations of equal norm should rarely evade
        "random_matched_asr": (
            attack_success_rate(
                predict(chained, x_rand, dev), len(C), {"name": "random_matched"}
            )
            if len(C)
            else metric_entry(
                "attack_success_rate", 0.0, {"name": "random_matched"}, denominator=0
            )
        ),
    }
    if len(C):
        metrics["pgd_achieved_perturbation_raw"] = {
            "value": achieved_perturbation(
                fc.to_raw(x_c), fc.to_raw(x_pgd), fc.classes, data.feature_names
            )
        }

    # T2: GAN evasion from the most populous non-benign training class.
    nonbenign = [c for c in np.unique(data.y_train) if c != 0]
    if nonbenign:
        src = max(nonbenign, key=lambda c: int(np.sum(data.y_train == c)))
        x_src = data.X_train[data.y_train == src]
        _, metrics["gan_asr"] = gan_evasion(
            chained, x_src, fc, dev, cfg.seed, steps=cfg.training.gan_steps
        )

    # T3: activation inference (aux = train, target = test). Stated assumption below.
    metrics["activation_leakage"] = activation_inference(
        chained.client, data.X_train, data.X_test, dev, cfg.seed, epochs=inv_epochs
    )

    # T4: client attribution over the Dirichlet partitions.
    parts = partition_dirichlet(
        data.y_train, cfg.training.clients, cfg.training.dirichlet_alpha, cfg.seed
    )
    metrics["client_attribution"] = client_attribution(
        chained.client,
        [data.X_train[p] for p in parts],
        dev,
        cfg.seed,
        converge="attribution_converge" in cfg.corrections,
    )

    np.savez(
        run_dir(rid) / "adversarial_sets.npz",
        clean_correct_idx=C,
        fgsm=x_fgsm,
        pgd=x_pgd,
        random_matched=x_rand,
    )
    payload = {
        "run_id": rid,
        "metrics": metrics,
        "deviations": [
            *_map_deviation(data.feature_names),
            {
                "id": "T3_trained_clientnet_aux",
                "what": "T3 inversion embeds auxiliary data through the trained "
                "ClientNet, not a shadow model (human-approved, 02_THREAT_MODEL §2).",
                "effect": "Leakage is an upper-ish bound, not a pure passive lower "
                "bound. Mirrored in 11_LIMITATIONS.",
            },
        ],
        "preliminary": True,
    }
    with open(
        run_dir(rid) / "metrics" / "stage4_attacks.json", "w", encoding="utf-8"
    ) as fh:
        json.dump(payload, fh, indent=2)


def _plot_cutdepth(
    path: Any, depths: list[int], agg: dict[int, dict], n_seeds: int
) -> None:
    """F1: the three-axis cut-depth trade-off figure (macro-F1, leakage, FLOPs)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        ("macro_f1", "macro-F1 (utility)", "tab:blue"),
        ("leakage", "leakage ratio (privacy)", "tab:red"),
        ("flops", "client forward FLOPs (cost)", "tab:green"),
    ]
    for ax, (key, label, color) in zip(axes, panels, strict=True):
        means = [agg[d][key][0] for d in depths]
        stds = [agg[d][key][1] for d in depths]
        ax.errorbar(depths, means, yerr=stds, marker="o", color=color, capsize=4)
        ax.set_xlabel("cut depth d")
        ax.set_ylabel(label)
        ax.set_xticks(depths)
    fig.suptitle(f"Cut-depth trade-off (E4), mean +/- std over {n_seeds} seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def stage5(cfg: Config, rid: str) -> None:
    """E4 cut-depth ablation — the headline (03_ARCHITECTURE §2 Stage 5).

    For d in {1,2,3} x seeds: retrain A1, re-run T3 inversion, compute client cost.
    Reports macro-F1 (utility), leakage ratio vs the mean-predictor (privacy), and
    client FLOPs/params (cost), then the Spearman trend of leakage over d with BH
    correction, and an explicit H1 verdict. On the fixture (no class signal) the
    trend is expected to be inconclusive — reported plainly, not tuned away.
    """
    from dataclasses import replace as _replace

    import numpy as np

    from src.attack.activation_inference import activation_inference
    from src.attack.constraints import FeatureConstraints  # noqa: F401 (parity import)
    from src.data.sources import get_spec
    from src.eval.cost import client_cost
    from src.eval.metrics import metric_entry
    from src.eval.stats import benjamini_hochberg, spearman_trend
    from src.training.split_sequential import train_split_sequential

    dev = get_primary_device()
    try:
        spec = get_spec(cfg.dataset)
    except KeyError as exc:
        _log.warning("Stage 5 skipped (%s); writing stub", exc)
        _write_metrics(rid, "stage5_cutdepth.json", {"skipped": {"value": str(exc)}})
        return

    depths = [1, 2, 3]
    # 5 seeds for the headline (§5); a --fast smoke uses 2 (preliminary, non-result).
    seeds = [0, 1] if cfg.runtime.fast else [0, 1, 2, 3, 4]

    # (d, seed) -> (macro_f1, leakage, flops); plus flat arrays for the trend test.
    per_run: dict[int, list[dict[str, float]]] = {d: [] for d in depths}
    trend_levels, trend_leak, trend_f1 = [], [], []
    try:
        for d in depths:
            for s in seeds:
                rcfg = _replace(cfg, seed=s, model=_replace(cfg.model, cut_depth=d))
                data = _prep(spec, rcfg, s)
                chained, res = train_split_sequential(data, rcfg, dev)
                leak = activation_inference(
                    chained.client,
                    data.X_train,
                    data.X_test,
                    dev,
                    s,
                    epochs=rcfg.training.inversion_epochs,
                )
                cost = client_cost(len(data.feature_names), rcfg.model.hidden_width, d)
                f1 = res["test"]["macro_f1"]
                per_run[d].append(
                    {
                        "macro_f1": f1,
                        "leakage": leak["value"],
                        "flops": cost["client_forward_flops"],
                        "params": cost["client_params"],
                        "activation_bytes": cost["cut_activation_bytes_per_sample"],
                    }
                )
                trend_levels.append(d)
                trend_leak.append(leak["value"])
                trend_f1.append(f1)
                # 15 iterations in one process: drop the big frame + model and
                # collect before the next re-preprocess, so RAM doesn't fragment.
                del data, chained, res, leak
                import gc

                gc.collect()
    except (FileNotFoundError, KeyError) as exc:
        _log.warning("Stage 5 skipped (%s); writing stub", exc)
        _write_metrics(rid, "stage5_cutdepth.json", {"skipped": {"value": str(exc)}})
        return

    def _ms(d: int, key: str) -> tuple[float, float]:
        vals = [r[key] for r in per_run[d]]
        return float(np.mean(vals)), float(np.std(vals))

    agg = {d: {k: _ms(d, k) for k in ("macro_f1", "leakage", "flops")} for d in depths}
    leak_trend = spearman_trend(np.array(trend_levels), np.array(trend_leak))
    f1_trend = spearman_trend(np.array(trend_levels), np.array(trend_f1))
    adj = benjamini_hochberg([leak_trend["p"], f1_trend["p"]])
    leak_trend["p_adjusted"], f1_trend["p_adjusted"] = adj[0], adj[1]

    # H1 verdict (§1 H1): leakage should decrease monotonically with d.
    if leak_trend["rho"] < 0 and leak_trend["p_adjusted"] < 0.05:
        verdict = "supported"
    elif leak_trend["p_adjusted"] >= 0.05:
        verdict = "falsified_or_inconclusive"  # no significant monotone trend
    else:
        verdict = "inconclusive"

    _plot_cutdepth(
        run_dir(rid) / "figures" / "F1_cutdepth.png", depths, agg, len(seeds)
    )

    by_depth = {
        str(d): {
            "macro_f1": {"mean": agg[d]["macro_f1"][0], "std": agg[d]["macro_f1"][1]},
            # aggregate leakage ratio (mean/std over seeds) with its mean-predictor
            # control; per-seed baseline MSEs vary and live in each run's stage4.
            "leakage_ratio": metric_entry(
                "leakage_ratio",
                agg[d]["leakage"][0],
                control={"name": "mean_predictor"},
                std=agg[d]["leakage"][1],
            ),
            "client_cost": {
                "forward_flops": per_run[d][0]["flops"],
                "params": per_run[d][0]["params"],
                "cut_activation_bytes_per_sample": per_run[d][0]["activation_bytes"],
            },
        }
        for d in depths
    }

    _write_metrics(
        rid,
        "stage5_cutdepth.json",
        {
            "n_seeds": {"value": len(seeds)},
            "by_depth": {"value": by_depth},
            "leakage_trend_spearman": metric_entry(
                "leakage_ratio",  # controlled family; carries the trend + control
                leak_trend["rho"],
                control={
                    "name": "mean_predictor",
                    "note": "trend over per-run leakage",
                },
                p=leak_trend["p"],
                p_adjusted=leak_trend["p_adjusted"],
                n=leak_trend["n"],
            ),
            "macro_f1_trend_spearman": {
                "value": f1_trend["rho"],
                "p": f1_trend["p"],
                "p_adjusted": f1_trend["p_adjusted"],
            },
            "H1_verdict": {"value": verdict},
        },
    )


def _plot_drift_panels(
    path: Any,
    feat_names: list[str],
    phi_clean: Any,
    phi_adv: Any,
    phi_rand: Any,
    class_name: str,
    drift_adv: float,
    drift_rand: float,
    top: int = 12,
) -> None:
    """F2: the three-panel drift figure (clean / adversarial / random-matched).

    The most important figure in the project (Task 7). One malicious flow's
    fixed-class attribution, then its adversarial counterpart's shifted
    attribution, then the random-matched counterpart showing how much drift comes
    from perturbation magnitude alone. Two panels is the version that gets
    dismissed (03_ARCHITECTURE Stage 6) -- so, three.
    """
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Order features by clean importance and show the top-N for legibility.
    order = np.argsort(np.abs(phi_clean))[::-1][: min(top, len(feat_names))]
    labels = [feat_names[i] for i in order]
    y = np.arange(len(order))[::-1]  # first (most important) at the top
    panels = [
        ("Clean (correctly flagged)", phi_clean, "tab:blue", None),
        (f"Adversarial (PGD)\ndrift={drift_adv:.2f}", phi_adv, "tab:red", None),
        (f"Random-matched\ndrift={drift_rand:.2f}", phi_rand, "tab:gray", None),
    ]
    lim = max(1e-9, float(np.abs([phi_clean, phi_adv, phi_rand]).max())) * 1.1

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    for ax, (title, phi, color, _) in zip(axes, panels, strict=True):
        ax.barh(y, phi[order], color=color, edgecolor="black", linewidth=0.3)
        ax.axvline(0.0, color="black", linewidth=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("SHAP attribution (fixed class)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    fig.suptitle(
        f"F2 Explanation drift under perturbation (fixed class = {class_name!r}); "
        "drift = 1 - Spearman vs clean",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def stage6(cfg: Config, rid: str) -> None:
    """E6 explanation stability — H2, the fixed-class drift with its control.

    The single most important discipline here: fixed-class drift (true label held
    constant, 01_RESEARCH_DESIGN §4.4) with the **random-matched** control. For the
    same clean-correct, non-benign instances we compute SHAP attributions on the
    clean input and on FGSM / PGD / random-matched perturbations (identical
    explainer + background as Stage 3, so any drift reflects the input change), take
    the fixed-class vectors, and drift = 1 - similarity on Spearman / Jaccard /
    cosine. H2 is the paired difference PGD-drift - random-matched-drift, one-sided
    Wilcoxon over instances (§5). On the signal-free fixture this is machinery only;
    real H2 numbers wait on the Kaggle download.

    GAN drift (E6) is deferred: the GAN emits new generated samples, not per-instance
    perturbations of the fixed instances, so it has no paired clean/random-matched
    counterpart for this test. Recorded in deviations and 11_LIMITATIONS §2.
    """
    import numpy as np

    from src.attack.constraints import FeatureConstraints
    from src.attack.fgsm_pgd import fgsm, pgd
    from src.attack.random_matched import random_matched
    from src.data.sources import get_spec
    from src.eval.metrics import clean_correct_index, metric_entry
    from src.eval.stats import wilcoxon_paired
    from src.explain.drift import (
        drift_summary,
        fixed_class_matrix,
        per_instance_drift,
    )
    from src.explain.explainer import compute_shap
    from src.training.centralized import predict
    from src.training.split_sequential import train_split_sequential

    dev = get_primary_device()
    try:
        spec = get_spec(cfg.dataset)
        data = _prep(spec, cfg, cfg.seed)
    except (FileNotFoundError, KeyError) as exc:
        _log.warning("Stage 6 skipped (%s); writing stub", exc)
        _write_metrics(rid, "stage6_drift.json", {"skipped": {"value": str(exc)}})
        return

    chained, _ = train_split_sequential(data, cfg, dev)
    fc = FeatureConstraints.from_run(
        data.feature_names,
        data.scaler,
        data.scaler.inverse_transform(data.X_train),
        corrected="derived_v2" in cfg.corrections,
    )
    n_classes = len(data.label_names)
    n_feat = len(data.feature_names)
    k = min(10, n_feat)
    eps, iters = cfg.training.epsilon, cfg.training.attack_iters
    rng = np.random.default_rng(cfg.seed)

    # Instance set: clean-correct, non-benign test samples (§3.5) — "one malicious
    # flow, correctly flagged". Fall back to the first test rows if the fixture
    # carries none (no class signal by design); machinery still exercises.
    clean_pred = predict(chained, data.X_test, dev)
    C = clean_correct_index(data.y_test, clean_pred)
    if len(C) == 0:
        C = np.arange(min(8, len(data.X_test)))
    C = C[: min(cfg.training.drift_instances, len(C))]
    x_clean = data.X_test[C]
    y_true = np.asarray(data.y_test)[C]  # fixed class = true label

    # Paired perturbations of the SAME instances. random-matched matches PGD's
    # per-instance L-inf norm on the same mutable features (the H2 control).
    x_fgsm = fgsm(chained, x_clean, eps, fc, dev)
    x_pgd = pgd(chained, x_clean, eps, iters, fc, dev)
    x_rand = random_matched(x_clean, x_pgd, fc.mask, fc, np.inf, rng)

    background = data.X_train[: cfg.training.shap_background]
    ki = min(5, len(x_clean))

    # Full attribution cube (n, f, C) once per input; identical explainer +
    # background as Stage 3, so any drift reflects the input change, not the
    # explainer. Reused for both fixed-class (primary) and predicted-class drift.
    def _attr(x: np.ndarray):  # noqa: ANN202 - local ShapResult
        return compute_shap(chained, x, background, n_classes, dev, kernel_instances=ki)

    a_clean = _attr(x_clean)
    a_fgsm = _attr(x_fgsm)
    a_pgd = _attr(x_pgd)
    a_rand = _attr(x_rand)
    explainer, crossval = a_clean.explainer, a_clean.kernel_crossval_spearman

    # Fixed-class drift (primary): true label held constant. PGD is the headline arm.
    phi_clean = fixed_class_matrix(a_clean.attributions, y_true)
    phi_pgd = fixed_class_matrix(a_pgd.attributions, y_true)
    phi_rand = fixed_class_matrix(a_rand.attributions, y_true)
    d_fgsm = per_instance_drift(
        phi_clean, fixed_class_matrix(a_fgsm.attributions, y_true), k
    )
    d_pgd = per_instance_drift(phi_clean, phi_pgd, k)
    d_rand = per_instance_drift(phi_clean, phi_rand, k)

    # H2: paired difference (PGD drift - random-matched drift), one-sided "greater"
    # to match the pre-registered falsification criterion (§1 H2). Reported on the
    # primary Spearman measure. Median_diff gives direction/magnitude.
    h2 = wilcoxon_paired(d_pgd["spearman"], d_rand["spearman"], alternative="greater")
    if h2["median_diff"] > 0 and h2["p"] < 0.05:
        h2_verdict = "supported"  # adversarial drift exceeds the control
    else:
        h2_verdict = "not_supported"  # publishable negative result (§ Task 7)

    # Predicted-class drift (secondary) — CONFOUNDED by the class change; reported,
    # never headline. Uses each input's own predicted class before/after, reusing
    # the already-computed attribution cubes.
    pc_clean = predict(chained, x_clean, dev)
    pc_pgd = predict(chained, x_pgd, dev)
    pred_drift = per_instance_drift(
        fixed_class_matrix(a_clean.attributions, pc_clean),
        fixed_class_matrix(a_pgd.attributions, pc_pgd),
        k,
    )

    # F2 — pick the most illustrative instance (largest PGD Spearman drift).
    j = int(np.argmax(d_pgd["spearman"]))
    _plot_drift_panels(
        run_dir(rid) / "figures" / "F2_drift.png",
        data.feature_names,
        phi_clean[j],
        phi_pgd[j],
        phi_rand[j],
        data.label_names[int(y_true[j])],
        float(d_pgd["spearman"][j]),
        float(d_rand["spearman"][j]),
    )

    def _drift_metric(measure: str) -> dict[str, Any]:
        # each drift_* metric carries its random-matched control (REQUIRES_CONTROL);
        # value is the adversarial (PGD) mean drift, control the matched mean drift.
        return metric_entry(
            f"drift_{measure}",
            float(d_pgd[measure].mean()),
            control={
                "name": "random_matched",
                "value": float(d_rand[measure].mean()),
                "std": float(d_rand[measure].std()),
            },
            std=float(d_pgd[measure].std()),
            n=int(len(d_pgd[measure])),
        )

    payload = {
        "run_id": rid,
        "metrics": {
            "n_instances": {"value": int(len(x_clean))},
            "explainer": {"value": explainer, "kernel_crossval_spearman": crossval},
            # primary: fixed-class drift, each measure with its random-matched control
            "drift_spearman": _drift_metric("spearman"),
            "drift_jaccard": _drift_metric("jaccard"),
            "drift_cosine": _drift_metric("cosine"),
            "fgsm_fixed_class_drift": drift_summary(d_fgsm),
            "random_matched_fixed_class_drift": drift_summary(d_rand),
            # H2 paired test (PGD vs random-matched, one-sided greater, over instances)
            "H2_wilcoxon_pgd_vs_random": {
                "statistic": h2["statistic"],
                "p": h2["p"],
                "median_diff": h2["median_diff"],
                "n": h2["n"],
                "alternative": "greater",
            },
            "H2_verdict": {"value": h2_verdict},
            # secondary: predicted-class drift, labelled confounded (never headline)
            "predicted_class_drift_confounded": {
                "value": drift_summary(pred_drift),
                "note": "Confounded by the attack's class change (§4.4); not a "
                "headline quantity.",
            },
        },
        "deviations": [
            *_map_deviation(data.feature_names),
            {
                "id": "E6_gan_drift_deferred",
                "what": "GAN drift (E6) is not computed: the GAN emits new generated "
                "samples, not per-instance perturbations of the fixed instances, so "
                "it has no paired clean/random-matched counterpart for the H2 test.",
                "effect": "H2 is evaluated on FGSM/PGD vs random-matched only. GAN "
                "explanation drift is out of scope for Task 7; mirrored in "
                "11_LIMITATIONS §2.",
            },
        ],
        "preliminary": True,
    }
    with open(
        run_dir(rid) / "metrics" / "stage6_drift.json", "w", encoding="utf-8"
    ) as fh:
        json.dump(payload, fh, indent=2)


def _plot_frontier(
    path: Any, points: list[dict], f1_base: float, f1_def: float
) -> None:
    """F3: the privacy/utility frontier (D2 σ sweep) with the D1 clean-cost markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f1 = [p["macro_f1"] for p in points]
    leak = [p["leakage"]["value"] for p in points]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(leak, f1, "-o", color="tab:purple", label="D2 activation-noise sweep")
    for p, x, y in zip(points, leak, f1, strict=True):
        ax.annotate(
            f"σ={p['sigma']:g}",
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.axhline(
        f1_base,
        color="tab:gray",
        ls="--",
        lw=1,
        label=f"undefended macro-F1={f1_base:.2f}",
    )
    ax.axhline(
        f1_def,
        color="tab:red",
        ls=":",
        lw=1.2,
        label=f"D1 adv-trained macro-F1={f1_def:.2f}",
    )
    ax.set_xlabel("leakage ratio (privacy risk →)")
    ax.set_ylabel("macro-F1 (utility →)")
    ax.set_title("F3 Privacy/utility frontier: D2 noise sweep + D1 clean-accuracy cost")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def stage7(cfg: Config, rid: str) -> None:
    """E7 defences — D1 adversarial training and D2 activation-noise sweep (H3, P7).

    D1 (targets T1/T2): a simulated coordinator holding both halves hardens the
    model with adversarial examples; attacks used to EVALUATE it are regenerated
    against the DEFENDED model (01_RESEARCH_DESIGN §7), never the base model's set.
    H3's control is the clean-accuracy delta (base vs defended) — robustness must
    not be credited to a degraded clean model. Task 7's fixed-class drift is re-run
    on the defended model (DV2). D2 (targets T3): sweep σ over cut-layer activation
    noise and plot the leakage/macro-F1 frontier (F3) — a sweep, not one point.

    On the signal-free fixture this is machinery only; real H3/P7 numbers wait on
    the Kaggle download.
    """
    import numpy as np

    from src.attack.constraints import FeatureConstraints
    from src.attack.fgsm_pgd import fgsm, pgd
    from src.attack.gan_attack import gan_evasion
    from src.attack.random_matched import random_matched
    from src.data.sources import get_spec
    from src.defense.activation_noise import noise_sweep
    from src.defense.adversarial_training import adversarial_train_split
    from src.eval.metrics import (
        attack_success_rate,
        clean_correct_index,
        leakage_ratio,
        metric_entry,
    )
    from src.eval.stats import wilcoxon_paired
    from src.explain.drift import fixed_class_matrix, per_instance_drift
    from src.explain.explainer import compute_shap
    from src.training.centralized import predict

    dev = get_primary_device()
    try:
        spec = get_spec(cfg.dataset)
        data = _prep(spec, cfg, cfg.seed)
    except (FileNotFoundError, KeyError) as exc:
        _log.warning("Stage 7 skipped (%s); writing stub", exc)
        _write_metrics(rid, "stage7_defense.json", {"skipped": {"value": str(exc)}})
        return

    base, defended, def_res = adversarial_train_split(data, cfg, dev)
    fc = FeatureConstraints.from_run(
        data.feature_names,
        data.scaler,
        data.scaler.inverse_transform(data.X_train),
        corrected="derived_v2" in cfg.corrections,
    )
    n_classes = len(data.label_names)
    eps, iters = cfg.training.epsilon, cfg.training.attack_iters

    def _asr_against(model: Any) -> dict[str, Any]:
        """FGSM/PGD ASR with attacks regenerated against THIS model (§3.5, §7)."""
        pred = predict(model, data.X_test, dev)
        C = clean_correct_index(data.y_test, pred)
        if len(C) == 0:
            return {
                "fgsm": metric_entry(
                    "attack_success_rate",
                    0.0,
                    {"name": "clean_correct_filter"},
                    denominator=0,
                ),
                "pgd": metric_entry(
                    "attack_success_rate",
                    0.0,
                    {"name": "clean_correct_filter"},
                    denominator=0,
                ),
                "n_clean_correct": 0,
            }
        x_c = data.X_test[C]
        ctrl = {"name": "clean_correct_filter"}
        return {
            "fgsm": attack_success_rate(
                predict(model, fgsm(model, x_c, eps, fc, dev), dev), len(C), ctrl
            ),
            "pgd": attack_success_rate(
                predict(model, pgd(model, x_c, eps, iters, fc, dev), dev), len(C), ctrl
            ),
            "n_clean_correct": int(len(C)),
        }

    asr_base = _asr_against(base)
    asr_def = _asr_against(defended)

    # GAN ASR before/after, regenerated against each model (§7).
    nonbenign = [c for c in np.unique(data.y_train) if c != 0]
    gan_base = gan_def = None
    if nonbenign:
        src = max(nonbenign, key=lambda c: int(np.sum(data.y_train == c)))
        x_src = data.X_train[data.y_train == src]
        _, gan_base = gan_evasion(
            base, x_src, fc, dev, cfg.seed, steps=cfg.training.gan_steps
        )
        _, gan_def = gan_evasion(
            defended, x_src, fc, dev, cfg.seed, steps=cfg.training.gan_steps
        )

    # H3 control: clean macro-F1 delta (base vs defended) — no robustness credit for
    # a degraded clean model.
    from src.training.common import eval_macro_f1

    f1_base = eval_macro_f1(base, data.X_test, data.y_test, n_classes, dev)
    f1_def = def_res["test"]["macro_f1"]

    # DV2: re-run Task 7 fixed-class drift on the DEFENDED model (PGD vs random-matched).
    pred_def = predict(defended, data.X_test, dev)
    Cd = clean_correct_index(data.y_test, pred_def)
    if len(Cd) == 0:
        Cd = np.arange(min(8, len(data.X_test)))
    Cd = Cd[: min(16, len(Cd))]
    x_clean = data.X_test[Cd]
    y_true = np.asarray(data.y_test)[Cd]
    rng = np.random.default_rng(cfg.seed)
    x_pgd = pgd(defended, x_clean, eps, iters, fc, dev)
    x_rand = random_matched(x_clean, x_pgd, fc.mask, fc, np.inf, rng)
    bg = data.X_train[: cfg.training.shap_background]
    ki = min(5, len(x_clean))
    a_clean = compute_shap(defended, x_clean, bg, n_classes, dev, kernel_instances=ki)
    a_pgd = compute_shap(defended, x_pgd, bg, n_classes, dev, kernel_instances=ki)
    a_rand = compute_shap(defended, x_rand, bg, n_classes, dev, kernel_instances=ki)
    phi_clean = fixed_class_matrix(a_clean.attributions, y_true)
    d_pgd = per_instance_drift(
        phi_clean, fixed_class_matrix(a_pgd.attributions, y_true)
    )
    d_rand = per_instance_drift(
        phi_clean, fixed_class_matrix(a_rand.attributions, y_true)
    )
    drift_h2 = wilcoxon_paired(
        d_pgd["spearman"], d_rand["spearman"], alternative="greater"
    )

    # D2: activation-noise sweep -> the privacy/utility frontier.
    points = noise_sweep(
        defended,
        data,
        cfg.training.sigma_sweep,
        dev,
        cfg.seed,
        cfg.training.inversion_epochs,
        n_classes,
    )
    _plot_frontier(
        run_dir(rid) / "figures" / "F3_frontier.png", points, f1_base, f1_def
    )

    # H3 verdict: ASR down (DV1) AND drift not worse than its control (DV2), with the
    # clean-accuracy cost reported (control). PGD ASR is the DV1 summary.
    asr_dropped = asr_def["pgd"]["value"] < asr_base["pgd"]["value"]
    drift_controlled = not (drift_h2["median_diff"] > 0 and drift_h2["p"] < 0.05)
    if asr_dropped and drift_controlled:
        h3_verdict = "supported"
    elif asr_dropped and not drift_controlled:
        h3_verdict = "robustness_without_explanation_recovery"  # §1 H3 negative result
    else:
        h3_verdict = "not_supported"

    payload = {
        "run_id": rid,
        "metrics": {
            "D1_adversarial_training": {
                "clean_macro_f1": {
                    # H3 control: base vs defended clean accuracy (delta reported)
                    "base": f1_base,
                    "defended": f1_def,
                    "delta": f1_def - f1_base,
                },
                "asr_fgsm": {"before": asr_base["fgsm"], "after": asr_def["fgsm"]},
                "asr_pgd": {"before": asr_base["pgd"], "after": asr_def["pgd"]},
                "asr_gan": {"before": gan_base, "after": gan_def},
                "defended_fixed_class_drift": {
                    "drift_spearman": metric_entry(
                        "drift_spearman",
                        float(d_pgd["spearman"].mean()),
                        control={
                            "name": "random_matched",
                            "value": float(d_rand["spearman"].mean()),
                        },
                        std=float(d_pgd["spearman"].std()),
                        n=int(len(d_pgd["spearman"])),
                    ),
                    "H2_on_defended_wilcoxon": {
                        "p": drift_h2["p"],
                        "median_diff": drift_h2["median_diff"],
                        "n": drift_h2["n"],
                        "alternative": "greater",
                    },
                },
            },
            "D2_activation_noise_sweep": {
                "value": [
                    {
                        "sigma": p["sigma"],
                        "macro_f1": p["macro_f1"],
                        "leakage_ratio": leakage_ratio(
                            p["leakage"]["control"]["mse_attack"],
                            p["leakage"]["control"]["mse_baseline"],
                        ),
                    }
                    for p in points
                ],
                "note": "Frontier, not one point (§3). σ=0 is the undefended reference.",
            },
            "H3_verdict": {"value": h3_verdict},
        },
        "deviations": [
            *_map_deviation(data.feature_names),
            {
                "id": "D1_coordinator_capability_gap",
                "what": "D1 adversarial training is run by a simulated coordinator "
                "holding BOTH model halves; a real split-learning client cannot craft "
                "white-box adversarial examples against ServerNet (02_THREAT_MODEL §3).",
                "effect": "Overstates D1's practicality — a strictly stronger defender "
                "than the architecture permits. Mirrored in 11_LIMITATIONS §2.",
            },
            {
                "id": "D1_one_shot_augmentation",
                "what": "Adversarial examples are generated once against the base "
                f"model and mixed in at fraction {cfg.training.adv_fraction} "
                "(cfg.training.adv_fraction, hashed), not regenerated every step "
                "against the hardening model.",
                "effect": "Number-affecting; a weaker form of adversarial training. "
                "The mix ratio is hashed, so it enters run comparability.",
            },
        ],
        "preliminary": True,
    }
    with open(
        run_dir(rid) / "metrics" / "stage7_defense.json", "w", encoding="utf-8"
    ) as fh:
        json.dump(payload, fh, indent=2)


def _plot_transfer(path: Any, names: list[str], matrix: list[list[float]]) -> None:
    """F10: the cross-dataset transfer heatmap (rows=train domain, cols=test domain)."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.array(matrix)
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 1.4 * len(names) + 2))
    im = ax.imshow(m, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("test domain")
    ax.set_ylabel("train domain")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{m[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if m[i, j] < 0.6 else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, label="macro-F1")
    ax.set_title("F10 Cross-dataset transfer (diagonal = in-domain)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def stage8(cfg: Config, rid: str) -> None:
    """E8 external validity — cross-dataset transfer over the shared NetFlow schema.

    Real E8 trains on ``nf-cse-cic-ids2018-v2`` and tests on ``nf-unsw-nb15-v2``
    (and the reverse) over the shared 43-feature NetFlow v9 schema (04_DATA §2).
    Those datasets need the Kaggle download; until then this exercises the
    transfer-matrix + F10 machinery on the fixture by splitting it into two pseudo
    domains (A/B halves). Expect a large real-world drop off-diagonal — the normal,
    honest result for NIDS cross-dataset transfer; report it, do not bury it.
    """
    import numpy as np

    from src.data.preprocess import preprocess_frame
    from src.data.sources import get_spec
    from src.training.common import eval_macro_f1
    from src.training.split_sequential import train_split_sequential

    dev = get_primary_device()

    # Prefer the real E8 pair; fall back to fixture pseudo-domains when absent.
    domains: dict[str, Any] = {}
    fixture_fallback = False
    try:
        if cfg.runtime.fast:  # a smoke run never loads multi-GB real datasets
            raise FileNotFoundError("--fast: using fixture pseudo-domains")
        pair = ["nf-cse-cic-ids2018-v2", "nf-unsw-nb15-v2"]
        for name in pair:
            spec = get_spec(name)
            domains[name] = preprocess_frame(_raw(spec), spec, cfg, seed=cfg.seed)
    except (FileNotFoundError, KeyError) as exc:
        _log.warning("E8 datasets unavailable (%s); fixture pseudo-domains", exc)
        domains = {}
        fixture_fallback = True
        try:
            spec = get_spec(cfg.dataset)
            raw = _raw(spec)
        except (FileNotFoundError, KeyError) as exc2:
            _log.warning("Stage 8 skipped (%s); writing stub", exc2)
            _write_metrics(
                rid, "stage8_transfer.json", {"skipped": {"value": str(exc2)}}
            )
            return
        half = len(raw) // 2
        for name, part in (
            ("fixtureA", raw.iloc[:half]),
            ("fixtureB", raw.iloc[half:]),
        ):
            domains[name] = preprocess_frame(part, spec, cfg, seed=cfg.seed)

    names = list(domains)
    # Train one model per domain (feature space is shared by construction).
    models = {n: train_split_sequential(domains[n], cfg, dev)[0] for n in names}
    n_classes = len(domains[names[0]].label_names)

    # transfer[i][j] = macro-F1 of model trained on i, evaluated on j's test set.
    matrix: list[list[float]] = []
    for i in names:
        row = []
        for j in names:
            dj = domains[j]
            row.append(eval_macro_f1(models[i], dj.X_test, dj.y_test, n_classes, dev))
        matrix.append(row)

    _plot_transfer(run_dir(rid) / "figures" / "F10_transfer.png", names, matrix)

    diag = [matrix[i][i] for i in range(len(names))]
    off = [matrix[i][j] for i in range(len(names)) for j in range(len(names)) if i != j]
    payload = {
        "run_id": rid,
        "metrics": {
            "domains": {"value": names},
            "transfer_macro_f1": {"value": matrix, "rows": "train", "cols": "test"},
            "mean_in_domain_macro_f1": {"value": float(np.mean(diag))},
            "mean_transfer_macro_f1": {"value": float(np.mean(off)) if off else 0.0},
        },
        "deviations": (
            [
                {
                    "id": "E8_fixture_pseudo_domains",
                    "what": "Real E8 datasets (nf-cse-cic-ids2018-v2 / nf-unsw-nb15-v2) "
                    "are unavailable (Kaggle download blocked); transfer is exercised "
                    "on two halves of the signal-free fixture instead.",
                    "effect": "Machinery only — no genuine domain shift, so off-diagonal "
                    "≈ diagonal. Not a transfer result. Mirrored in 11_LIMITATIONS §2.",
                }
            ]
            if fixture_fallback
            else []
        ),
        "preliminary": True,
    }
    with open(
        run_dir(rid) / "metrics" / "stage8_transfer.json", "w", encoding="utf-8"
    ) as fh:
        json.dump(payload, fh, indent=2)


def stage9(cfg: Config, rid: str) -> None:
    """Stage 9 aggregation: regenerate results/ tables from every run in runs/."""
    from src.eval.aggregate import write_results

    summary = write_results()
    _log.info(
        "Stage 9 aggregate: results/ rebuilt from %d runs", summary["n_runs_total"]
    )


_STAGE_FNS: dict[int, Callable[[Config, str], None]] = {
    0: stage0,
    1: stage1,
    2: stage2,
    3: stage3,
    4: stage4,
    5: stage5,
    6: stage6,
    7: stage7,
    8: stage8,
    9: stage9,
}


def _parse_stage(value: str) -> list[int]:
    if value == "all":
        return list(STAGES)
    out = []
    for part in value.split(","):
        try:
            n = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--stage must be 0-9, a comma list like 1,2,3, or 'all'"
            ) from exc
        if n not in STAGES:
            raise argparse.ArgumentTypeError(f"--stage {n} out of range 0-9")
        out.append(n)
    return out


def _resumable(cfg: Config, tag: str) -> tuple[str, set[int]] | None:
    """Newest unfinished run dir with the same experiment, tag, seed and config
    hash, and the stages it already finished (metrics written, not a failure stub)."""
    import re

    from config.base import config_hash
    from utils.paths import RUNS_DIR

    head = "__".join([cfg.experiment_id, *([tag] if tag else []), f"seed{cfg.seed}"])
    pat = re.compile(rf"^{re.escape(head)}__\d{{8}}T\d{{4}}Z__{config_hash(cfg)}$")
    dirs = [
        d
        for d in RUNS_DIR.glob(f"{head}__*")
        if pat.match(d.name) and not (d / "run_complete.json").exists()
    ]
    if not dirs:
        return None
    d = max(dirs, key=lambda x: x.stat().st_mtime)
    done = set()
    for f in (d / "metrics").glob("stage*_*.json"):
        m = re.match(r"stage(\d+)_(.+)\.json$", f.name)
        if m and m.group(2) != "failed":
            done.add(int(m.group(1)))
    return d.name, done


def _note_resume(rid: str, skipped: list[int]) -> None:
    """Record the resume in the manifest (reproducibility: stages ran in two goes)."""
    mf = run_dir(rid) / "manifest.json"
    try:
        m = json.loads(mf.read_text("utf-8"))
    except (OSError, ValueError):
        return
    m.setdefault("resumed", []).append(
        {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "stages_already_done": skipped}
    )
    mf.write_text(json.dumps(m, indent=2), "utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="SplitShield-XAI pipeline")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--experiment", default="smoke")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument(
        "--seed", type=int, default=None, help="default: first seed in the YAML"
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--aggregate",
        action="store_true",
        help="Rebuild results/ tables from runs/ and exit (make results).",
    )
    ap.add_argument(
        "--download-only",
        action="store_true",
        help="Reserved for src/data/download.py; no dataset creds yet (fails loud).",
    )
    args = ap.parse_args()

    configure_logging()

    # `make results`: aggregate every run and exit, no new run created.
    if args.aggregate:
        from src.eval.aggregate import write_results

        summary = write_results()
        print(f"[splitshield] aggregated {summary['n_runs_total']} runs -> results/")
        return

    if args.download_only:
        raise SystemExit(
            "--download-only: dataset credentials not configured; run "
            "src/data/download.py with --source/--slug (fails loud by design)."
        )

    stages = _parse_stage(args.stage)

    raw = load_experiment(f"{args.experiment}.yaml")
    seed = args.seed if args.seed is not None else (raw.get("seeds") or [0])[0]
    cfg = build_config(raw, seed=seed, fast=args.fast)
    set_seed(cfg.seed)

    tag = "smoke" if args.fast else ""
    # --resume (owner 2026-10-14, power cuts): continue the newest unfinished run of
    # this experiment/seed/config in place, skipping stages whose metrics exist
    prev = _resumable(cfg, tag) if args.resume else None
    if prev:
        rid, done = prev
        skipped = [n for n in stages if n in done]
        stages = [n for n in stages if n not in done]
        _log.info("--resume: continuing %s; stages already done %s", rid, skipped)
        _note_resume(rid, skipped)
    else:
        rid = make_run_id(cfg, tag=tag)
        write_manifest(cfg, rid, dataset_stats={})

    import gc
    import traceback

    t0 = time.time()
    failed: list[int] = []
    for n in stages:
        # One stage failing (e.g. out of memory on a huge dataset) must not throw
        # away the hours spent on the others: record it and continue.
        try:
            _STAGE_FNS[n](cfg, rid)
            # a resumed retry fixed this stage: keep the old failure stub, renamed,
            # so aggregation does not read it as the stage's result
            stale = run_dir(rid) / "metrics" / f"stage{n}_failed.json"
            if stale.exists():
                stale.replace(stale.with_name(stale.name + ".superseded"))
        except Exception as exc:  # noqa: BLE001 - isolate every stage
            failed.append(n)
            _log.error("Stage %d FAILED: %s\n%s", n, exc, traceback.format_exc())
            _write_metrics(
                rid,
                f"stage{n}_failed.json",
                {"skipped": {"value": f"failed: {type(exc).__name__}: {exc}"}},
            )
        gc.collect()  # release each stage's frames before the next re-preprocesses
    dt = time.time() - t0
    if not failed:  # a later --resume starts a fresh run instead of reusing this one
        (run_dir(rid) / "run_complete.json").write_text(
            json.dumps({"stages": stages, "elapsed_s": round(dt, 1)}), "utf-8"
        )

    print(
        f"[splitshield] {args.experiment} stages={args.stage} "
        f"run_id={rid} elapsed={dt:.1f}s"
        + (f" FAILED stages={failed}" if failed else "")
    )
    if failed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
