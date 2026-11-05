"""Freeze a deployable detector for the interactive assistant.

Runs once after training/self-improvement. Produces ``artifacts/deploy/``:

  * ``model.pt``       - SharedMLP state_dict of the promoted compact arch,
  * ``scaler.pkl``     - the fitted preprocessing scaler (feature parity at inference),
  * ``columns.json``   - ``feature_names`` + ``label_names`` (order matters),
  * ``top_features.json`` - per-class top SHAP indicators (the "reason" strings).

The assistant loads only these small files, so it starts in seconds without
re-reading the multi-GB training CSV.

    python deploy.py --dataset cicids2017-improved [--fast]
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import replace
from typing import Any

import torch

from config.base import build_config, experiment_for_dataset
from src.data.preprocess import preprocess_frame
from src.data.sources import get_spec, load_raw
from src.explain.attack_report import per_class_top_features
from src.models.mlp import SharedMLP
from src.selfimprove.loop import auto_model, best_dir
from src.training.centralized import train_centralized
from src.training.common import eval_macro_f1
from utils.device import get_primary_device
from utils.logging_config import configure_logging, get_logger
from utils.paths import PROJECT_ROOT
from utils.seed import set_seed

_log = get_logger(__name__)
DEPLOY = PROJECT_ROOT / "artifacts" / "deploy"


def deploy(dataset: str, fast: bool, seed: int = 0) -> None:
    device = get_primary_device()
    # Same config the self-improve pass trained this dataset with (same caps + seed
    # -> same split and scaler as the promoted weights), plus its promoted arch.
    raw = experiment_for_dataset(dataset, fast)
    arch = auto_model(dataset)
    if arch:
        raw["model"] = {**raw.get("model", {}), **arch}
    cfg = build_config(raw, seed=seed, fast=fast)
    data = preprocess_frame(
        load_raw(get_spec(dataset)), get_spec(dataset), cfg, seed=seed
    )

    n_classes = len(data.label_names)
    from src.data.robust import input_stats

    live_stats = input_stats(data.scaler, data.scaler.inverse_transform(data.X_train))
    # deployment preprocessing (owner 2026-10-14): plain vs robust scaling, chosen on
    # validation with the same probe model; research keeps the registered scaler
    prep: dict[str, Any] = {"chosen": "standard"}
    if not fast:
        data, prep = _choose_preprocessing(data, cfg, seed, device, n_classes)

    def build(c: Any) -> torch.nn.Module:
        return SharedMLP(
            n_features=len(data.feature_names),
            n_classes=n_classes,
            hidden_layers=c.model.hidden_layers,
            hidden_width=c.model.hidden_width,
            dropout=c.model.dropout,
            norm=c.model.norm,
        ).to(device)

    # Candidates: the self-improve winner (small by design) and, owner decision
    # 2026-10-13, a tuning search for the most ACCURATE detector (the research MLP
    # underfits vs RF; deployment is free to use a stronger one). Selection is on
    # VALIDATION macro-F1 only; test is reported, never used to choose.
    cands: list[tuple[str, Any, torch.nn.Module]] = []
    fits: dict[str, dict[str, Any]] = {}  # candidate -> automatic fit check
    weights = best_dir(dataset) / "model.pt"
    # the promoted weights were trained on the plain scaler's inputs
    if weights.exists() and prep["chosen"] == "standard":
        m = build(cfg)
        try:
            m.load_state_dict(torch.load(weights, map_location=device))
            cands.append(("self-improve winner", cfg, m))
        except (RuntimeError, KeyError) as exc:  # arch mismatch -> not a candidate
            _log.warning("Promoted weights incompatible (%s); skipped", exc)
    for name, tcfg in _tuning_grid(cfg, fast, seed):
        m, _, met = train_centralized(data, tcfg, device)
        cands.append((name, tcfg, m))
        fits[name] = met["fit"]

    rows = []
    for name, c, m in cands:
        rows.append(
            {
                "name": name,
                "hidden_layers": c.model.hidden_layers,
                "hidden_width": c.model.hidden_width,
                "dropout": c.model.dropout,
                "lr": c.training.lr,
                "val_macro_f1": eval_macro_f1(
                    m, data.X_val, data.y_val, n_classes, device
                ),
                "test_macro_f1": eval_macro_f1(
                    m, data.X_test, data.y_test, n_classes, device
                ),
                "fit": fits.get(name, {}).get("verdict", "n/a"),
            }
        )
    best_i = max(range(len(rows)), key=lambda i: rows[i]["val_macro_f1"])
    name, cfg, model = cands[best_i]
    # automatic fit doctor: fix what the fit check found, keep a fix only if
    # validation improves (overconfidence is handled by temperature calibration)
    if name in fits:
        cfg, model, doctor = _fit_doctor(
            data, cfg, model, fits[name], device, n_classes
        )
        if doctor:
            rows.append(
                {
                    "name": f"{name} + fit doctor",
                    "hidden_layers": cfg.model.hidden_layers,
                    "hidden_width": cfg.model.hidden_width,
                    "dropout": cfg.model.dropout,
                    "lr": cfg.training.lr,
                    "val_macro_f1": eval_macro_f1(model, data.X_val, data.y_val, n_classes, device),
                    "test_macro_f1": eval_macro_f1(model, data.X_test, data.y_test, n_classes, device),
                    "fit": doctor[-1]["fit_after"] if doctor[-1]["kept"] else fits[name]["verdict"],
                    "fit_doctor": doctor,
                }
            )  # fmt: skip
            if any(d["kept"] for d in doctor):
                best_i = len(rows) - 1
    # dynamic neuron growth (owner 2026-10-14): widen/deepen with function-preserving
    # Net2Net copies while validation improves, then prune back (src/models/grow.py)
    if not fast:
        from src.models.grow import PRUNE_TOL, grow

        gcfg = replace(
            cfg,
            training=replace(
                cfg.training,
                max_epochs=max(cfg.training.max_epochs, 100),
                patience=max(cfg.training.patience, 15),
            ),
        )
        gcfg, gmodel, glog = grow(
            data, gcfg, model, device, n_classes,
            train=train_centralized,
            evaluate=lambda m: eval_macro_f1(m, data.X_val, data.y_val, n_classes, device),
        )  # fmt: skip
        if any(e["kept"] for e in glog):
            gval = eval_macro_f1(gmodel, data.X_val, data.y_val, n_classes, device)
            rows.append(
                {
                    "name": f"{rows[best_i]['name']} + grown",
                    "hidden_layers": gcfg.model.hidden_layers,
                    "hidden_width": gcfg.model.hidden_width,
                    "dropout": gcfg.model.dropout,
                    "lr": gcfg.training.lr,
                    "val_macro_f1": gval,
                    "test_macro_f1": eval_macro_f1(gmodel, data.X_test, data.y_test, n_classes, device),
                    "fit": "n/a",
                    "growth": glog,
                }
            )  # fmt: skip
            # same accuracy within the prune tolerance -> the grown/pruned one is
            # at least as good and never needlessly large
            if gval >= rows[best_i]["val_macro_f1"] - PRUNE_TOL:
                best_i, cfg, model = len(rows) - 1, gcfg, gmodel
    _log.info("Deploy: chose %s (val macro-F1 %.3f)", rows[best_i]["name"],
              rows[best_i]["val_macro_f1"])  # fmt: skip
    # ensemble (owner 2026-10-14): MLP, LightGBM, RF alone or averaged, chosen on
    # validation; `scorer` is what alerting uses, the MLP keeps SHAP + D3 energy
    from src.models.ensemble import choose, fit_trees

    trees = fit_trees(data, seed, fast)
    scorer, ens_rows = choose(
        model, trees, data, device,
        evaluate=lambda m: eval_macro_f1(m, data.X_val, data.y_val, n_classes, device),
    )  # fmt: skip
    ensemble = scorer.weights if scorer.weights != {"mlp": 1.0} else None
    _log.info("Deploy: detector members %s", sorted(scorer.weights))
    _write_report(dataset, data, scorer, rows, best_i, device, prep, ens_rows)

    tops = per_class_top_features(model, data, device, k=8)
    # D3 runtime detector: energy threshold at 1% false-alarm rate on clean validation
    # flows the model never trained on.
    from src.defense.detector import fit_energy_detector

    energy_thr = fit_energy_detector(
        model, data.X_val, device, target_fpr=0.01
    ).threshold
    temperature = _fit_temperature(scorer, data.X_val, data.y_val, device)
    high_thr = _alert_threshold(scorer, data, temperature, device, target_fpr=0.01)
    by_class = _class_thresholds(scorer, data, temperature, device, target_fpr=0.01)
    qhat = conformal_qhat(scorer, data.X_val, data.y_val, temperature, device)

    DEPLOY.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), DEPLOY / "model.pt")
    if ensemble:
        import joblib

        members = {k: trees[k] for k in ensemble if k != "mlp"}
        joblib.dump(members, DEPLOY / "ensemble.pkl", compress=3)
    with open(DEPLOY / "scaler.pkl", "wb") as fh:
        pickle.dump(data.scaler, fh)
    (DEPLOY / "columns.json").write_text(
        json.dumps(
            {
                "feature_names": list(data.feature_names),
                "label_names": list(data.label_names),
                "hidden_layers": cfg.model.hidden_layers,
                "hidden_width": cfg.model.hidden_width,
                "dropout": cfg.model.dropout,
                "norm": cfg.model.norm,
                "dataset": dataset,
                "energy_threshold": energy_thr,
                # confidence calibration on validation (the model grows overconfident:
                # val loss rises while val F1 still improves); RCL refines it later
                "temperature": temperature,
                # two-tier alerting (owner 2026-10-13): above this attack probability
                # an alert is confirmed (~1% of validation benign flows exceed it)
                "alert_threshold_high": high_thr,
                # per likeliest-attack-class thresholds (same ~1% FPR budget, each
                # class's own benign confusions); missing class -> the global one
                "alert_threshold_by_class": by_class,
                # split-conformal (alpha 0.05): classes whose 1 - p <= qhat form a
                # set that holds the true class ~95 % of the time; a set of 2+
                # classes marks the alert "model unsure"
                "conformal_qhat": qhat,
                "conformal_alpha": CONFORMAL_ALPHA,
                "alert_target_fpr": 0.01,
                # None = MLP only; else member weights (ensemble.pkl holds the trees)
                "ensemble": ensemble,
                # live-input preprocessing: medians fill missing values, ranges flag
                # flows in different units than training (src/data/robust.py)
                "preprocessing": prep["chosen"],
                "input_stats": live_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (DEPLOY / "top_features.json").write_text(
        json.dumps(tops, indent=2), encoding="utf-8"
    )
    _log.info("Deployed detector -> %s", DEPLOY)
    print(f"[deploy] detector frozen -> {DEPLOY}")


def _choose_preprocessing(
    data: Any, cfg: Any, seed: int, device: str, n_classes: int
) -> tuple[Any, dict[str, Any]]:
    """Train the first tuning candidate on plain-scaled and robust-scaled inputs;
    keep the preprocessing with the higher VALIDATION macro-F1."""
    from dataclasses import replace as dc_replace

    import numpy as np

    from src.data.robust import RobustFlowScaler

    raw = [
        data.scaler.inverse_transform(x)
        for x in (data.X_train, data.X_val, data.X_test)
    ]
    rob = RobustFlowScaler().fit(raw[0])
    data_r = dc_replace(
        data,
        X_train=rob.transform(raw[0]).astype(np.float32),
        X_val=rob.transform(raw[1]).astype(np.float32),
        X_test=rob.transform(raw[2]).astype(np.float32),
        scaler=rob,
    )
    probe = _tuning_grid(cfg, False, seed)[0][1]
    vals = {}
    for name, d in (("standard", data), ("robust", data_r)):
        m, _, _ = train_centralized(d, probe, device)
        vals[name] = eval_macro_f1(m, d.X_val, d.y_val, n_classes, device)
    chosen = "robust" if vals["robust"] > vals["standard"] else "standard"
    _log.info("Deploy preprocessing: standard %.4f vs robust %.4f -> %s",
              vals["standard"], vals["robust"], chosen)  # fmt: skip
    info = {"chosen": chosen, "val_macro_f1": vals,
            "log1p_features": int(rob.log_cols_.sum())}  # fmt: skip
    return (data_r if chosen == "robust" else data), info


def _fit_temperature(model: torch.nn.Module, X: Any, y: Any, device: str) -> float:
    """Temperature T minimising validation log-loss of softmax(logits / T)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu()
    target = torch.as_tensor(y, dtype=torch.long)
    grid = torch.logspace(-1, 1, 81)  # 0.1 .. 10
    nll = [float(torch.nn.functional.cross_entropy(logits / t, target)) for t in grid]
    return float(grid[int(torch.tensor(nll).argmin())])


def _alert_threshold(
    model: torch.nn.Module,
    data: Any,
    temperature: float,
    device: str,
    target_fpr: float,
) -> float | None:
    """Attack-probability threshold that ~``target_fpr`` of VALIDATION benign flows
    exceed (calibrated probabilities, as live alerting uses). None: no BENIGN class."""
    import numpy as np

    names = list(data.label_names)
    if "BENIGN" not in names:
        return None
    b = names.index("BENIGN")
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(data.X_val, dtype=torch.float32, device=device)
        mass = 1 - torch.softmax(model(x) / temperature, 1)[:, b].cpu().numpy()
    benign = mass[np.asarray(data.y_val) == b]
    if len(benign) == 0:
        return None
    return float(max(0.5, np.quantile(benign, 1 - target_fpr)))


CONFORMAL_ALPHA = 0.05


def _probs(model: torch.nn.Module, X: Any, temperature: float, device: str) -> Any:
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float32, device=device)
        return torch.softmax(model(x) / temperature, 1).cpu().numpy()


def conformal_qhat(
    model: torch.nn.Module,
    X: Any,
    y: Any,
    temperature: float,
    device: str,
    alpha: float = CONFORMAL_ALPHA,
) -> float:
    """Split-conformal threshold on VALIDATION (score = 1 - p(true class))."""
    import numpy as np

    p = _probs(model, X, temperature, device)
    yy = np.asarray(y)
    s = 1 - p[np.arange(len(yy)), yy]
    n = len(s)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def conformal_sets(p: Any, qhat: float) -> Any:
    """Boolean (n, classes): class c is in a flow's set when 1 - p_c <= qhat."""
    return (1 - p) <= qhat


def _class_thresholds(
    model: torch.nn.Module,
    data: Any,
    temperature: float,
    device: str,
    target_fpr: float,
    min_count: int = 50,
) -> dict[str, float]:
    """Per attack class: the attack-probability threshold that ~``target_fpr`` of
    the VALIDATION benign flows whose likeliest attack is that class exceed. Classes
    rarely confused with benign traffic (< ``min_count`` such flows) keep the global
    threshold (left out)."""
    import numpy as np

    names = list(data.label_names)
    if "BENIGN" not in names:
        return {}
    b = names.index("BENIGN")
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(data.X_val, dtype=torch.float32, device=device)
        p = torch.softmax(model(x) / temperature, 1).cpu().numpy()
    benign = p[np.asarray(data.y_val) == b]
    mass = 1 - benign[:, b]
    att = benign.copy()
    att[:, b] = -1
    likely = att.argmax(1)
    out = {}
    for c, name in enumerate(names):
        sel = mass[likely == c]
        if c != b and len(sel) >= min_count:
            out[name] = float(max(0.5, np.quantile(sel, 1 - target_fpr)))
    return out


def _fit_doctor(
    data: Any,
    cfg: Any,
    model: torch.nn.Module,
    fit: dict[str, Any],
    device: str,
    n_classes: int,
    rounds: int = 2,
) -> tuple[Any, torch.nn.Module, list[dict[str, Any]]]:
    """Apply the fit check's tweaks (src/eval/fitcheck.py) and retrain; a round is
    kept only if VALIDATION macro-F1 improves. At most ``rounds`` rounds."""
    log: list[dict[str, Any]] = []
    best_val = eval_macro_f1(model, data.X_val, data.y_val, n_classes, device)
    for _ in range(rounds):
        tw = set(fit.get("tweaks", []))
        if not tw & {"regularize", "train_longer"}:
            break
        c = cfg
        if "regularize" in tw:  # overfitting: more dropout + weight decay
            c = replace(
                c,
                model=replace(c.model, dropout=min(0.5, c.model.dropout + 0.1)),
                training=replace(c.training, weight_decay=c.training.weight_decay * 10),
            )
        if "train_longer" in tw:  # under-trained / noisy stop: longer, gentler
            c = replace(
                c,
                training=replace(
                    c.training,
                    max_epochs=int(c.training.max_epochs * 1.5),
                    patience=c.training.patience * 2,
                    lr=c.training.lr * 0.5,
                ),
            )
        m2, _, met = train_centralized(data, c, device)
        v2 = eval_macro_f1(m2, data.X_val, data.y_val, n_classes, device)
        kept = v2 > best_val
        log.append(
            {
                "tweaks": sorted(tw),
                "fit_before": fit.get("verdict"),
                "fit_after": met["fit"]["verdict"],
                "val_before": best_val,
                "val_after": v2,
                "kept": kept,
            }  # fmt: skip
        )
        _log.info("fit doctor: %s -> val %.3f -> %.3f (%s)", sorted(tw), best_val, v2,
                  "kept" if kept else "reverted")  # fmt: skip
        if not kept:
            break
        cfg, model, fit, best_val = c, m2, met["fit"], v2
    return cfg, model, log


def _tuning_grid(cfg: Any, fast: bool, seed: int) -> list[tuple[str, Any]]:
    """Wider/deeper MLPs trained longer (more patience); class weights kept."""
    # Fit diagnostic (results/fit_diagnostic.json): width does not help (4x512 was
    # worse); the limit is a noisy validation curve that early stopping stops on.
    # So: gentler learning rates, longer patience, the research size included.
    grid = [(128, 4, 0.2, 5e-4), (128, 4, 0.2, 1e-3), (256, 4, 0.1, 5e-4),
            (256, 5, 0.2, 5e-4), (384, 4, 0.1, 5e-4)]  # fmt: skip
    if fast:
        grid = grid[:1]
    out = []
    for w, L, dr, lr in grid:
        c = replace(
            cfg,
            model=replace(cfg.model, hidden_width=w, hidden_layers=L, dropout=dr),
            training=replace(
                cfg.training,
                lr=lr,
                max_epochs=cfg.training.max_epochs if fast else 100,
                patience=cfg.training.patience if fast else 15,
            ),
        )
        out.append((f"tuned {L}x{w} dropout {dr} lr {lr:g}", c))
    return out


def _write_report(
    dataset: str,
    data: Any,
    model: torch.nn.Module,
    rows: list,
    best_i: int,
    device: str,
    prep: dict[str, Any] | None = None,
    ensemble_rows: list[dict[str, Any]] | None = None,
) -> None:
    """results/deploy_report.json: candidates, per-class test F1, and the alert-rule
    check (top-class rule vs total-attack-probability rule) on held-out test flows."""
    import numpy as np

    from src.eval.metrics import classification_metrics
    from src.training.centralized import predict
    from utils.paths import RESULTS_DIR

    pred = predict(model, data.X_test, device)
    per_class = classification_metrics(data.y_test, pred, data.label_names)["per_class"]
    with torch.no_grad():
        logits = model(torch.as_tensor(data.X_test, dtype=torch.float32, device=device))
        # the same calibrated probabilities live alerting uses (src/soc/engine.py)
        temp = _fit_temperature(model, data.X_val, data.y_val, device)
        p = torch.softmax(logits / temp, 1).cpu().numpy()
    names = list(data.label_names)
    alerting: dict[str, Any] = {}
    if "BENIGN" in names:
        b = names.index("BENIGN")
        truth = np.asarray(data.y_test) != b
        attack = p.copy()
        attack[:, b] = -1
        top_rule = (p.argmax(1) != b) & (attack.max(1) >= 0.5)
        mass_rule = (1 - p[:, b]) >= 0.5

        def score(alert: np.ndarray) -> dict[str, float]:
            tp, fp = int((alert & truth).sum()), int((alert & ~truth).sum())
            fn = int((~alert & truth).sum())
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec),
                    "false_alarms": fp, "missed_attacks": fn}  # fmt: skip

        alerting = {
            "top_class_rule": score(top_rule),
            "attack_probability_rule": score(mass_rule),
        }
        high = _alert_threshold(model, data, temp, device, target_fpr=0.01)
        if high is not None:
            alerting["two_tier_confirmed"] = score((1 - p[:, b]) >= high) | {
                "threshold": high
            }
            by_class = _class_thresholds(model, data, temp, device, target_fpr=0.01)
            thr = np.array([by_class.get(n, high) for n in names])[attack.argmax(1)]
            alerting["two_tier_confirmed_per_class"] = score((1 - p[:, b]) >= thr) | {
                "thresholds": by_class
            }
    qhat = conformal_qhat(model, data.X_val, data.y_val, temp, device)
    sets = conformal_sets(p, qhat)
    sizes = sets.sum(1)
    conformal = {
        "alpha": CONFORMAL_ALPHA,
        "qhat": qhat,
        "test_coverage": float(
            sets[np.arange(len(sets)), np.asarray(data.y_test)].mean()
        ),
        "mean_set_size": float(sizes.mean()),
        "share_unsure": float((sizes != 1).mean()),
    }
    report = {
        "dataset": dataset,
        "conformal": conformal,
        "candidates": rows,
        "deployed": rows[best_i]["name"],
        "per_class_test": per_class,
        "alerting_on_test": alerting,
        "temperature": temp,
        "preprocessing": prep or {"chosen": "standard"},
        "ensemble": ensemble_rows or [],
        "note": "selection on validation macro-F1; test reported only",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "deploy_report.json").write_text(
        json.dumps(report, indent=2), "utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze a deployable detector")
    ap.add_argument("--dataset", default="cicids2017-improved")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    configure_logging()
    set_seed(args.seed)
    deploy(args.dataset, args.fast, args.seed)


if __name__ == "__main__":
    main()
