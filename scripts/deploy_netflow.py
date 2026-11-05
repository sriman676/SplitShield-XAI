"""Train + freeze a deployable NetFlow v9 detector (owner 2026-10-15).

Purpose: a SOC detector for networks that export NetFlow/IPFIX instead of
CICFlowMeter features. Same recipe as ``deploy.py``, binary BENIGN vs ATTACK:

  1. data: NF-CSE-CIC-IDS2018-v2 (registry id ``nf-cse-cic-ids2018-v2``), read in
     batches with a per-class pre-sample (memory guard: the 17M-row parquet is never
     held whole), then the registered pipeline ``preprocess_frame`` with the E8
     experiment config (dedup, NaN/Inf clean, class cap 60k, 70/15/15 stratified
     split, scaler fit on train only);
  2. preprocessing: plain vs robust scaling, chosen on VALIDATION with a probe MLP;
  3. members: MLP (two tuning candidates, best on validation), LightGBM (capped
     balanced weights), Random Forest; ensemble weighting chosen on VALIDATION;
  4. temperature fitted on validation; alert threshold = attack probability that
     ~1 % of validation benign flows exceed (floor 0.5, as deploy.py);
  5. TEST reported only; transfer checked on NF-UNSW-NB15-v2 and NF-ToN-IoT-v2.

Seeds 0-2 are trained and reported; seed 0's model is saved to
``artifacts/deploy_netflow/`` (loader: ``src/soc/netflow_model.py``).

    python -m scripts.deploy_netflow --root "<project>" [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.logging_config import configure_logging, get_logger
from utils.paths import PROJECT_ROOT

_log = get_logger(__name__)
SOURCE = "nf-cse-cic-ids2018-v2"
TRANSFER = {
    "nf-unsw-nb15-v2": "nf-unsw-nb15-v2/NF-UNSW-NB15-V2.parquet",
    "nf-ton-iot-v2": "nf-ton-iot-v2/NF-ToN-IoT-v2-test.csv",
}
# never features: endpoint identity / labels; the OS-chosen source port and the
# random DNS transaction id are per-flow noise a model can only memorise
DROP = (
    "Attack",
    "Label",
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "DNS_QUERY_ID",
)
PRE_PER_CLASS = 250_000  # pre-sample before dedup; the E8 cap (60k) applies after
TRANSFER_PER_CLASS = 50_000
TARGET_FPR = 0.01


# --- reading (memory-bounded) ---------------------------------------------------


def _label_counts(path: Path) -> pd.Series:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        lab = pq.read_table(path, columns=["Label"]).column(0).to_pandas()
    else:
        import pyarrow.csv as pacsv

        opts = pacsv.ConvertOptions(include_columns=["Label"])
        lab = pacsv.read_csv(path, convert_options=opts).column(0).to_pandas()
    return pd.Series(lab).astype(int).value_counts()


def _batches(path: Path, cols: list[str] | None) -> Any:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for b in pq.ParquetFile(path).iter_batches(batch_size=500_000, columns=cols):
            yield b.to_pandas()
    else:
        yield from pd.read_csv(path, chunksize=500_000, usecols=cols, low_memory=True)


def sample_netflow(path: Path, per_class: int, seed: int = 0) -> pd.DataFrame:
    """Per-class random sample (~``per_class`` rows per Label) read in batches."""
    counts = _label_counts(path)
    frac = (per_class / counts).clip(upper=1.0)
    rng = np.random.default_rng(seed)
    parts = []
    for chunk in _batches(path, None):
        f = chunk["Label"].astype(int).map(frac).fillna(1.0).to_numpy()
        parts.append(chunk[rng.random(len(chunk)) < f])
    out = pd.concat(parts, ignore_index=True)
    _log.info("%s: sampled %d of %d rows", path.name, len(out), int(counts.sum()))
    return out


# --- data -----------------------------------------------------------------------


def source_data(root: Path, seed: int) -> tuple[Any, dict[int, str]]:
    """ProcessedData for the source + test-row attack names (for per-type recall)."""
    import src.data.sources as sources
    from config.base import build_config, experiment_for_dataset
    from src.data.preprocess import preprocess_frame

    spec = sources.get_spec(SOURCE)
    df = sample_netflow(
        root / "data" / "raw" / SOURCE / "NF-CSE-CIC-IDS2018-V2.parquet", PRE_PER_CLASS
    )
    keep_id = tuple(c for c in DROP if c in df.columns and c != spec.label_col)
    spec = replace(spec, id_columns=keep_id)
    cfg = build_config(experiment_for_dataset(SOURCE, False), seed=seed)
    data = preprocess_frame(df, spec, cfg, seed=seed)
    feats = list(data.feature_names)
    h_src = pd.util.hash_pandas_object(df[feats], index=False).to_numpy()
    is_att = df["Label"].astype(int).to_numpy() != 0
    # attack names from attack rows only; a test attack whose exact feature vector
    # also occurs as a BENIGN row is tagged, since no model can separate it
    name_of = dict(zip(h_src[is_att], df["Attack"].astype(str)[is_att], strict=True))
    benign_h = set(h_src[~is_att].tolist())
    h_out = pd.util.hash_pandas_object(data.frame[feats], index=False).to_numpy()
    attack = {
        i: name_of.get(h_out[r], "?")
        + (" [identical to a benign flow]" if h_out[r] in benign_h else "")
        for i, r in enumerate(data.idx_test)
    }
    return (data, cfg), attack


def raw_splits(data: Any) -> list[np.ndarray]:
    """Raw-unit train/val/test rows straight from the preprocessed frame.

    NF-CSE holds a few garbage rates (up to 6.6e304 bytes/s) that overflow the
    plain scaler's variance, so inverting a fitted scaler is not safe."""
    full = data.frame[list(data.feature_names)].to_numpy(dtype=np.float64)
    return [full[i] for i in (data.idx_train, data.idx_val, data.idx_test)]


def robust_variant(data: Any) -> Any:
    """Same split, RobustFlowScaler (clip -> log1p heavy tails -> standardize)."""
    from src.data.robust import RobustFlowScaler

    raw = raw_splits(data)
    rob = RobustFlowScaler().fit(raw[0])
    return replace(
        data,
        X_train=rob.transform(raw[0]).astype(np.float32),
        X_val=rob.transform(raw[1]).astype(np.float32),
        X_test=rob.transform(raw[2]).astype(np.float32),
        scaler=rob,
    )


_TRANSFER_CACHE: dict[str, pd.DataFrame] = {}


def transfer_frame(
    root: Path, rel: str, feats: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Raw features (by name) + binary labels of a capped transfer sample."""
    key = str(root / rel)
    if key not in _TRANSFER_CACHE:  # same sample for every seed, read once
        _TRANSFER_CACHE[key] = sample_netflow(
            root / "data" / "raw" / rel, TRANSFER_PER_CLASS, seed=1
        )
    df = _TRANSFER_CACHE[key]
    X = np.column_stack(
        [pd.to_numeric(df[f], errors="coerce").to_numpy(np.float64) for f in feats]
    )
    ok = np.isfinite(X).all(1)
    return X[ok], (df["Label"].astype(int).to_numpy() != 0)[ok].astype(np.int64)


# --- training -------------------------------------------------------------------


def mlp_grid(cfg: Any) -> list[tuple[str, Any]]:
    """Two of deploy.py's tuning candidates (longer training, gentle LR)."""
    out = []
    for w, layers, dr, lr in ((128, 4, 0.2, 5e-4), (256, 4, 0.1, 5e-4)):
        c = replace(
            cfg,
            model=replace(cfg.model, hidden_width=w, hidden_layers=layers, dropout=dr),
            training=replace(cfg.training, lr=lr, max_epochs=100, patience=15),
        )
        out.append((f"mlp {layers}x{w} dropout {dr}", c))
    return out


def fit_trees(data: Any, seed: int) -> dict[str, Any]:
    """LightGBM (main's current recipe: capped balanced weights) + Random Forest."""
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    w = compute_sample_weight("balanced", data.y_train)
    w = np.minimum(w, 50.0 * float(np.median(w)))
    gbm = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                         reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)  # fmt: skip
    gbm.fit(data.X_train, data.y_train, sample_weight=w)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=seed)  # fmt: skip
    rf.fit(data.X_train, data.y_train)
    return {"lgbm": gbm, "rf": rf}


def probs(model: Any, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 50_000):
            x = torch.as_tensor(np.asarray(X[i : i + 50_000], dtype=np.float32))
            out.append(torch.softmax(model(x) / temperature, 1).numpy())
    return np.concatenate(out)


def fit_temperature(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    """T minimising validation log-loss of softmax(logits / T) (grid 0.1..10)."""
    import torch

    with torch.no_grad():
        logits = model(torch.as_tensor(np.asarray(X, dtype=np.float32)))
    t = torch.as_tensor(y, dtype=torch.long)
    grid = torch.logspace(-1, 1, 81)
    nll = [float(torch.nn.functional.cross_entropy(logits / g, t)) for g in grid]
    return float(grid[int(np.argmin(nll))])


def alert_threshold(
    p_attack_benign_val: np.ndarray, target_fpr: float = TARGET_FPR
) -> float:
    """Attack probability ~``target_fpr`` of validation benign flows exceed (>= 0.5)."""
    return float(max(0.5, np.quantile(p_attack_benign_val, 1 - target_fpr)))


def split_metrics(p: np.ndarray, y: np.ndarray, thr: float) -> dict[str, Any]:
    """Macro-F1/accuracy (argmax), AUC, and alert counts at ``thr``."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    pa = p[:, 1]
    alert = pa >= thr
    att = y == 1
    tp, fp = int((alert & att).sum()), int((alert & ~att).sum())
    return {
        "macro_f1": float(
            f1_score(y, p.argmax(1), average="macro", labels=[0, 1], zero_division=0)
        ),
        "accuracy": float(accuracy_score(y, p.argmax(1))),
        "roc_auc": float(roc_auc_score(y, pa)) if len(np.unique(y)) == 2 else None,
        "attacks": int(att.sum()),
        "benign": int((~att).sum()),
        "attacks_caught": tp,
        "recall_at_threshold": tp / max(1, int(att.sum())),
        "false_alarms": fp,
        "false_alarm_rate": fp / max(1, int((~att).sum())),
        "precision_at_threshold": tp / max(1, tp + fp),
    }


def fit_verdict(train_f1: float, val_f1: float) -> str:
    gap = train_f1 - val_f1
    if gap > 0.05:
        return f"overfitting (train - val = {gap:.3f} > 0.05)"
    return f"good fit (train - val = {gap:.3f})"


def train_seed(root: Path, seed: int) -> dict[str, Any]:
    """Full recipe for one seed; returns metrics + the fitted objects."""
    from src.models.ensemble import choose
    from src.training.centralized import train_centralized
    from src.training.common import eval_macro_f1
    from utils.seed import set_seed

    set_seed(seed)
    (data, cfg), attack_names = source_data(root, seed)
    grid = mlp_grid(cfg)
    probe: dict[str, tuple[Any, Any, float]] = {}
    for prep, d in (("standard", data), ("robust", robust_variant(data))):
        m, _, met = train_centralized(d, grid[0][1], "cpu")
        probe[prep] = (d, m, eval_macro_f1(m, d.X_val, d.y_val, 2, "cpu"))
    prep = max(probe, key=lambda k: probe[k][2])
    data, mlp, best = probe[prep]
    mlp_rows = [{"name": grid[0][0], "val_macro_f1": best}]
    for name, c in grid[1:]:
        m, _, _ = train_centralized(data, c, "cpu")
        v = eval_macro_f1(m, data.X_val, data.y_val, 2, "cpu")
        mlp_rows.append({"name": name, "val_macro_f1": v})
        if v > best:
            mlp, best = m, v
    trees = fit_trees(data, seed)
    scorer, ens_rows = choose(
        mlp,
        trees,
        data,
        "cpu",
        evaluate=lambda m: eval_macro_f1(m, data.X_val, data.y_val, 2, "cpu"),
    )
    T = fit_temperature(scorer, data.X_val, data.y_val)
    pv = probs(scorer, data.X_val, T)
    thr = alert_threshold(pv[data.y_val == 0, 1])
    splits = {
        s: split_metrics(
            probs(scorer, getattr(data, f"X_{s}"), T), getattr(data, f"y_{s}"), thr
        )
        for s in ("train", "val", "test")
    }
    members = _member_f1(mlp, trees, data)
    return {
        "seed": seed,
        "preprocessing": {
            "chosen": prep,
            "probe_val_macro_f1": {k: v[2] for k, v in probe.items()},
        },
        "mlp_candidates": mlp_rows,
        "ensemble_candidates": ens_rows,
        "members_chosen": sorted(scorer.weights),
        "temperature": T,
        "alert_threshold": thr,
        "splits": splits,
        "fit": fit_verdict(splits["train"]["macro_f1"], splits["val"]["macro_f1"]),
        "members_macro_f1": members,
        "test_recall_by_attack": _per_attack(
            probs(scorer, data.X_test, T)[:, 1] >= thr, data.y_test, attack_names
        ),
        "class_counts": data.stats.get("class_counts"),
        "_objects": (data, mlp, trees, scorer),
    }


def _member_f1(
    mlp: Any, trees: dict[str, Any], data: Any
) -> dict[str, dict[str, float]]:
    """Train/val/test macro-F1 of each member alone (trees fit train ~perfectly)."""
    from sklearn.metrics import f1_score

    out: dict[str, dict[str, float]] = {}
    models: dict[str, Any] = {"mlp": lambda X: probs(mlp, X).argmax(1)}
    for k, m in trees.items():
        models[k] = m.predict
    for k, fn in models.items():
        out[k] = {
            s: float(
                f1_score(
                    getattr(data, f"y_{s}"),
                    fn(getattr(data, f"X_{s}")),
                    average="macro",
                )
            )
            for s in ("train", "val", "test")
        }
    return out


def _per_attack(
    alert: np.ndarray, y: np.ndarray, names: dict[int, str]
) -> dict[str, Any]:
    by: dict[str, list[bool]] = {}
    for i in np.flatnonzero(y == 1):
        by.setdefault(names.get(int(i), "?"), []).append(bool(alert[i]))
    return {
        k: {"n": len(v), "recall": float(np.mean(v))} for k, v in sorted(by.items())
    }


def transfer(
    root: Path, feats: list[str], scaler: Any, scorer: Any, T: float, thr: float
) -> dict[str, Any]:
    out = {}
    for name, rel in TRANSFER.items():
        X, y = transfer_frame(root, rel, feats)
        out[name] = split_metrics(
            probs(scorer, np.asarray(scaler.transform(X)), T), y, thr
        )
    return out


def speed(path: Path, feats: list[str], X_raw: np.ndarray) -> dict[str, float]:
    """Flows/second through the saved artifact's loader (DataFrame in, alerts out)."""
    from src.soc.netflow_model import NetflowDetector

    det = NetflowDetector(path)
    df = pd.DataFrame(X_raw, columns=feats)
    t0 = time.perf_counter()
    det.score(df)
    dt = time.perf_counter() - t0
    return {
        "flows": len(df),
        "seconds": dt,
        "flows_per_second": len(df) / max(dt, 1e-9),
    }


# --- report ---------------------------------------------------------------------


def _ms(vals: list[float]) -> str:
    a = np.asarray(vals, dtype=float)
    return f"{a.mean():.4f} ± {a.std():.4f}"


def write_md(doc: dict[str, Any], path: Path) -> None:
    runs = doc["runs"]
    lines = [
        "# 🌐 Deployable NetFlow detector (NF-CSE-CIC-IDS2018-v2, binary)",
        "",
        f"Seeds {[r['seed'] for r in runs]}; saved model = seed {runs[0]['seed']} -> `{doc['artifacts']}`. "
        f"{len(doc['feature_names'])} NetFlow v9 features. Selection on VALIDATION; TEST reported only.",
        "",
        "## 📊 Train / validation / test (chosen detector, mean ± std over seeds)",
        "",
        "| split | macro-F1 | accuracy | ROC-AUC |",
        "|---|---|---|---|",
    ]
    for s in ("train", "val", "test"):
        lines.append(
            f"| {s} | {_ms([r['splits'][s]['macro_f1'] for r in runs])} | "
            f"{_ms([r['splits'][s]['accuracy'] for r in runs])} | {_ms([r['splits'][s]['roc_auc'] for r in runs])} |"
        )
    lines += ["", "Fit verdict per seed: " + "; ".join(f"seed {r['seed']}: {r['fit']}" for r in runs),
              "(tree members fit the training split almost perfectly by design; the verdict uses the "
              "chosen ensemble's train vs validation macro-F1).", ""]  # fmt: skip
    lines += ["Members chosen on validation: " + "; ".join(f"seed {r['seed']}: {' + '.join(r['members_chosen'])} "
              f"({r['preprocessing']['chosen']} scaling)" for r in runs), ""]  # fmt: skip
    lines += ["## 🚨 Alerts on TEST at the chosen threshold (~1 % validation false alarms)", "",
              "| seed | threshold | attacks caught | recall | false alarms | false-alarm rate | precision |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in runs:
        t = r["splits"]["test"]
        lines.append(
            f"| {r['seed']} | {r['alert_threshold']:.3f} | {t['attacks_caught']}/{t['attacks']} | {t['recall_at_threshold']:.4f} "
            f"| {t['false_alarms']}/{t['benign']} | {t['false_alarm_rate']:.4f} | {t['precision_at_threshold']:.4f} |"
        )
    lines += [
        "",
        "The threshold has deploy.py's 0.5 floor; where the validation 1 %-FPR quantile "
        "is lower, the floor applies and false alarms stay well under 1 %.",
    ]
    lines += ["", f"Recall by attack type (seed {runs[0]['seed']}, TEST):", ""]
    lines += [
        f"- {k}: {v['recall']:.3f} (n={v['n']})"
        for k, v in runs[0]["test_recall_by_attack"].items()
    ]
    lines += ["", "## 🔁 Transfer to other networks (no retraining; same threshold)", "",
              "| target | macro-F1 | ROC-AUC | recall at threshold | false-alarm rate |", "|---|---|---|---|---|"]  # fmt: skip
    for name in TRANSFER:
        rs = [r["transfer"][name] for r in runs]
        lines.append(
            f"| {name} | {_ms([x['macro_f1'] for x in rs])} | {_ms([x['roc_auc'] for x in rs])} | "
            f"{_ms([x['recall_at_threshold'] for x in rs])} | {_ms([x['false_alarm_rate'] for x in rs])} |"
        )
    sp = doc["speed"]
    lines += ["", "## ⚡ Speed", "", f"{sp['flows_per_second']:,.0f} flows/s through `score_netflow` "
              f"({sp['flows']:,} flows, single process, CPU).", "", "## 🧾 Verdict", "",
              *[f"- {v}" for v in doc["verdict"]], "", "## ⚠️ Deviations / caveats", "",
              *[f"- {c}" for c in doc["caveats"]], ""]  # fmt: skip
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(doc: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = doc["runs"]
    groups = ["test (source)", *TRANSFER]
    rec = [[r["splits"]["test"]["recall_at_threshold"] for r in runs]] + [
        [r["transfer"][t]["recall_at_threshold"] for r in runs] for t in TRANSFER
    ]
    far = [[r["splits"]["test"]["false_alarm_rate"] for r in runs]] + [
        [r["transfer"][t]["false_alarm_rate"] for r in runs] for t in TRANSFER
    ]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.bar(x - 0.18, [np.mean(v) * 100 for v in rec], 0.36, yerr=[np.std(v) * 100 for v in rec],
           color="#2a6fdb", label="attacks caught (%)", capsize=3)  # fmt: skip
    ax.bar(x + 0.18, [np.mean(v) * 100 for v in far], 0.36, yerr=[np.std(v) * 100 for v in far],
           color="#d9480f", label="false alarms (% of benign)", capsize=3)  # fmt: skip
    ax.set_xticks(x, groups)
    ax.set_ylabel("%")
    ax.set_title("NetFlow detector at the ~1% FPR threshold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def verdict(runs: list[dict[str, Any]]) -> list[str]:
    t = [r["splits"]["test"] for r in runs]
    out = [
        f"Source TEST: macro-F1 {np.mean([x['macro_f1'] for x in t]):.4f}, catches "
        f"{np.mean([x['recall_at_threshold'] for x in t]):.1%} of attacks at "
        f"{np.mean([x['false_alarm_rate'] for x in t]):.2%} false alarms."
    ]
    for name in TRANSFER:
        x = [r["transfer"][name] for r in runs]
        out.append(
            f"{name}: AUC {np.mean([v['roc_auc'] for v in x]):.3f}, catches "
            f"{np.mean([v['recall_at_threshold'] for v in x]):.1%} at "
            f"{np.mean([v['false_alarm_rate'] for v in x]):.1%} false alarms -> "
            + (
                "usable without retraining"
                if np.mean([v["macro_f1"] for v in x]) > 0.9
                else "NOT usable without retraining on that network"
            )
        )
    return out


# --- main -----------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + freeze the NetFlow detector")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "extra")
    args = ap.parse_args()
    configure_logging()
    from src.data.robust import input_stats
    from src.soc.netflow_model import save_netflow

    art = args.root / "artifacts" / "deploy_netflow"
    runs = []
    for seed in args.seeds:
        r = train_seed(args.root, seed)
        data, mlp, trees, scorer = r.pop("_objects")
        feats = list(data.feature_names)
        r["transfer"] = transfer(
            args.root,
            feats,
            data.scaler,
            scorer,
            r["temperature"],
            r["alert_threshold"],
        )
        if seed == args.seeds[0]:
            raw_tr, _, raw_te = raw_splits(data)
            meta = {
                "feature_names": feats, "label_names": list(data.label_names),
                "hidden_layers": len(mlp.blocks), "hidden_width": int(mlp.head.in_features),
                "dropout": float(mlp.blocks[0][-1].p), "norm": "none",
                "dataset": SOURCE, "seed": seed,
                "ensemble": dict(scorer.weights), "temperature": r["temperature"],
                "alert_threshold": r["alert_threshold"], "alert_target_fpr": TARGET_FPR,
                "preprocessing": r["preprocessing"]["chosen"],
                "input_stats": input_stats(data.scaler, raw_tr),
                "test_summary": r["splits"]["test"],
            }  # fmt: skip
            save_netflow(art, mlp, trees, data.scaler, meta)
            sp = speed(art, feats, raw_te)
        runs.append(r)
        _log.info("seed %d: test macro-F1 %.4f", seed, r["splits"]["test"]["macro_f1"])
    doc = {
        "artifacts": str(art),
        "feature_names": feats,
        "runs": runs,
        "speed": sp,
        "verdict": verdict(runs),
        "caveats": [
            "Binary BENIGN vs ATTACK only (NetFlow datasets share no attack taxonomy).",
            "NF-CSE-CIC-IDS2018-v2 contains garbage rates (SRC_TO_DST_SECOND_BYTES up to "
            "6.6e304 in 289 rows); when a seed's sample includes them, plain standard "
            "scaling overflows and its probe MLP fails (val macro-F1 0.33), so the "
            "validation-chosen robust scaler (clips to training quantiles) is used.",
            f"Memory guard: the source parquet is pre-sampled to ~{PRE_PER_CLASS:,} rows per "
            "label while reading, then the registered pipeline (dedup, clean, 60k class cap, "
            "split) runs on that sample; not the identical draw of the E8 research runs.",
            "Dropped as non-features (deployment choice): L4_SRC_PORT (OS-chosen ephemeral "
            "port) and DNS_QUERY_ID (random transaction id); E8 research keeps them.",
            "Class-capped data are ~50/50 benign/attack; precision on real traffic, where "
            "attacks are rare, will be lower than the TEST precision shown.",
            f"Transfer samples: up to {TRANSFER_PER_CLASS:,} flows per label from "
            "NF-UNSW-NB15-v2 and the NF-ToN-IoT-v2 'test' CSV; rows with NaN/Inf dropped.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "netflow_deploy.json").write_text(
        json.dumps(doc, indent=2, default=float), "utf-8"
    )
    write_md(doc, args.out / "netflow_deploy.md")
    plot(doc, args.out / "netflow_deploy.png")
    print(f"[deploy_netflow] -> {art}; report {args.out / 'netflow_deploy.md'}")


if __name__ == "__main__":
    main()
