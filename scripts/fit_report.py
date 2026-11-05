"""Over/underfitting report for every deployed model (owner 2026-10-16).

For each model: a LEARNING CURVE (score on its own training rows and on validation
while the training set grows: 5 %, 10 %, 25 %, 50 %, 100 %) and the train /
validation / test GAP at full size, with a verdict from src/eval/fitcheck.py:

  good fit       small gap, validation flat at full size
  overfitting    train well above validation and more data no longer closes it
  underfitting   train and validation both low (model or features too weak)
  data-limited   validation still rising at full size (more data should help)
  split shift    validation and test disagree (the splits come from different
                 conditions; a threshold set on validation may not transfer)

The curves use the deployed model's family and settings, refitted here on the same
training split (never the blind sets; test is only reported). ShieldLM has no curve
(one long training run): its train vs held-out perplexity gap is reported.

    python -m scripts.fit_report                    # all models
    python -m scripts.fit_report --only phishing ics
Writes results/extra/fit_report.{json,md}; src/eval/fitcheck.model_report() reads it.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from src.eval import fitcheck
from utils.paths import PROJECT_ROOT, RESULTS_DIR

OUT = RESULTS_DIR / "extra" / "fit_report.json"
SIZES = (0.05, 0.1, 0.25, 0.5, 1.0)
MAX_EVAL = 50_000  # rows scored per split for the curve (seeded sample)
SEED = 0


def _sub(n: int, frac: float, y: np.ndarray | None = None, seed: int = SEED) -> np.ndarray:
    """Seeded (stratified when y is given) fraction of ``range(n)``."""
    idx = np.arange(n)
    if frac >= 1.0:
        return idx
    if y is None:
        return np.sort(np.random.default_rng(seed).choice(idx, size=max(1, int(n * frac)), replace=False))
    from sklearn.model_selection import train_test_split

    keep, _ = train_test_split(idx, train_size=frac, random_state=seed, stratify=y)
    return np.sort(keep)


def _cap(X: np.ndarray, y: np.ndarray, m: int = MAX_EVAL) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= m:
        return X, y
    i = np.sort(np.random.default_rng(SEED).choice(len(y), size=m, replace=False))
    return X[i], y[i]


def curve(fit: Callable[[np.ndarray, np.ndarray], Any], score: Callable[[Any, np.ndarray, np.ndarray], float],
          Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
          Xte: np.ndarray | None = None, yte: np.ndarray | None = None,
          stratify: bool = True) -> dict[str, Any]:  # fmt: skip
    """Learning curve + full-size split verdict for one tabular model."""
    pts = []
    Xv, yv = _cap(Xva, yva)
    model = None
    for frac in SIZES:
        i = _sub(len(ytr), frac, ytr if stratify else None)
        t0 = time.time()
        model = fit(Xtr[i], ytr[i])
        Xs, ys = _cap(Xtr[i], ytr[i])
        pts.append({"frac": frac, "n": int(len(i)), "train": round(score(model, Xs, ys), 4),
                    "val": round(score(model, Xv, yv), 4), "fit_s": round(time.time() - t0, 1)})  # fmt: skip
        print(f"   {frac:>5.0%} n={len(i):>8,} train {pts[-1]['train']:.4f} val {pts[-1]['val']:.4f}",
              flush=True)  # fmt: skip
    test = None
    if Xte is not None and yte is not None:
        Xt, yt = _cap(Xte, yte)
        test = score(model, Xt, yt)
    last = pts[-1]
    return {"points": pts, "curve": fitcheck.curve_verdict(pts),
            "split": fitcheck.split_verdict(last["train"], last["val"], test)}  # fmt: skip


def _f1(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y, model.predict(X), zero_division=0))


def _macro_f1(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y, model.predict(X), average="macro", zero_division=0))


# --- models ---------------------------------------------------------------------------


def phishing() -> dict[str, Any]:
    from src.mlareas import phishing as ph

    b = ph.load(refresh=True)
    data = ph.load_phiusiil()
    X, y, _n, _k, _s = ph._variant_data(data, "url_host")
    if b.get("mode", "host") != "host":
        X = ph.featurize(_k, b["mode"])
    tr, va, te = ph.split_indices(y, int(b.get("seed", 0)))
    out = curve(lambda a, c: ph.make_model("lightgbm", 0).fit(a, c), _f1,
                X[tr], y[tr], X[va], y[va], X[te], y[te])  # fmt: skip
    out |= {"metric": "F1 (phishing class, 0.5 cut)", "learner": f"LightGBM, features {b.get('mode', 'host')}",
            "data": "PhiUSIIL hosts (deployed split)"}  # fmt: skip
    return out


def malware() -> dict[str, Any]:
    import lightgbm as lgb

    from scripts import train_malware as tm
    from src.mlareas.malware_data import Stacked

    d = tm.load_split()
    xs, xt = Stacked(d["tr_paths"]), Stacked(d["te_paths"])
    ytr_all = np.asarray(d["mtr"]["label"])
    rows = d["split"].train
    rows = rows[ytr_all[rows] >= 0]
    rows = np.sort(np.random.default_rng(SEED).choice(rows, size=min(60_000, rows.size), replace=False))
    Xtr, ytr = xs.take(rows), ytr_all[rows].astype(int)
    vrows = d["split"].val
    vrows = np.sort(np.random.default_rng(SEED).choice(vrows, size=min(MAX_EVAL, vrows.size), replace=False))
    Xva, yva = xs.take(vrows), ytr_all[vrows].astype(int)
    trows = np.sort(np.random.default_rng(SEED).choice(d["test_rows"], size=min(MAX_EVAL, len(d["test_rows"])),
                                                       replace=False))  # fmt: skip
    Xte, yte = xt.take(trows), np.asarray(d["mte"]["label"])[trows].astype(int)
    params = {**tm.LGB_BASE, "num_leaves": 512, "learning_rate": 0.1, "num_threads": 6, "seed": 0}

    class Booster:
        def __init__(self, a: np.ndarray, c: np.ndarray) -> None:
            self.b = lgb.train(params, lgb.Dataset(a, label=c), num_boost_round=400)

        def predict(self, a: np.ndarray) -> np.ndarray:
            return (np.asarray(self.b.predict(a)) >= 0.5).astype(int)

    out = curve(Booster, _f1, Xtr, ytr, Xva, yva, Xte, yte)
    out |= {"metric": "F1 (malware, 0.5 cut)", "learner": "LightGBM 512 leaves (deployed family)",
            "data": "EMBER 2018: 60,000 training rows sample, time-split validation and test"}  # fmt: skip
    return out


def _lgbm_multi(a: np.ndarray, c: np.ndarray) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                          reg_lambda=1.0, random_state=SEED, n_jobs=6, verbose=-1).fit(a, c)  # fmt: skip


def flow() -> dict[str, Any]:
    d = np.load(PROJECT_ROOT / "data" / "processed" / "improve_flows" / "cic17.npz")
    out = curve(_lgbm_multi, _macro_f1, d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                d["X_test"], d["y_test"])  # fmt: skip
    out |= {"metric": "macro-F1 (all classes)", "learner": "LightGBM member of the deployed ensemble",
            "data": "CIC-IDS2017 (deploy.py split)"}  # fmt: skip
    return out


def netflow() -> dict[str, Any]:
    from scripts import deploy_netflow as dn

    (data, _cfg), _att = dn.source_data(PROJECT_ROOT, 0)
    tr, va, te = dn.raw_splits(data)
    out = curve(_lgbm_multi, _macro_f1, tr.astype(np.float32), np.asarray(data.y_train),
                va.astype(np.float32), np.asarray(data.y_val), te.astype(np.float32),
                np.asarray(data.y_test))  # fmt: skip
    out |= {"metric": "macro-F1", "learner": "LightGBM member of the deployed ensemble",
            "data": "NF-CSE-CIC-IDS2018-v2 (deploy_netflow split, seed 0)"}  # fmt: skip
    return out


def zeek() -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier

    from scripts.improve_zeek import _data

    d = _data()

    def fit(a: np.ndarray, c: np.ndarray) -> Any:
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features="sqrt",
                                      class_weight="balanced", random_state=SEED, n_jobs=6).fit(a, c)  # fmt: skip

    out = curve(fit, _f1, *d["train"], *d["val"], *d["test"])
    out |= {"metric": "F1 (malicious)", "learner": "random forest (deployed family)",
            "data": "IoT-23 captures (ZEEK_SPLIT; test captures never in training)"}  # fmt: skip
    return out


def insider() -> dict[str, Any]:
    import pandas as pd

    from src.mlareas import insider as ins
    from src.mlareas import insider_features as feats

    base = ins.area_dir() / "data"
    tr, va = pd.read_parquet(base / "train.parquet"), pd.read_parquet(base / "val.parquet")
    cols = list(feats.feature_columns())
    params = json.loads((RESULTS_DIR / "extra" / "improve_insider.json").read_text("utf-8"))["chosen_params"]

    def fit(a: np.ndarray, c: np.ndarray) -> Any:
        return ins.make_model("lightgbm", SEED, params).fit(a, c)

    def pr_auc(model: Any, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y, model.predict_proba(X)[:, 1])) if y.any() else math.nan

    out = curve(fit, pr_auc, tr[cols].to_numpy(np.float32), tr["label"].to_numpy(),
                va[cols].to_numpy(np.float32), va["label"].to_numpy())  # fmt: skip
    out |= {"metric": "PR-AUC (few positives: a threshold-free score)", "learner": "LightGBM (deployed settings)",
            "data": "CERT r4.2 user-days, time split (test not stored in the SOC copy)"}  # fmt: skip
    return out


def ics() -> dict[str, Any]:
    from src.mlareas import ics as ic

    sp = ic.make_splits(*ic.load_hai())
    idx = ic.valid_index(sp.train_lengths)
    pts = []
    res: dict[str, Any] = {}
    for frac in SIZES:
        sub = idx[_sub(len(idx), frac)]
        t0 = time.time()
        det = ic.fit_detector("autoencoder", sp.train, sub, SEED, epochs=5,
                              per_epoch=min(200_000, len(sub)))  # fmt: skip
        res, _thr, _pct, _ = ic._evaluate(det, sp)
        pts.append({"frac": frac, "n": int(len(sub)), "train": round(res["train"]["f1"], 4),
                    "val": round(res["val"]["f1"], 4), "test_precision": round(res["test"]["precision"], 4),
                    "test_recall": round(res["test"]["recall"], 4), "fit_s": round(time.time() - t0, 1)})  # fmt: skip
        print(f"   {frac:>5.0%} n={len(sub):>8,} train {pts[-1]['train']:.4f} val {pts[-1]['val']:.4f}",
              flush=True)  # fmt: skip
        del det
        gc.collect()
    last = pts[-1]
    return {"points": pts, "curve": fitcheck.curve_verdict(pts),
            "split": fitcheck.split_verdict(last["train"], last["val"], res["test"]["f1"]),
            "metric": "F1 on injected faults (train slice, validation); test = real HAI attacks",
            "learner": "window autoencoder (deployed family, deployed normalisation)",
            "data": "HAI 22.04 (normal training, chronological validation, labelled test)"}  # fmt: skip


def intent() -> dict[str, Any]:
    from scripts import improve_intent as ii
    from src.nlp import intent as it

    cfg = json.loads((RESULTS_DIR / "extra" / "improve_intent.json").read_text("utf-8"))["chosen"]["config"]
    cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()}  # JSON lists -> ranges
    xtr, ytr, xte, yte = it.dataset(0)
    texts, labels = np.array(xtr, dtype=object), np.array(ytr, dtype=object)
    tx, ty = np.array(xte, dtype=object), np.array(yte, dtype=object)

    def acc(model: Any, X: np.ndarray, y: np.ndarray) -> float:
        return float((model.predict(list(X)) == y).mean())

    def fit(a: np.ndarray, c: np.ndarray) -> Any:  # train-only augmentation, as deployed
        import random

        rng = random.Random(1)
        return ii.build(cfg).fit(list(a) + [it.augment(x, rng) for x in a], list(c) * 2)

    out = curve(fit, acc, texts, labels, tx, ty, stratify=False)
    out |= {"metric": "accuracy", "learner": "TF-IDF word+char logistic regression (deployed settings)",
            "data": "training templates; 'validation' = the held-out templates (no choice made on them here)"}  # fmt: skip
    return out


def shieldlm() -> dict[str, Any]:
    m = json.loads((RESULTS_DIR / "nlp_metrics.json").read_text("utf-8"))["shieldlm"]
    train_ppl, val_ppl = math.exp(float(m["final_train_loss"])), float(m["heldout_perplexity"])
    s = fitcheck.split_verdict(1 / train_ppl, 1 / val_ppl)
    s["note"] = (f"perplexity train {train_ppl:.3f} vs held-out {val_ppl:.3f} (scored as 1/perplexity); "
                 "narratives are templated, so very low perplexity on both is expected")  # fmt: skip
    return {"points": [], "curve": {"verdict": "unknown", "findings": []}, "split": s,
            "metric": "1 / perplexity", "learner": "ShieldLM (one training run; no curve)",
            "data": "incident narratives (nlp_metrics.json)"}  # fmt: skip


MODELS: dict[str, Callable[[], dict[str, Any]]] = {
    "flow": flow, "netflow": netflow, "zeek": zeek, "phishing": phishing, "malware": malware,
    "insider": insider, "ics": ics, "intent": intent, "shieldlm": shieldlm,
}  # fmt: skip


def write_md(doc: dict[str, Any]) -> None:
    lines = ["# Model fit report", "", f"Generated {doc['generated']}. Curves refit the deployed "
             "model family on growing shares of its own training split; blind sets never used.", "",
             "| Model | Metric | Train | Validation | Test | Gap | Split verdict | Curve verdict |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, m in doc["models"].items():
        if "error" in m:
            lines.append(f"| {name} | - | - | - | - | - | not run: {m['error'][:60]} | - |")
            continue
        s = m["split"]
        lines.append(f"| {name} | {m['metric']} | {s['train']} | {s['val']} | {s['test'] if s['test'] is not None else '-'} "
                     f"| {s['gap']} | {s['verdict']} | {m['curve']['verdict']} |")  # fmt: skip
    lines += ["", "## Findings", ""]
    for name, m in doc["models"].items():
        f = [*m.get("split", {}).get("findings", []), *m.get("curve", {}).get("findings", [])]
        if m.get("split", {}).get("note"):
            f.append(m["split"]["note"])
        lines.append(f"- **{name}**: " + ("; ".join(f) if f else "no problem found"))
    lines += ["", "## Learning curves (train / validation)", ""]
    for name, m in doc["models"].items():
        if m.get("points"):
            lines.append(f"- **{name}** ({m['learner']}): " + ", ".join(
                f"{p['n']:,}: {p['train']}/{p['val']}" for p in m["points"]))
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", "utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=list(MODELS))
    a = ap.parse_args()
    try:
        doc = json.loads(OUT.read_text("utf-8"))
    except (OSError, ValueError):
        doc = {"models": {}}
    for name in a.only or list(MODELS):
        print(f"== {name}", flush=True)
        t0 = time.time()
        try:
            doc["models"][name] = MODELS[name]() | {"seconds": round(time.time() - t0)}
        except Exception as exc:  # noqa: BLE001 - one missing dataset must not stop the report
            doc["models"][name] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        gc.collect()
        doc["models"] = {k: doc["models"][k] for k in MODELS if k in doc["models"]}
        doc["generated"] = time.strftime("%Y-%m-%d %H:%M")
        OUT.write_text(json.dumps(doc, indent=1, default=float), "utf-8")
        write_md(doc)
        print(json.dumps({k: doc["models"][name].get(k) for k in ("split", "curve", "error")}, default=float)[:600],
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
