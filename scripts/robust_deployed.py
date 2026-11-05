"""Evasion test of the DEPLOYED ensemble detector (owner 2026-10-15).

Purpose: the research attacks (Stage 4) target the research MLP. The SOC runs a
different model: MLP + LightGBM averaged, on robust-scaled raw CIC-IDS2017 features
(``artifacts/deploy/``). This script loads that detector READ-ONLY, rebuilds its
held-out TEST split (same dataset, experiment config and seed as ``deploy.py``) and
asks: can a constrained attacker flip detected attack flows to BENIGN?

Attack families, each at several L-inf budgets ``eps`` (robust-scaled units):

  1. random  - control: random +/-eps on the attacker-controllable features;
  2. pgd     - white-box PGD on the MLP member only, TRANSFERRED to the ensemble
               (the trees have no gradient; this is what a gradient attacker gets);
  3. query   - black-box greedy coordinate search on the ENSEMBLE's output: each
               step tries every controllable feature at the +/-eps ball edge and
               keeps the move that raises P(BENIGN) most (tree-aware: needs no
               gradient, only scores).

Realism: every candidate is projected with ``src/attack/constraints.py``
(``FeatureConstraints``, corrected formulas) in RAW units: immutable features
frozen (destination port), counts/bytes/duration may only grow, IAT/length stats
stay inside the training range, integers rounded, rates and ratios recomputed.
Success = a flow the clean ensemble flagged (argmax != BENIGN) is now BENIGN.

    python -m scripts.robust_deployed --root "<project>" [--n 2000] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from utils.logging_config import configure_logging, get_logger
from utils.paths import PROJECT_ROOT

_log = get_logger(__name__)
EPSILONS = (0.03, 0.1, 0.3)
FAMILIES = ("random", "pgd", "query")
OUT_DIR = PROJECT_ROOT / "results" / "extra"


# --- loading --------------------------------------------------------------------


def load_detector(root: Path) -> Any:
    """The deployed detector, read-only (``artifacts/deploy`` under ``root``)."""
    from src.soc.model import Detector

    return Detector(root / "artifacts" / "deploy")


def rebuild_split(root: Path, dataset: str, seed: int = 0) -> dict[str, Any]:
    """Raw-unit train range + TEST split exactly as deploy.py built them."""
    import src.data.sources as sources
    from config.base import build_config, experiment_for_dataset
    from src.data.preprocess import preprocess_frame

    sources.RAW_DIR = root / "data" / "raw"  # read the project's raw data
    cfg = build_config(experiment_for_dataset(dataset, False), seed=seed)
    spec = sources.get_spec(dataset)
    data = preprocess_frame(sources.load_raw(spec), spec, cfg, seed=seed)
    feats = list(data.feature_names)
    raw = data.frame[feats].to_numpy(dtype=np.float64)
    tr = raw[data.idx_train]
    return {
        "feature_names": feats,
        "label_names": list(data.label_names),
        "X_test_raw": raw[data.idx_test],
        "y_test": np.asarray(data.y_test),
        "train_min": tr.min(axis=0),
        "train_max": tr.max(axis=0),
        "train_mean_check": _robust_mean(tr),
    }


def _robust_mean(x_train_raw: np.ndarray) -> np.ndarray:
    from src.data.robust import RobustFlowScaler

    return RobustFlowScaler().fit(x_train_raw).mean_


def cached_split(root: Path, dataset: str, cache: Path) -> dict[str, Any]:
    """``rebuild_split`` once, then from an npz cache (the raw CSVs are ~1 GB)."""
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        out: dict[str, Any] = {k: z[k] for k in z.files}
        out["feature_names"] = [str(s) for s in z["feature_names"]]
        out["label_names"] = [str(s) for s in z["label_names"]]
        return out
    sp = rebuild_split(root, dataset)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{k: np.asarray(v) for k, v in sp.items()})
    return sp


def build_constraints(det: Any, sp: dict[str, Any]) -> Any:
    """FeatureConstraints over the deployed scaler (corrected DERIVED formulas)."""
    from src.attack.constraints import FeatureConstraints

    bounds = np.vstack([sp["train_min"], sp["train_max"]])
    return FeatureConstraints.from_run(
        list(sp["feature_names"]), det.scaler, bounds, corrected=True
    )


# --- scoring --------------------------------------------------------------------


def proba(scorer: Any, Xs: np.ndarray, batch: int = 50_000) -> np.ndarray:
    """Softmax of a torch scorer (EnsembleModel or MLP) in batches."""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), batch):
            x = torch.as_tensor(np.asarray(Xs[i : i + batch], dtype=np.float32))
            out.append(torch.softmax(scorer(x), 1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def mlp_energy(mlp: Any, Xs: np.ndarray) -> np.ndarray:
    """D3 energy score (-logsumexp of the MLP logits), as the SOC computes it."""
    import torch

    from src.defense.detector import energy_from_logits

    with torch.no_grad():
        logits = mlp(torch.as_tensor(np.asarray(Xs, dtype=np.float32))).numpy()
    return energy_from_logits(logits)


# --- attacks --------------------------------------------------------------------


def attack_random(
    x0: np.ndarray, eps: float, cons: Any, rng: np.random.Generator
) -> np.ndarray:
    """Control: +/-eps random signs on the controllable features, projected."""
    noise = rng.choice([-1.0, 1.0], size=x0.shape) * eps * cons.mask
    return np.asarray(cons.project_scaled(x0 + noise, x0))


def _increase_only(cons: Any) -> np.ndarray:
    from config.constraints import INCREASE_ONLY

    return np.array([cons.classes[n] == INCREASE_ONLY for n in cons.feature_names])


def attack_pgd(
    mlp: Any, x0: np.ndarray, eps: float, cons: Any, benign: int, iters: int = 20
) -> np.ndarray:
    """PGD toward BENIGN on the MLP member, realism-projected once at the end.

    ``src/attack/fgsm_pgd.pgd`` projects (and rounds integer counts) after EVERY
    step; with small steps the rounding undoes each step and PGD stalls below the
    random control. Here the iterate stays continuous, with the cheap scaled-space
    constraints each step (L-inf ball, frozen features fixed, increase-only
    features never below the original: the scaler is monotone per feature); the
    full raw-unit projection (rounding, bounds, DERIVED recomputation) is applied
    to the final point, exactly as for the other attacks."""
    import torch

    mask, inc = cons.mask.astype(bool), _increase_only(cons)
    x = np.array(x0, dtype=np.float64, copy=True)
    alpha = 2.5 * eps / iters
    target = torch.full((len(x0),), benign, dtype=torch.long)
    mlp.eval()
    for _ in range(iters):
        t = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        loss = torch.nn.functional.cross_entropy(mlp(t), target)
        (g,) = torch.autograd.grad(loss, t)
        x = np.clip(x - alpha * np.sign(g.numpy()), x0 - eps, x0 + eps)
        x[:, ~mask] = x0[:, ~mask]
        x[:, inc] = np.maximum(x[:, inc], x0[:, inc])
    return np.asarray(cons.project_scaled(x, x0))


def _candidates(
    x: np.ndarray, x0: np.ndarray, eps: float, cols: list[tuple[int, float]]
) -> np.ndarray:
    """(n, c, d): each row with one coordinate moved to x0 +/- eps."""
    cand = np.repeat(x[:, None, :], len(cols), axis=1)
    for c, (j, s) in enumerate(cols):
        cand[:, c, j] = x0[:, j] + s * eps
    return cand


def attack_query(
    score: Any,
    x0: np.ndarray,
    eps: float,
    cons: Any,
    benign: int,
    steps: int = 8,
    chunk: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Black-box greedy coordinate search on ``score`` (rows -> probabilities).

    Returns (adversarial examples, queries used per flow)."""
    inc = _increase_only(cons)
    cols = [
        (int(j), s)
        for j in np.flatnonzero(cons.mask)
        for s in (-1.0, 1.0)
        if not (s < 0 and inc[j])
    ]
    out = np.array(x0, dtype=np.float64, copy=True)
    queries = np.zeros(len(x0), dtype=np.int64)
    for a in range(0, len(x0), chunk):
        sl = slice(a, a + chunk)
        out[sl], queries[sl] = _greedy_chunk(
            score, x0[sl], eps, cons, benign, cols, steps
        )
    return out, queries


def _greedy_chunk(
    score: Any,
    x0: np.ndarray,
    eps: float,
    cons: Any,
    benign: int,
    cols: list[tuple[int, float]],
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(x0, copy=True)
    p = score(x)
    best = p[:, benign]
    active = p.argmax(1) != benign
    q = np.ones(len(x0), dtype=np.int64)
    d = x0.shape[1]
    for _ in range(steps):
        idx = np.flatnonzero(active)
        if not len(idx):
            break
        cand = _candidates(x[idx], x0[idx], eps, cols)
        flat = cand.reshape(-1, d)
        orig = np.repeat(x0[idx], len(cols), axis=0)
        flat = np.asarray(cons.project_scaled(flat, orig))
        pc = score(flat).reshape(len(idx), len(cols), -1)
        q[idx] += len(cols)
        k = pc[:, :, benign].argmax(1)
        gain = pc[np.arange(len(idx)), k, benign]
        better = gain > best[idx] + 1e-6
        rows = idx[better]
        x[rows] = flat.reshape(len(idx), len(cols), d)[
            np.flatnonzero(better), k[better]
        ]
        best[rows] = gain[better]
        won = pc[np.arange(len(idx)), k].argmax(1) == benign
        active[idx[~better | (better & won)]] = False
    return x, q


# --- evaluation -----------------------------------------------------------------


def clean_metrics(det: Any, Xs: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Accuracy / macro-F1 on the full TEST split: ensemble and each member."""
    from sklearn.metrics import accuracy_score, f1_score

    scorers: dict[str, Any] = {"ensemble": det.ensemble or det.model, "mlp": det.model}
    out: dict[str, Any] = {}
    for name, sc in scorers.items():
        pred = proba(sc, Xs).argmax(1)
        out[name] = {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        }
    trees = getattr(det.ensemble, "trees", {}) if det.ensemble is not None else {}
    for name, m in trees.items():
        pred = np.asarray(m.classes_)[m.predict_proba(Xs).argmax(1)]
        out[name] = {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        }
    return out


def outcome(
    det: Any,
    x0: np.ndarray,
    xa: np.ndarray,
    benign: int,
    thr: float,
    cons: Any = None,
) -> dict[str, Any]:
    """Success rates of one attack on clean-detected attack flows."""
    scorer = det.ensemble or det.model
    pa = proba(scorer, xa)
    flipped = pa.argmax(1) == benign
    below = (1 - pa[:, benign]) < thr
    mlp_flip = proba(det.model, xa).argmax(1) == benign
    thr_e = getattr(det, "energy_threshold", None)
    caught = (
        float((mlp_energy(det.model, xa[flipped]) > thr_e).mean())
        if thr_e is not None and flipped.any()
        else None
    )
    # changes are measured against the PROJECTED original, so the tiny DERIVED
    # recomputation differences of an unperturbed flow do not count as attack moves
    base = np.asarray(cons.project_scaled(x0, x0)) if cons is not None else x0
    ch_all = np.abs(xa - base) > 1e-6
    moved = ch_all.sum(1)
    rel, top = None, []
    if cons is not None and flipped.any():
        r0, ra = cons.to_raw(base[flipped]), cons.to_raw(xa[flipped])
        ch = ch_all[flipped]
        if ch.any():
            rel = float(np.median((np.abs(ra - r0) / (np.abs(r0) + 1.0))[ch]))
        counts = ch.mean(0)
        names = list(cons.feature_names)
        top = [
            [names[j], float(counts[j])]
            for j in np.argsort(-counts)[:5]
            if counts[j] > 0
        ]
    return {
        "median_raw_relative_change": rel,
        "most_changed_features_in_successes": top,
        "success_rate": float(flipped.mean()),
        "below_alert_threshold": float(below.mean()),
        "mlp_member_success": float(mlp_flip.mean()),
        "energy_flags_successes": caught,
        "features_changed_mean": float(moved[flipped].mean()) if flipped.any() else 0.0,
        "flipped": flipped,
    }


def per_class(flipped: np.ndarray, y: np.ndarray, names: list[str]) -> dict[str, float]:
    return {names[c]: float(flipped[y == c].mean()) for c in np.unique(y)}


def run_seed(
    det: Any,
    Xs: np.ndarray,
    y: np.ndarray,
    cons: Any,
    seed: int,
    n: int,
    eps_list: tuple[float, ...] = EPSILONS,
    query_steps: int = 8,
) -> dict[str, Any]:
    """All attack families x budgets for one seed's sample of detected attacks."""
    names = list(det.label_names)
    benign = names.index("BENIGN")
    scorer = det.ensemble or det.model
    pred = proba(scorer, Xs).argmax(1)
    pool = np.flatnonzero((y != benign) & (pred != benign))
    rng = np.random.default_rng(seed)
    pick = rng.choice(pool, size=min(n, len(pool)), replace=False)
    x0, y0 = Xs[pick].astype(np.float64), y[pick]
    thr = float(getattr(det, "alert_threshold_high", None) or 0.5)
    res: dict[str, Any] = {"seed": seed, "n_flows": len(pick), "attacks": {}}
    for eps in eps_list:
        for fam in FAMILIES:
            t0 = time.perf_counter()
            q = None
            if fam == "random":
                xa = attack_random(x0, eps, cons, rng)
            elif fam == "pgd":
                xa = attack_pgd(det.model, x0, eps, cons, benign)
            else:
                xa, q = attack_query(
                    lambda z: proba(scorer, z), x0, eps, cons, benign, query_steps
                )
            o = outcome(det, x0, xa, benign, thr, cons)
            flipped = o.pop("flipped")
            o["per_class_success"] = per_class(flipped, y0, names)
            o["seconds"] = round(time.perf_counter() - t0, 1)
            if q is not None:
                o["queries_mean"] = float(q.mean())
            res["attacks"][f"{fam}@{eps:g}"] = o
            _log.info(
                "seed %d %s eps %.2f: success %.3f", seed, fam, eps, o["success_rate"]
            )
    return res


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean / std over seeds per attack x budget."""
    keys = runs[0]["attacks"].keys()
    out: dict[str, Any] = {}
    for k in keys:
        vals = [r["attacks"][k] for r in runs]
        row: dict[str, Any] = {}
        for m in ("success_rate", "below_alert_threshold", "mlp_member_success"):
            a = np.array([v[m] for v in vals], dtype=float)
            row[m] = {"mean": float(a.mean()), "std": float(a.std(ddof=0))}
        e = [
            v["energy_flags_successes"]
            for v in vals
            if v["energy_flags_successes"] is not None
        ]
        row["energy_flags_successes_mean"] = float(np.mean(e)) if e else None
        rc = [
            v["median_raw_relative_change"]
            for v in vals
            if v["median_raw_relative_change"] is not None
        ]
        row["median_raw_relative_change"] = float(np.mean(rc)) if rc else None
        if "queries_mean" in vals[0]:
            row["queries_mean"] = float(np.mean([v["queries_mean"] for v in vals]))
        out[k] = row
    return out


# --- report ---------------------------------------------------------------------


def verdict(summary: dict[str, Any], eps_list: tuple[float, ...]) -> list[str]:
    """Plain statements comparing each family with the random control."""
    lines = []
    for eps in eps_list:
        r = summary[f"random@{eps:g}"]["success_rate"]["mean"]
        for fam in ("pgd", "query"):
            s = summary[f"{fam}@{eps:g}"]["success_rate"]["mean"]
            lines.append(
                f"eps {eps:g}: {fam} {s:.1%} vs random {r:.1%} "
                f"({'above' if s > r + 0.01 else 'not above'} the control)"
            )
    lo, hi = min(eps_list), max(eps_list)
    r_lo = summary[f"random@{lo:g}"]["success_rate"]["mean"]
    if r_lo > 0.05:
        lines.append(
            f"Fragility: even RANDOM noise at the smallest budget (eps {lo:g}) flips "
            f"{r_lo:.0%} of detected attacks, so part of every attack's success is "
            "not adversarial skill but sensitivity to small realistic changes."
        )
    en = [
        summary[f"{f}@{hi:g}"]["energy_flags_successes_mean"]
        for f in FAMILIES
        if summary[f"{f}@{hi:g}"]["energy_flags_successes_mean"] is not None
    ]
    if en:
        lines.append(
            f"The D3 energy detector flags only {min(en):.0%}-{max(en):.0%} of the "
            f"successful evasions at eps {hi:g}; it is not a defence against them."
        )
    return lines


def _per_class_md(doc: dict[str, Any]) -> list[str]:
    """Success per traffic class at the largest budget (mean over seeds)."""
    eps = max(doc["epsilons"])
    classes = sorted(
        {
            c
            for r in doc["runs"]
            for c in r["attacks"][f"random@{eps:g}"]["per_class_success"]
        }
    )
    lines = ["", f"### Per attack class at eps {eps:g} (mean over seeds)", "",
             "| class | " + " | ".join(FAMILIES) + " |", "|---" * (len(FAMILIES) + 1) + "|"]  # fmt: skip
    for c in classes:
        cells = []
        for fam in FAMILIES:
            v = [
                r["attacks"][f"{fam}@{eps:g}"]["per_class_success"].get(c)
                for r in doc["runs"]
            ]
            v = [x for x in v if x is not None]
            cells.append(f"{np.mean(v):.0%}" if v else "n/a")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    r0 = doc["runs"][0]
    lines += [
        "",
        f"Features most often changed in successful flips (eps {eps:g}, seed {r0['seed']}; share of successes):",
        "",
    ]
    for fam in FAMILIES:
        top = r0["attacks"][f"{fam}@{eps:g}"].get(
            "most_changed_features_in_successes", []
        )
        lines.append(
            f"- {fam}: " + (", ".join(f"{n} {s:.0%}" for n, s in top) or "none")
        )
    return lines


def write_md(doc: dict[str, Any], path: Path) -> None:
    s, eps_list = doc["summary"], tuple(doc["epsilons"])
    lines = [
        "# 🛡️ Evasion test of the deployed detector",
        "",
        f"Detector: `{doc['detector']}` (members {doc['members']}), dataset "
        f"`{doc['dataset']}`, TEST split only. {doc['n_flows']} detected attack flows "
        f"per seed, seeds {doc['seeds']}. Budgets are L-inf in robust-scaled units; "
        "every candidate is projected onto realistic flows in raw units.",
        "",
        "## 🎯 Clean accuracy (full TEST split)",
        "",
        "| model | accuracy | macro-F1 |",
        "|---|---|---|",
        *[
            f"| {k} | {v['accuracy']:.4f} | {v['macro_f1']:.4f} |"
            for k, v in doc["clean"].items()
        ],
        "",
        "## ⚔️ Attack success (flows flipped to BENIGN), mean ± std over seeds",
        "",
        "| attack | eps | ensemble flipped | below alert threshold | MLP member flipped | energy flags successes | median raw change* | queries/flow |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for eps in eps_list:
        for fam in FAMILIES:
            r = s[f"{fam}@{eps:g}"]
            en = r["energy_flags_successes_mean"]
            rc = r.get("median_raw_relative_change")
            lines.append(
                f"| {fam} | {eps:g} | {r['success_rate']['mean']:.1%} ± {r['success_rate']['std']:.1%} "
                f"| {r['below_alert_threshold']['mean']:.1%} | {r['mlp_member_success']['mean']:.1%} "
                f"| {'n/a' if en is None else f'{en:.0%}'} | {'n/a' if rc is None else f'{rc:.0%}'} "
                f"| {r.get('queries_mean', 0):.0f} |"
            )
    lines += [
        "",
        "*median over changed features of |raw change| / (|original raw value| + 1); "
        "100% is e.g. a count or microsecond value moving from 0 to 1, or a duration doubling.",
    ]
    lines += _per_class_md(doc)
    lines += [
        "",
        f"Split check: robust-scaler means refit on the rebuilt TRAIN split match the "
        f"deployed scaler for {doc['scaler_means_reproduced']} features "
        f"(differing: {doc['scaler_means_differing'] or 'none'}).",
    ]
    lines += ["", "## 🧾 Verdict", "", *[f"- {v}" for v in doc["verdict"]], ""]
    lines += ["## ⚠️ Caveats", "", *[f"- {c}" for c in doc["caveats"]], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(summary: dict[str, Any], eps_list: tuple[float, ...], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4))
    colors = {"random": "#8a8a8a", "pgd": "#2a6fdb", "query": "#d9480f"}
    labels = {
        "random": "random control",
        "pgd": "PGD on MLP (transfer)",
        "query": "black-box query",
    }
    for fam in FAMILIES:
        m = [summary[f"{fam}@{e:g}"]["success_rate"]["mean"] * 100 for e in eps_list]
        sd = [summary[f"{fam}@{e:g}"]["success_rate"]["std"] * 100 for e in eps_list]
        ax.errorbar(
            eps_list,
            m,
            yerr=sd,
            marker="o",
            capsize=3,
            color=colors[fam],
            label=labels[fam],
        )
    ax.set_xscale("log")
    ax.set_xlabel("budget eps (L-inf, robust-scaled units)")
    ax.set_ylabel("detected attacks flipped to BENIGN (%)")
    ax.set_title("Deployed ensemble under constrained evasion")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- main -----------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Evasion test of the deployed detector")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eps", type=float, nargs="+", default=list(EPSILONS))
    ap.add_argument("--query-steps", type=int, default=8)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    configure_logging()
    det = load_detector(args.root)
    dataset = json.loads(
        (args.root / "artifacts" / "deploy" / "columns.json").read_text("utf-8")
    )["dataset"]
    cache = args.cache or args.root / "data" / "processed" / "robust_deployed_split.npz"
    sp = cached_split(args.root, dataset, cache)
    if list(sp["feature_names"]) != list(det.feature_names):
        raise SystemExit("rebuilt split's features differ from the deployed detector")
    same = np.isclose(sp["train_mean_check"], det.scaler.mean_, rtol=1e-6, atol=1e-9)
    Xs = det.scaler.transform(sp["X_test_raw"]).astype(np.float32)
    y = np.asarray(sp["y_test"])
    cons = build_constraints(det, sp)
    eps_list = tuple(args.eps)
    runs = [
        run_seed(det, Xs, y, cons, s, args.n, eps_list, args.query_steps)
        for s in args.seeds
    ]
    summary = summarize(runs)
    doc = {
        "detector": str(args.root / "artifacts" / "deploy"),
        "dataset": dataset,
        "members": sorted(det.ensemble.weights)
        if det.ensemble is not None
        else ["mlp"],
        "split": "test (held out; deploy.py split, seed 0)",
        # split check: refit robust scaler on the rebuilt TRAIN split vs deployed one
        "scaler_means_reproduced": f"{int(same.sum())}/{len(same)}",
        "scaler_means_differing": [
            n for n, s in zip(sp["feature_names"], same, strict=True) if not s
        ],
        "n_test": len(y),
        "n_flows": runs[0]["n_flows"],
        "seeds": args.seeds,
        "epsilons": list(eps_list),
        "clean": clean_metrics(det, Xs, y),
        "summary": summary,
        "verdict": verdict(summary, eps_list),
        "mutable_features": int(cons.mask.sum()),
        "frozen_derived_without_formula": list(cons.unknown_derived),
        "caveats": [
            "Feature-space attack: a flipped flow is a feature vector that passes the "
            "realism projection, not a replayed packet capture (upper bound on evadability).",
            "Success counts only flows the clean ensemble flagged (argmax != BENIGN).",
            "PGD has white-box access to the MLP member only; its effect on the "
            "ensemble is transfer.",
            "The query attack uses the ensemble's probabilities (score access), "
            "which a real attacker rarely has; label-only access would be weaker.",
            "eps is in robust-scaled units (log1p for heavy-tailed features), so one "
            "eps means different raw changes for different features.",
        ],
        "runs": runs,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "robust_deployed.json").write_text(json.dumps(doc, indent=2), "utf-8")
    write_md(doc, args.out / "robust_deployed.md")
    plot(summary, eps_list, args.out / "robust_deployed.png")
    print(f"[robust_deployed] -> {args.out / 'robust_deployed.md'}")


if __name__ == "__main__":
    main()
