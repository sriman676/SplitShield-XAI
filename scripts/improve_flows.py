"""Improve the deployed flow detectors (owner 2026-10-15) - champion vs challengers.

CICFlowMeter detector (artifacts/deploy):
  * challenger "multi"     - the deployed MLP fine-tuned and LightGBM refitted on
    CIC-IDS2017 + CIC-DDoS2019 (day 03-11: reflection / UDP / SYN floods the 2017
    capture lacks), every CIC-DDoS2019 attack mapped to the DDoS family;
  * challenger "multi+adv" - the same with PGD adversarial training of the MLP
    member (attack rows perturbed toward the wrong class inside the realistic
    constraint set, as in the research H3 arm).
  The deployed robust scaler and feature set are kept (live inputs and the probe
  stay valid). Temperature, alert thresholds and the D3 energy threshold are
  re-fitted on the combined VALIDATION split.

NetFlow detector (artifacts/deploy_netflow):
  * challenger "multi"     - tree members refitted and the MLP fine-tuned on
    NF-CSE-CIC-IDS2018-v2 + NF-ToN-IoT-v2 + NF-UNSW-NB15-v2 (binary), alert
    threshold re-fitted on the combined validation split.

Owner rules: selection on VALIDATION, TEST reported; a challenger replaces the
champion only if, on every dataset the champion was trained on, test precision AND
attack recall are not lower and the false-alarm rate is not higher, and it is not
more than 10 % slower. BLIND sets are never read (data/blind/*). The current
artifacts are backed up before anything is replaced.

    python -m scripts.improve_flows cic      [--epochs 6]
    python -m scripts.improve_flows netflow  [--epochs 4]
    python -m scripts.improve_flows cascade  (champion OR DDoS specialist; candidate only)
Writes results/extra/improve_flows.{json,md}.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

from utils.logging_config import configure_logging, get_logger
from utils.paths import PROJECT_ROOT

_log = get_logger(__name__)
CACHE = PROJECT_ROOT / "data" / "processed" / "improve_flows"
OUT = PROJECT_ROOT / "results" / "extra"
DEPLOY = PROJECT_ROOT / "artifacts" / "deploy"
NETFLOW = PROJECT_ROOT / "artifacts" / "deploy_netflow"
SEED = 0
DDOS_PER_CLASS = 40_000  # CIC-DDoS2019 rows kept per native class (before the split)
NF_PER_CLASS = 60_000  # NetFlow rows per binary class per extra dataset
SPEED_TOL = 0.10


def _free_gb() -> float:
    """Available physical memory in GiB (Windows, Linux, macOS; utils/sysinfo.py)."""
    from utils.sysinfo import ram_gb

    return ram_gb()[1]


def _ram_ok(need_gb: float = 6.0) -> None:
    free = _free_gb()
    if free < need_gb:
        raise SystemExit(f"only {free:.1f} GB RAM free (need {need_gb}); try later")


# --- data -----------------------------------------------------------------------------


def _split(
    n: int, y: np.ndarray, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 70 / 15 / 15 indices."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(n)
    tr, rest = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)
    va, te = train_test_split(rest, test_size=0.50, random_state=seed, stratify=y[rest])
    return tr, va, te


def _clean(df: pd.DataFrame, names: list[str], medians: list[float]) -> np.ndarray:
    """Same cleaning the live SOC applies (engine.flow_logits): numeric, Inf -> NaN,
    NaN / missing -> the training median."""
    frame = (df.reindex(columns=names).apply(pd.to_numeric, errors="coerce")
             .replace([np.inf, -np.inf], np.nan))  # fmt: skip
    return frame.fillna(dict(zip(names, medians, strict=True))).to_numpy(np.float32)


def cic17(det: Any) -> dict[str, np.ndarray]:
    """CIC-IDS2017 original, split exactly as deploy.py did (raw units, cached)."""
    path = CACHE / "cic17.npz"
    if path.is_file():
        return dict(np.load(path))
    import src.data.sources as sources
    from config.base import build_config, experiment_for_dataset
    from src.data.preprocess import preprocess_frame

    _ram_ok()
    ds = "cicids2017-original"
    cfg = build_config(experiment_for_dataset(ds, False), seed=SEED)
    spec = sources.get_spec(ds)
    data = preprocess_frame(sources.load_raw(spec), spec, cfg, seed=SEED)
    if list(data.label_names) != list(det.label_names) or list(
        data.feature_names
    ) != list(det.feature_names):
        raise SystemExit(
            "rebuilt CIC-IDS2017 split does not match the deployed detector"
        )
    raw = data.frame[list(data.feature_names)].to_numpy(np.float32)
    out = {f"X_{s}": raw[getattr(data, f"idx_{s}")] for s in ("train", "val", "test")}
    out |= {
        f"y_{s}": np.asarray(getattr(data, f"y_{s}")) for s in ("train", "val", "test")
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def ddos19(det: Any) -> dict[str, np.ndarray]:
    """CIC-DDoS2019 day 03-11 (not a blind set), aligned to the deployed features;
    BENIGN stays BENIGN, every attack becomes the DDoS family."""
    path = CACHE / "ddos19.npz"
    if path.is_file():
        return dict(np.load(path))
    from src.soc.columns import align

    _ram_ok(3.0)
    # memory-light: stream every 03-11 CSV and keep, per label, the rows with the
    # DDOS_PER_CLASS smallest seeded random keys (a uniform sample of each label)
    feats = list(det.feature_names)
    rng = np.random.default_rng(SEED)
    keep: dict[str, pd.DataFrame] = {}
    for f in sorted(
        (PROJECT_ROOT / "data" / "raw" / "cic-ddos2019" / "03-11").glob("*.csv")
    ):
        for chunk in pd.read_csv(f, chunksize=200_000, low_memory=False):
            chunk, _info = align(chunk, feats)
            chunk = chunk[[*[c for c in feats if c in chunk.columns], "Label"]]
            chunk = chunk.assign(
                _k=rng.random(len(chunk)), Label=chunk["Label"].astype(str).str.strip()
            )
            for lab, g in chunk.groupby("Label"):
                both = pd.concat([keep[lab], g]) if lab in keep else g
                keep[lab] = both.nsmallest(DDOS_PER_CLASS, "_k")
        _log.info(
            "CIC-DDoS2019 %s read; labels so far %s",
            f.name,
            {k: len(v) for k, v in keep.items()},
        )
    df = pd.concat(keep.values(), ignore_index=True).drop(columns="_k")
    # identical flows must not sit in both train and test (CIC-IDS2017 is de-duplicated too)
    df = df.drop_duplicates(subset=[c for c in feats if c in df.columns] + ["Label"])
    native = df["Label"].astype(str).str.strip()
    names = list(det.label_names)
    y = np.where(
        native.str.upper() == "BENIGN", names.index("BENIGN"), names.index("DDoS")
    )
    X = _clean(df, list(det.feature_names), det.input_stats["median"])
    tr, va, te = _split(len(y), native.to_numpy())
    out = {"X_train": X[tr], "y_train": y[tr], "X_val": X[va], "y_val": y[va],
           "X_test": X[te], "y_test": y[te], "native_test": native.to_numpy()[te].astype(str)}  # fmt: skip
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


# --- training ---------------------------------------------------------------------------


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    from src.models.ensemble import capped_weights

    w = capped_weights(y)
    cw = np.ones(n_classes)
    for c in np.unique(y):
        cw[c] = w[y == c].mean()
    return torch.as_tensor(cw / cw.mean(), dtype=torch.float32)


def _macro_f1(model: torch.nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(
        f1_score(y, _proba(model, X).argmax(1), average="macro", zero_division=0)
    )


def _proba(model: Any, X: np.ndarray, batch: int = 50_000) -> np.ndarray:
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(
                torch.softmax(
                    model(torch.as_tensor(X[i : i + batch], dtype=torch.float32)), 1
                ).numpy()
            )
    return np.concatenate(out)


def _pgd_rows(mlp: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
              inc: torch.Tensor, eps: float, steps: int = 5) -> torch.Tensor:  # fmt: skip
    """Adversarial versions of ``x`` (maximise the loss of the TRUE class) inside the
    L-inf ball, attacker-controllable features only, increase-only features never
    lowered (the realistic constraint set in scaled space)."""
    x0 = x.detach()
    xa = x0.clone()
    alpha = 2.5 * eps / steps
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(mlp(xa), y)
        (g,) = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign() * mask
        xa = torch.max(torch.min(xa, x0 + eps), x0 - eps)
        xa = torch.where(inc, torch.max(xa, x0), xa)
    return xa.detach()


def finetune(mlp0: torch.nn.Module, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray,
             yva: np.ndarray, epochs: int, adv: dict[str, Any] | None = None,
             lr: float = 3e-4, batch: int = 1024) -> tuple[torch.nn.Module, list[float]]:  # fmt: skip
    """Fine-tune a copy of the champion MLP; the best VALIDATION macro-F1 epoch wins."""
    torch.manual_seed(SEED)
    mlp = copy.deepcopy(mlp0)
    n_classes = int(mlp.head.out_features)  # type: ignore[union-attr, arg-type]
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss(weight=_class_weights(ytr, n_classes))
    X, Y = (
        torch.as_tensor(Xtr, dtype=torch.float32),
        torch.as_tensor(ytr, dtype=torch.long),
    )
    best, best_f1, hist, bad = (
        copy.deepcopy(mlp.state_dict()),
        _macro_f1(mlp, Xva, yva),
        [],
        0,
    )
    hist.append(best_f1)
    rng = np.random.default_rng(SEED)
    for ep in range(epochs):
        mlp.train()
        order = torch.as_tensor(rng.permutation(len(X)))
        for i in range(0, len(X), batch):
            b = order[i : i + batch]
            xb, yb = X[b], Y[b]
            if adv is not None:
                att = yb != adv["benign"]
                if att.any():
                    mlp.eval()
                    xa = _pgd_rows(
                        mlp, xb[att], yb[att], adv["mask"], adv["inc"], adv["eps"]
                    )
                    mlp.train()
                    xb, yb = torch.cat([xb, xa]), torch.cat([yb, yb[att]])
            opt.zero_grad()
            lossf(mlp(xb), yb).backward()
            opt.step()
        f1 = _macro_f1(mlp, Xva, yva)
        hist.append(f1)
        _log.info("fine-tune epoch %d: val macro-F1 %.4f", ep + 1, f1)
        if f1 > best_f1:
            best, best_f1, bad = copy.deepcopy(mlp.state_dict()), f1, 0
        else:
            bad += 1
            if bad >= 2:
                break
    mlp.load_state_dict(best)
    return mlp.eval(), hist


# --- evaluation -------------------------------------------------------------------------


def det_metrics(
    scorer: Any, Xs: np.ndarray, y: np.ndarray, benign: int
) -> dict[str, Any]:
    """Attack recall / false alarms / precision (argmax != BENIGN) + macro-F1."""
    from sklearn.metrics import f1_score

    pred = _proba(scorer, Xs).argmax(1)
    att, alert = y != benign, pred != benign
    tp, fn = int((att & alert).sum()), int((att & ~alert).sum())
    fp, tn = int((~att & alert).sum()), int((~att & ~alert).sum())
    return {"attacks": tp + fn, "caught": tp, "benign": fp + tn, "false_alarms": fp,
            "recall": tp / max(1, tp + fn), "false_alarm_rate": fp / max(1, fp + tn),
            "precision": tp / max(1, tp + fp),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}  # fmt: skip


def _metrics_from_pred(pred: np.ndarray, y: np.ndarray, benign: int) -> dict[str, Any]:
    att, alert = y != benign, pred != benign
    tp, fn = int((att & alert).sum()), int((att & ~alert).sum())
    fp, tn = int((~att & alert).sum()), int((~att & ~alert).sum())
    return {"recall": tp / max(1, tp + fn), "false_alarm_rate": fp / max(1, fp + tn),
            "precision": tp / max(1, tp + fp)}  # fmt: skip


def _f1_macro(pred: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y, pred, average="macro", zero_division=0))


def not_worse(new: dict[str, Any], old: dict[str, Any]) -> bool:
    """Owner rule: precision AND recall not lower, false alarms not higher."""
    return (new["precision"] >= old["precision"] - 1e-12 and new["recall"] >= old["recall"] - 1e-12
            and new["false_alarm_rate"] <= old["false_alarm_rate"] + 1e-12)  # fmt: skip


def speed(scorer: Any, Xs: np.ndarray, repeats: int = 3) -> float:
    x = Xs[:20000]
    _proba(scorer, x)
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _proba(scorer, x)
        t.append(time.perf_counter() - t0)
    return float(len(x) / np.median(t))


def robustness(scorer: Any, mlp: Any, energy_thr: float | None, Xs: np.ndarray, y: np.ndarray,
               benign: int, cons: Any, thr: float, n: int = 2000) -> dict[str, Any]:  # fmt: skip
    """Flipped-attack rate: PGD on the MLP member transferred to the ensemble vs the
    same-size random control (scripts/robust_deployed.py attacks, seed 0)."""
    from scripts import robust_deployed as rd

    pred = _proba(scorer, Xs).argmax(1)
    pool = np.flatnonzero((y != benign) & (pred != benign))
    rng = np.random.default_rng(SEED)
    pick = rng.choice(pool, size=min(n, len(pool)), replace=False)
    x0 = Xs[pick].astype(np.float64)
    det = SimpleNamespace(ensemble=scorer, model=mlp, energy_threshold=energy_thr)
    out = {}
    for eps in (0.1, 0.3):
        for fam in ("random", "pgd"):
            xa = (rd.attack_random(x0, eps, cons, rng) if fam == "random"
                  else rd.attack_pgd(mlp, x0, eps, cons, benign))  # fmt: skip
            o = rd.outcome(det, x0, xa, benign, thr, cons)
            out[f"{fam}@{eps:g}"] = round(float(o["success_rate"]), 4)
    return out


# --- CICFlowMeter -----------------------------------------------------------------------


def run_cic(epochs: int, apply: bool = True) -> dict[str, Any]:
    from deploy import (
        _alert_threshold,
        _class_thresholds,
        _fit_temperature,
        conformal_qhat,
    )
    from scripts import robust_deployed as rd
    from src.data.robust import input_stats
    from src.defense.detector import fit_energy_detector
    from src.models.ensemble import EnsembleModel, capped_weights
    from src.soc.model import Detector

    det = Detector(DEPLOY)
    names = list(det.label_names)
    benign = names.index("BENIGN")
    c17, dd = cic17(det), ddos19(det)
    sc = det.scaler

    def S(x: np.ndarray) -> np.ndarray:  # noqa: N802 - model space
        return np.asarray(sc.transform(x), dtype=np.float32)

    sets = {"cic-ids2017": {s: (S(c17[f"X_{s}"]), c17[f"y_{s}"]) for s in ("train", "val", "test")},
            "cic-ddos2019": {s: (S(dd[f"X_{s}"]), dd[f"y_{s}"]) for s in ("train", "val", "test")}}  # fmt: skip
    Xtr = np.vstack([sets[d]["train"][0] for d in sets])
    ytr = np.concatenate([sets[d]["train"][1] for d in sets])
    Xva = np.vstack([sets[d]["val"][0] for d in sets])
    yva = np.concatenate([sets[d]["val"][1] for d in sets])
    champion = det.ensemble or det.model
    sp = {"feature_names": list(det.feature_names), "train_min": c17["X_train"].min(0),
          "train_max": c17["X_train"].max(0)}  # fmt: skip
    cons = rd.build_constraints(det, sp)
    adv = {"benign": benign, "eps": 0.1,
           "mask": torch.as_tensor(cons.mask.astype(np.float32)),
           "inc": torch.as_tensor(rd._increase_only(cons))}  # fmt: skip

    # shared tree member: LightGBM refitted on the combined training data
    _ram_ok(3.0)
    import lightgbm
    from lightgbm import LGBMClassifier

    t0 = time.time()
    gbm = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                         reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)  # fmt: skip
    # early stopping on VALIDATION: no more trees than help (also keeps scoring fast)
    gbm.fit(Xtr, ytr, sample_weight=capped_weights(ytr), eval_set=[(Xva, yva)],
            callbacks=[lightgbm.early_stopping(50, verbose=False)])  # fmt: skip
    _log.info("LightGBM refit: %.0f s, %s trees", time.time() - t0, gbm.best_iteration_)
    weights: dict[str, float] = (
        dict(det.ensemble.weights) if det.ensemble is not None else {"mlp": 1.0}  # type: ignore[arg-type]
    )

    cands: dict[str, Any] = {"champion": {"scorer": champion, "mlp": det.model}}
    champ_val = det_metrics(champion, *sets["cic-ids2017"]["val"], benign)
    for name, a in (("multi", None), ("multi+adv", adv)):
        mlp, hist = finetune(det.model, Xtr, ytr, Xva, yva, epochs, a)
        trees = {"lgbm": gbm} if "lgbm" in weights else {}
        # class prior shift chosen on VALIDATION (log-probabilities + bias, so it is
        # searched on stored scores): favour BENIGN and/or disfavour DDoS until the
        # CIC-IDS2017 validation precision, recall and false alarms are all at least
        # as good as the champion's; among those, the best combined validation macro-F1
        base = EnsembleModel(mlp, trees, weights).eval()
        lp17 = np.log(_proba(base, sets["cic-ids2017"]["val"][0]).clip(1e-7))
        lpall = np.log(_proba(base, Xva).clip(1e-7))
        y17 = sets["cic-ids2017"]["val"][1]
        ddos = names.index("DDoS")
        best: tuple[float, np.ndarray] | None = None
        for bb in np.arange(0.0, 3.01, 0.25):
            for bd in np.arange(0.0, -4.01, -0.25):
                vec = np.zeros(len(names), dtype=np.float32)
                vec[benign], vec[ddos] = bb, bd
                m17 = _metrics_from_pred((lp17 + vec).argmax(1), y17, benign)
                if not not_worse(m17, champ_val):
                    continue
                f1 = _f1_macro((lpall + vec).argmax(1), yva)
                if best is None or f1 > best[0]:
                    best = (f1, vec)
        if best is not None:
            bias_vec = best[1]
            scorer = EnsembleModel(mlp, trees, weights, bias_vec).eval()
            bias = float(bias_vec[benign]), float(bias_vec[ddos])
        else:
            scorer, bias = base, None
        cands[name] = {"scorer": scorer, "mlp": mlp, "trees": trees, "val_history": hist,
                       "benign_log_bias": bias}  # fmt: skip

    rows: dict[str, Any] = {}
    for name, c in cands.items():
        thr_e = (det.energy_threshold if name == "champion"
                 else fit_energy_detector(c["mlp"], Xva, "cpu", target_fpr=0.01).threshold)  # fmt: skip
        c["energy"] = thr_e
        r: dict[str, Any] = {"val": {d: det_metrics(c["scorer"], *sets[d]["val"], benign) for d in sets},
                             "test": {d: det_metrics(c["scorer"], *sets[d]["test"], benign) for d in sets}}  # fmt: skip
        r["val_macro_f1_combined"] = _macro_f1(c["scorer"], Xva, yva)
        r["benign_log_bias"] = c.get("benign_log_bias", 0.0)
        r["val_history"] = c.get("val_history")
        r["flows_per_s"] = speed(c["scorer"], Xva)
        r["robustness"] = robustness(c["scorer"], c["mlp"], thr_e, *sets["cic-ids2017"]["test"], benign,
                                     cons, float(det.alert_threshold_high or 0.5))  # fmt: skip
        rows[name] = r
        _log.info("%s: %s", name, json.dumps(r["test"])[:400])

    champ = rows["champion"]
    # choose on VALIDATION: challengers whose validation passes the owner rule on the
    # champion's own dataset, best combined validation macro-F1; then TEST must pass
    ok_val = [n for n in cands if n != "champion"
              and not_worse(rows[n]["val"]["cic-ids2017"], champ["val"]["cic-ids2017"])]  # fmt: skip
    chosen = (
        max(ok_val, key=lambda n: rows[n]["val_macro_f1_combined"]) if ok_val else None
    )
    decision = "kept champion: no challenger passed validation on CIC-IDS2017"
    if chosen:
        t_new, t_old = rows[chosen]["test"]["cic-ids2017"], champ["test"]["cic-ids2017"]
        if not not_worse(t_new, t_old):
            decision = f"kept champion: {chosen} was worse on CIC-IDS2017 test"
        elif rows[chosen]["flows_per_s"] < champ["flows_per_s"] * (1 - SPEED_TOL):
            decision = f"kept champion: {chosen} more than 10 % slower"
        else:
            decision = f"promoted {chosen}"
    doc = {"what": "CICFlowMeter detector: champion vs multi-dataset / adversarial challengers",
           "datasets": {"cic-ids2017": "deploy.py split (seed 0)",
                        "cic-ddos2019": f"day 03-11, <= {DDOS_PER_CLASS} rows per native class, 70/15/15 stratified"},
           "blind_used": False, "rows": rows, "chosen_on_validation": chosen, "decision": decision}  # fmt: skip
    if apply and decision.startswith("promoted") and chosen:
        c = cands[chosen]
        backup = (
            PROJECT_ROOT
            / "artifacts"
            / "deploy_backup"
            / f"{time.strftime('%Y%m%d-%H%M')}_pre_improve_flows"
        )
        shutil.copytree(DEPLOY, backup / "deploy")
        data_va = SimpleNamespace(X_val=Xva, y_val=yva, label_names=names)
        T = _fit_temperature(c["scorer"], Xva, yva, "cpu")
        cols = json.loads((DEPLOY / "columns.json").read_text("utf-8"))
        cols.update(
            temperature=T,
            alert_threshold_high=_alert_threshold(c["scorer"], data_va, T, "cpu", target_fpr=0.01),
            alert_threshold_by_class=_class_thresholds(c["scorer"], data_va, T, "cpu", target_fpr=0.01),
            conformal_qhat=conformal_qhat(c["scorer"], Xva, yva, T, "cpu"),
            energy_threshold=c["energy"],
            class_log_bias=(c["scorer"].log_bias.tolist() if getattr(c["scorer"], "log_bias", None) is not None else None),
            input_stats=input_stats(sc, np.vstack([c17["X_train"], dd["X_train"]])),
            dataset="cicids2017-original",
            trained_on=["cicids2017-original", "cic-ddos2019 (03-11)"],
            improved=f"{time.strftime('%Y-%m-%d')} scripts/improve_flows.py: {chosen}",
        )  # fmt: skip
        torch.save(c["mlp"].state_dict(), DEPLOY / "model.pt")
        if c["trees"]:
            import joblib

            joblib.dump(c["trees"], DEPLOY / "ensemble.pkl", compress=3)
        (DEPLOY / "columns.json").write_text(json.dumps(cols, indent=2), "utf-8")
        doc["backup"] = str(backup)
    return doc


# --- NetFlow ----------------------------------------------------------------------------


def nf_extra(det: Any, name: str, rel: str) -> dict[str, np.ndarray]:
    """One extra NetFlow v2 dataset, binary, aligned to the deployed features."""
    path = CACHE / f"{name}.npz"
    if path.is_file():
        return dict(np.load(path))
    from scripts.deploy_netflow import sample_netflow

    _ram_ok(4.0)
    df = sample_netflow(PROJECT_ROOT / "data" / "raw" / rel, NF_PER_CLASS, seed=SEED)
    X, _missing = det.align(df)
    y = (df["Label"].astype(int).to_numpy() != 0).astype(int)
    tr, va, te = _split(len(y), y)
    out = {
        f"X_{s}": X[i].astype(np.float64)
        for s, i in (("train", tr), ("val", va), ("test", te))
    }
    out |= {f"y_{s}": y[i] for s, i in (("train", tr), ("val", va), ("test", te))}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def run_netflow(epochs: int, apply: bool = True) -> dict[str, Any]:
    from scripts.deploy_netflow import TRANSFER, alert_threshold, fit_trees
    from src.models.ensemble import EnsembleModel
    from src.soc import netflow_model as nm

    det = nm.load_netflow()
    if det is None:
        raise SystemExit("NetFlow detector not deployed")
    benign = det.label_names.index("BENIGN")
    hold, rep = np.load(NETFLOW / "holdout.npz"), np.load(NETFLOW / "replay.npz")

    def S(x: np.ndarray) -> np.ndarray:  # noqa: N802
        x = np.where(np.isfinite(x), x, det.medians)
        return np.asarray(det.scaler.transform(x), dtype=np.float32)

    sets = {"nf-cse-cic-ids2018-v2": {"train": (S(rep["X"]), rep["y"].astype(int)),
                                      "val": (S(hold["X_val"]), hold["y_val"].astype(int)),
                                      "test": (S(hold["X_test"]), hold["y_test"].astype(int))}}  # fmt: skip
    for name, rel in TRANSFER.items():
        d = nf_extra(det, name, rel)
        sets[name] = {
            s: (S(d[f"X_{s}"]), d[f"y_{s}"].astype(int))
            for s in ("train", "val", "test")
        }
    Xtr = np.vstack([v["train"][0] for v in sets.values()])
    ytr = np.concatenate([v["train"][1] for v in sets.values()])
    Xva = np.vstack([v["val"][0] for v in sets.values()])
    yva = np.concatenate([v["val"][1] for v in sets.values()])
    weights = det.meta.get("ensemble") or {"mlp": 1.0}
    champion = det.scorer

    def with_threshold(scorer: Any, thr: float, T: float) -> Any:
        class Thr(
            torch.nn.Module
        ):  # alert at the deployed threshold, as live scoring does
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                p = torch.softmax(scorer(x) / T, 1)
                att = (1 - p[:, benign]) >= thr
                out = torch.zeros_like(p)
                out[att, 1 - benign], out[~att, benign] = 1.0, 1.0
                return torch.log(out.clamp_min(1e-7))

        return Thr().eval()

    trees = fit_trees(SimpleNamespace(X_train=Xtr, y_train=ytr), SEED)
    mlp, hist = finetune(det.model, Xtr, ytr, Xva, yva, epochs)
    chal = EnsembleModel(
        mlp, {k: v for k, v in trees.items() if k in weights}, weights
    ).eval()
    pva = _proba(chal, Xva)
    thr_new = alert_threshold((1 - pva[:, benign])[yva == benign])
    scorers = {"champion": with_threshold(champion, det.alert_threshold, det.temperature),
               "multi": with_threshold(chal, thr_new, 1.0)}  # fmt: skip
    rows: dict[str, dict[str, Any]] = {n: {"val": {d: det_metrics(s, *v["val"], benign) for d, v in sets.items()},
                "test": {d: det_metrics(s, *v["test"], benign) for d, v in sets.items()},
                "flows_per_s": speed(s, Xva)} for n, s in scorers.items()}  # fmt: skip
    rows["multi"]["val_history"] = hist
    rows["multi"]["alert_threshold"] = thr_new
    home = "nf-cse-cic-ids2018-v2"
    decision = "promoted multi"
    if not not_worse(rows["multi"]["val"][home], rows["champion"]["val"][home]):
        decision = "kept champion: worse on NF-CSE validation"
    elif not not_worse(rows["multi"]["test"][home], rows["champion"]["test"][home]):
        decision = "kept champion: worse on NF-CSE test"
    elif rows["multi"]["flows_per_s"] < rows["champion"]["flows_per_s"] * (
        1 - SPEED_TOL
    ):
        decision = "kept champion: more than 10 % slower"
    doc = {"what": "NetFlow detector: champion vs multi-dataset challenger", "rows": rows,
           "decision": decision, "blind_used": False}  # fmt: skip
    if apply and decision.startswith("promoted"):
        backup = (
            PROJECT_ROOT
            / "artifacts"
            / f"deploy_netflow_backup_{time.strftime('%Y%m%d-%H%M')}"
        )
        shutil.copytree(NETFLOW, backup)
        meta = dict(det.meta) | {"alert_threshold": thr_new, "temperature": 1.0,
                                 "trained_on": [home, *TRANSFER],
                                 "improved": f"{time.strftime('%Y-%m-%d')} scripts/improve_flows.py"}  # fmt: skip
        nm.save_netflow(NETFLOW, mlp, trees, det.scaler, meta)
        nm._CACHE.clear()
        doc["backup"] = str(backup)
    return doc


# --- report -----------------------------------------------------------------------------


# --- CICFlowMeter: DDoS specialist cascade (owner 2026-10-16) -----------------------

CANDIDATES = PROJECT_ROOT / "artifacts" / "deploy_candidates" / "flow_ddos_specialist"


def run_cascade() -> dict[str, Any]:
    """Cross-dataset fix that cannot cost CIC-IDS2017 precision on validation.

    The deployed detector catches almost no CIC-DDoS2019 attacks (recall 0.0003), and
    retraining it on both datasets lost a little CIC-IDS2017 precision (2026-10-15).
    Here the champion is kept unchanged and a small binary LightGBM "specialist"
    (attack vs benign, trained on CIC-IDS2017 + CIC-DDoS2019 training rows) only ADDS
    alerts on flows the champion calls benign: alert = champion OR specialist >= t.
    t is chosen on VALIDATION as the lowest score above every CIC-IDS2017 validation
    benign flow the champion lets through (no extra false alarm on its home dataset),
    so recall can only rise there; TEST then decides (owner rule: precision AND
    recall not lower, false alarms not higher on CIC-IDS2017; not >10 % slower).
    The specialist is written to artifacts/deploy_candidates/ only - wiring it into
    the live detector needs a small change in src/soc/model.py (owner/lead decision).
    """
    import joblib
    import lightgbm
    from lightgbm import LGBMClassifier

    from src.soc.model import Detector

    det = Detector(DEPLOY)
    names = list(det.label_names)
    benign, ddos = names.index("BENIGN"), names.index("DDoS")
    c17, dd = cic17(det), ddos19(det)
    sc = det.scaler

    def S(x: np.ndarray) -> np.ndarray:  # noqa: N802 - model space
        return np.asarray(sc.transform(x), dtype=np.float32)

    sets = {"cic-ids2017": {s: (S(c17[f"X_{s}"]), c17[f"y_{s}"]) for s in ("train", "val", "test")},
            "cic-ddos2019": {s: (S(dd[f"X_{s}"]), dd[f"y_{s}"]) for s in ("train", "val", "test")}}  # fmt: skip
    champion = det.ensemble or det.model
    Xtr = np.vstack([sets[d]["train"][0] for d in sets])
    ytr = np.concatenate([(sets[d]["train"][1] != benign).astype(int) for d in sets])
    Xva = np.vstack([sets[d]["val"][0] for d in sets])
    yva = np.concatenate([(sets[d]["val"][1] != benign).astype(int) for d in sets])
    _ram_ok(3.0)
    t0 = time.time()
    spec = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                          reg_lambda=1.0, random_state=SEED, n_jobs=6, verbose=-1)  # fmt: skip
    spec.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lightgbm.early_stopping(30, verbose=False)])
    _log.info("specialist: %.0f s, %s trees", time.time() - t0, spec.best_iteration_)

    pred_c = {d: {s: _proba(champion, sets[d][s][0]).argmax(1) for s in ("val", "test")} for d in sets}
    score = {d: {s: spec.predict_proba(sets[d][s][0])[:, 1] for s in ("val", "test")} for d in sets}
    # threshold on VALIDATION: above every CIC-IDS2017 benign flow the champion passes
    y17v = sets["cic-ids2017"]["val"][1]
    passed = (y17v == benign) & (pred_c["cic-ids2017"]["val"] == benign)
    top = float(score["cic-ids2017"]["val"][passed].max()) if passed.any() else 0.5
    thr = float(min(max(float(np.nextafter(top, np.inf)), 0.5), 1.0))

    def cascade(d: str, s: str) -> np.ndarray:
        p = pred_c[d][s].copy()
        p[(p == benign) & (score[d][s] >= thr)] = ddos
        return p

    def champ_pred(d: str, s: str) -> np.ndarray:
        return pred_c[d][s]

    def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        return _metrics_from_pred(pred, y, benign) | {"macro_f1": _f1_macro(pred, y)}

    rows: dict[str, Any] = {}
    for name, fn in (("champion", champ_pred), ("ddos-specialist cascade", cascade)):
        rows[name] = {split: {d: metrics(fn(d, split), sets[d][split][1]) for d in sets}
                      for split in ("val", "test")}  # fmt: skip
    x = sets["cic-ids2017"]["test"][0][:20000]
    rows["champion"]["flows_per_s"] = speed(champion, x)

    class Both(torch.nn.Module):  # timing only: champion + specialist on the same rows
        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            spec.predict_proba(xb.numpy())
            return champion(xb)

    rows["ddos-specialist cascade"]["flows_per_s"] = speed(Both(), x)
    rows["ddos-specialist cascade"]["specialist_threshold"] = thr
    rows["ddos-specialist cascade"]["specialist_trees"] = int(spec.best_iteration_ or 300)
    new, old = rows["ddos-specialist cascade"], rows["champion"]
    passes = (not_worse(new["test"]["cic-ids2017"], old["test"]["cic-ids2017"])
              and new["test"]["cic-ddos2019"]["recall"] > old["test"]["cic-ddos2019"]["recall"])  # fmt: skip
    fast = new["flows_per_s"] >= old["flows_per_s"] * (1 - SPEED_TOL)
    if passes and fast:
        decision = ("passes the owner rule on test; NOT live yet - needs the src/soc/model.py "
                    "hook (alert if the specialist score >= threshold on champion-benign flows)")
    elif passes:
        decision = (f"passes precision/recall but is {1 - new['flows_per_s'] / old['flows_per_s']:.0%} "
                    "slower than the champion (rule: <= 10 %); kept champion")
    else:
        decision = "kept champion: CIC-IDS2017 test precision, recall or false alarms would get worse"
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": spec, "threshold": thr, "benign": benign, "ddos": ddos,
                 "feature_space": "deployed scaler output (artifacts/deploy/scaler.pkl)",
                 "trained": time.strftime("%Y-%m-%d %H:%M")}, CANDIDATES / "specialist.joblib", compress=3)  # fmt: skip
    return {"what": "CICFlowMeter detector: champion OR DDoS specialist (adds alerts only)",
            "datasets": {"cic-ids2017": "deploy.py split (seed 0)",
                         "cic-ddos2019": f"day 03-11, <= {DDOS_PER_CLASS} rows per native class, 70/15/15"},
            "blind_used": False, "rows": rows, "decision": decision,
            "candidate": str(CANDIDATES / "specialist.joblib")}  # fmt: skip


def _md_rows(doc: dict[str, Any]) -> list[str]:
    out = ["| Model | Dataset | Split | Precision | Attack recall | False alarms | Macro-F1 |",
           "|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, r in doc["rows"].items():
        for split in ("val", "test"):
            for ds, m in r[split].items():
                out.append(f"| {name} | {ds} | {split} | {m['precision']:.4f} | {m['recall']:.4f} | "
                           f"{m['false_alarm_rate']:.4%} | {m['macro_f1']:.4f} |")  # fmt: skip
    return out


def write(results: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "improve_flows.json"
    old = json.loads(p.read_text("utf-8")) if p.is_file() else {}
    old.update(results)
    p.write_text(json.dumps(old, indent=1, default=str), "utf-8")
    md = ["# Flow detector improvements (owner 2026-10-15)", "",
          "Selection on validation, test reported; a challenger replaces the deployed "
          "detector only if precision AND attack recall are not lower and false alarms "
          "not higher on the dataset it was trained on, and it is not >10 % slower. "
          "Blind sets were not used.", ""]  # fmt: skip
    for key, doc in old.items():
        md += [f"## {key}: {doc['decision']}", "", *_md_rows(doc), ""]
        speeds = {n: round(r["flows_per_s"]) for n, r in doc["rows"].items()}
        md.append(f"Speed (flows/s on 20k rows): {speeds}")
        rob = {n: r["robustness"] for n, r in doc["rows"].items() if "robustness" in r}
        if rob:
            md += ["", "Flipped detected attacks (CIC-IDS2017 test, 2,000 flows, seed 0):", "",
                   "| Model | " + " | ".join(next(iter(rob.values()))) + " |",
                   "|---|" + "---|" * len(next(iter(rob.values())))]  # fmt: skip
            md += [
                f"| {n} | " + " | ".join(f"{v:.1%}" for v in r.values()) + " |"
                for n, r in rob.items()
            ]
        if doc.get("backup"):
            md.append(f"\nPrevious artifacts backed up to `{doc['backup']}`.")
        md.append("")
    (OUT / "improve_flows.md").write_text("\n".join(md), "utf-8")


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("part", choices=("cic", "netflow", "cascade"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--no-apply", action="store_true")
    a = ap.parse_args()
    doc = (
        run_cic(a.epochs, not a.no_apply)
        if a.part == "cic"
        else run_cascade()
        if a.part == "cascade"
        else run_netflow(a.epochs, not a.no_apply)
    )
    write({a.part: doc})
    print(doc["decision"])


if __name__ == "__main__":
    main()
