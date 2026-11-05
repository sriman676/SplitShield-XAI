"""Flow detector across networks (owner 2026-10-24): a multi-dataset "generalist".

The deployed CICFlowMeter detector caught 99.99 % on its own network (CIC-IDS2017) but
56.1 % on the 2026-10-24 blind round (CIC-IDS2018): DDoS HOIC / LOIC-UDP 0 %, SQL
injection 2 %, Bot 32 %, Infiltration 31 %. The owner allowed the used blind sets as
normal data (2026-10-24), so this trains on four CICFlowMeter networks:

  * CIC-IDS2017 original  - the deployed split (scripts/improve_flows.cic17)
  * CIC-DDoS2019 day 03-11 - as for the DDoS specialist (improve_flows.ddos19)
  * CIC-IDS2018, 8 days   - <= PER_LABEL rows per label, 70 / 15 / 15 stratified
  * CIC IoT-DIAD 2024     - <= PER_LABEL rows per attack file group, 70 / 15 / 15

and compares, on every dataset's TEST split (selection on VALIDATION only):

  * champion   - the deployed detector (incl. the DDoS specialist), unchanged
  * cascade    - champion OR generalist: the generalist only ADDS alerts on flows the
                 champion calls benign (family = the generalist's top attack family);
                 threshold on validation = above every CIC-IDS2017 validation benign
                 flow the champion lets through (no extra false alarm at home)
  * cascade@1% - the same with the threshold at a 1 % false-alarm budget on the pooled
                 validation benign flows of the three new networks
  * generalist - the generalist alone (replaces the champion)

Owner rule: a variant replaces the champion only if on CIC-IDS2017 and CIC-DDoS2019
test precision AND attack recall are not lower and false alarms not higher; anything
else is reported for an owner decision. The new blind sets (D:/SplitShield-data/blind2)
are never read here.

    python -m scripts.improve_flows_v2 [--per-label 30000] [--cache DIR]
Writes results/extra/improve_flows_v2.{json,md}; the candidate goes to
artifacts/deploy_candidates/flow_generalist/ (nothing live is changed).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts import improve_flows as imf
from utils.logging_config import configure_logging, get_logger
from utils.paths import PROJECT_ROOT

_log = get_logger(__name__)
SEED = 0
PER_LABEL = 30_000
OUT = PROJECT_ROOT / "results" / "extra"
CANDIDATE = PROJECT_ROOT / "artifacts" / "deploy_candidates" / "flow_generalist"
C18 = PROJECT_ROOT / "data" / "blind" / "cic-ids2018"  # used blind set, owner-released 2026-10-24
DIAD = (PROJECT_ROOT / "data" / "raw" / "cic-iot-diad-2024"
        / "IoT device identification and anomaly detection dataset (CIC IoT-DIAD 2024)")  # fmt: skip

FAMILY_2018 = {"benign": "BENIGN", "bot": "Bot", "infilteration": "Infiltration", "infiltration": "Infiltration",
               "brute force -web": "WebAttack", "brute force -xss": "WebAttack", "sql injection": "WebAttack",
               "ftp-bruteforce": "BruteForce", "ssh-bruteforce": "BruteForce"}  # fmt: skip
FAMILY_DIAD = {"Benign": "BENIGN", "Brute Force": "BruteForce", "DDOS": "DDoS", "DOS": "DoS", "Mirai": "Bot",
               "Recon": "PortScan", "Web-Based": "WebAttack"}  # Spoofing (ARP / DNS) has no family: left out  # fmt: skip


def family_2018(label: str) -> str | None:
    s = label.strip().lower()
    if s in FAMILY_2018:
        return FAMILY_2018[s]
    if s.startswith("ddos"):
        return "DDoS"
    if s.startswith("dos"):
        return "DoS"
    return None


def _sample(frames: dict[str, pd.DataFrame], g: pd.DataFrame, key: str, n: int, rng: np.random.Generator) -> None:
    g = g.assign(_k=rng.random(len(g)))
    both = pd.concat([frames[key], g]) if key in frames else g
    frames[key] = both.nsmallest(n, "_k")


def _finish(frames: dict[str, pd.DataFrame], feats: list[str], det: Any, names: list[str]) -> dict[str, np.ndarray]:
    df = pd.concat(frames.values(), ignore_index=True).drop(columns="_k")
    df = df.drop_duplicates(subset=[c for c in feats if c in df.columns] + ["_native"])
    y = np.array([names.index(f) for f in df["_family"]])
    X = imf._clean(df, feats, det.input_stats["median"])
    native = df["_native"].astype(str).to_numpy().astype(str)  # unicode, not object: np.load needs no pickle
    tr, va, te = imf._split(len(y), native)
    return {"X_train": X[tr], "y_train": y[tr], "X_val": X[va], "y_val": y[va], "X_test": X[te],
            "y_test": y[te], "native_test": native[te]}  # fmt: skip


def cic18(det: Any, cache: Path, per_label: int) -> dict[str, np.ndarray]:
    path = cache / f"cic18_{per_label}.npz"
    if path.is_file():
        return dict(np.load(path))
    from src.soc.columns import align

    feats, names = list(det.feature_names), list(det.label_names)
    rng = np.random.default_rng(SEED)
    frames: dict[str, pd.DataFrame] = {}
    for f in sorted(C18.glob("*.csv")):
        for chunk in pd.read_csv(f, chunksize=300_000, low_memory=False):
            chunk = chunk[chunk["Label"] != "Label"]
            chunk, _ = align(chunk, feats)
            for lab, g in chunk.groupby("Label"):
                fam = family_2018(str(lab))
                if fam is None:
                    continue
                g = g[[c for c in feats if c in g.columns]].assign(_native=str(lab).strip(), _family=fam)
                _sample(frames, g, str(lab).strip(), per_label, rng)
        _log.info("CIC-IDS2018 %s read: %s", f.name, {k: len(v) for k, v in frames.items()})
    out = _finish(frames, feats, det, names)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def diad(det: Any, cache: Path, per_label: int) -> dict[str, np.ndarray]:
    path = cache / f"diad_{per_label}.npz"
    if path.is_file():
        return dict(np.load(path))
    from src.soc.columns import align

    feats, names = list(det.feature_names), list(det.label_names)
    rng = np.random.default_rng(SEED)
    frames: dict[str, pd.DataFrame] = {}
    for top, fam in FAMILY_DIAD.items():
        for f in sorted((DIAD / top).rglob("*.csv")):
            native = f.parent.name if f.parent != DIAD / top else top  # e.g. "DDOS HTTP FLOOD"
            for chunk in pd.read_csv(f, chunksize=300_000, low_memory=False):
                chunk, _ = align(chunk, feats)
                g = chunk[[c for c in feats if c in chunk.columns]].assign(_native=native, _family=fam)
                _sample(frames, g, native, per_label, rng)
        _log.info("DIAD %s read: %s", top, {k: len(v) for k, v in frames.items()})
    out = _finish(frames, feats, det, names)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def metrics(pred: np.ndarray, y: np.ndarray, benign: int) -> dict[str, Any]:
    return imf._metrics_from_pred(pred, y, benign) | {"macro_f1": imf._f1_macro(pred, y)}


def per_native(pred: np.ndarray, native: np.ndarray, benign: int) -> dict[str, float]:
    return {str(n): round(float((pred[native == n] != benign).mean()), 4) for n in sorted(set(native))}


ADV_EPS: list[float] = [0.1]  # PGD strengths of the adversarial rows (--adv-eps)
HIKARI_FAMILY = {"benign": "BENIGN", "background": "BENIGN", "bruteforce": "BruteForce", "bruteforce-xml": "BruteForce",
                 "probing": "PortScan", "xmrigcc cryptominer": "Bot"}  # fmt: skip


def hikari(det: Any, cache: Path) -> dict[str, np.ndarray]:
    """HIKARI-2021 + 2022 (Zeek FlowMeter, converted), owner-released 2026-10-25 after blind
    round 2 (NOT BLIND); 70 / 15 / 15 stratified by category."""
    path = cache / "hikari.npz"
    if path.is_file():
        return dict(np.load(path))
    from scripts.blind_test_v2 import _hikari
    from src.soc.columns import align

    feats, names = list(det.feature_names), list(det.label_names)
    df = _hikari()
    cat = df["attack_category"].where(df["attack_category"].notna(), df.get("traffic_category")).astype(str).str.lower()
    lab = pd.to_numeric(df["Label"], errors="coerce").fillna(0).to_numpy() > 0
    fam = np.where(lab, cat.map(HIKARI_FAMILY).fillna("Bot").to_numpy(), "BENIGN")
    aligned, _ = align(df, feats)
    X = imf._clean(aligned, feats, det.input_stats["median"])
    y = np.array([names.index(f) for f in fam])
    native = (df["_set"].astype(str) + ":" + cat).to_numpy().astype(str)
    tr, va, te = imf._split(len(y), native)
    out = {"X_train": X[tr], "y_train": y[tr], "X_val": X[va], "y_val": y[va], "X_test": X[te], "y_test": y[te],
           "native_test": native[te]}  # fmt: skip
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def run(per_label: int, cache: Path, base: Path | None = None, use_hikari: bool = False,
        adversarial: bool = False) -> dict[str, Any]:  # fmt: skip
    import joblib
    import lightgbm
    from lightgbm import LGBMClassifier

    from src.soc.model import Detector

    det = Detector(base or imf.DEPLOY)  # base: the scorer the stack sits on
    names = list(det.label_names)
    benign = names.index("BENIGN")
    raw = {"cic-ids2017": imf.cic17(det), "cic-ddos2019": imf.ddos19(det),
           "cic-ids2018": cic18(det, cache, per_label), "cic-iot-diad-2024": diad(det, cache, per_label)}  # fmt: skip
    if use_hikari:
        raw["hikari"] = hikari(det, cache)

    def S(x: np.ndarray) -> np.ndarray:  # noqa: N802 - model space (deployed scaler)
        return np.asarray(det.scaler.transform(x), dtype=np.float32)

    sets = {d: {s: (S(r[f"X_{s}"]), r[f"y_{s}"]) for s in ("train", "val", "test")} for d, r in raw.items()}
    champion = det.ensemble or det.model
    imf._ram_ok(3.0)
    # balanced share per dataset in training: no network dominates
    rng = np.random.default_rng(SEED)
    cap = 250_000
    parts = []
    for d in sets:
        X, y = sets[d]["train"]
        idx = rng.permutation(len(y))[:cap]
        parts.append((X[idx], y[idx]))
    Xtr, ytr = np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    adv_rows = 0
    if adversarial:  # PGD versions of attack rows (true class kept): the generalist learns them too
        import torch

        from scripts import robust_deployed as rd

        sp = {"feature_names": list(det.feature_names), "train_min": raw["cic-ids2017"]["X_train"].min(0),
              "train_max": raw["cic-ids2017"]["X_train"].max(0)}  # fmt: skip
        cons = rd.build_constraints(det, sp)
        mask = torch.as_tensor(cons.mask.astype(np.float32))
        inc = torch.as_tensor(rd._increase_only(cons))
        att = np.flatnonzero(ytr != benign)
        adv, adv_y = [], []
        for eps in ADV_EPS:  # one batch of attack rows per strength (v4: 0.1 and 0.3)
            pick = rng.choice(att, size=min(150_000 // len(ADV_EPS) * (2 if len(ADV_EPS) > 1 else 1), len(att)),
                              replace=False)  # fmt: skip
            for k in range(0, len(pick), 20_000):
                ix = pick[k : k + 20_000]
                xa = imf._pgd_rows(det.model, torch.as_tensor(Xtr[ix]), torch.as_tensor(ytr[ix]), mask, inc, eps)
                adv.append(xa.numpy().astype(np.float32))
                adv_y.append(ytr[ix])
        Xtr = np.vstack([Xtr, *adv])
        ytr = np.concatenate([ytr, *adv_y])
        adv_rows = int(sum(len(a) for a in adv))
    Xva = np.vstack([sets[d]["val"][0] for d in sets])
    yva = np.concatenate([sets[d]["val"][1] for d in sets])
    present = sorted(set(ytr.tolist()))
    t0 = time.time()
    gen = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                         class_weight="balanced", random_state=SEED, n_jobs=8, verbose=-1)  # fmt: skip
    gen.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lightgbm.early_stopping(40, verbose=False)])
    train_s = time.time() - t0
    _log.info("generalist: %.0f s, %s trees, classes %s", train_s, gen.best_iteration_, present)
    cls = np.asarray(gen.classes_)
    bcol = int(np.flatnonzero(cls == benign)[0])

    def gen_scores(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = gen.predict_proba(X)
        att = 1.0 - p[:, bcol]
        q = p.copy()
        q[:, bcol] = -1
        return att, cls[q.argmax(1)]

    Pc = {d: {s: imf._proba(champion, sets[d][s][0]) for s in ("val", "test")} for d in sets}
    pc = {d: {s: Pc[d][s].argmax(1) for s in ("val", "test")} for d in sets}
    gs = {d: {s: gen_scores(sets[d][s][0]) for s in ("val", "test")} for d in sets}
    n_cls = len(names)

    def gen_full(X: np.ndarray) -> np.ndarray:  # generalist probabilities in the 9 deployed columns
        p = gen.predict_proba(X)
        out = np.zeros((len(X), n_cls), dtype=np.float32)
        out[:, cls] = p
        return out

    Pg = {d: {s: gen_full(sets[d][s][0]) for s in ("val", "test")} for d in sets}

    # stack: a small model over both models' log-probabilities, fitted on half of every
    # dataset's VALIDATION split (each dataset weighted equally); its alert threshold is
    # set on the other half so that on CIC-IDS2017 and CIC-DDoS2019 it raises no more
    # false alarms than the champion does there. TEST is untouched until the end.
    def meta(d: str, s: str) -> np.ndarray:
        return np.hstack([np.log(Pc[d][s] + 1e-6), np.log(Pg[d][s] + 1e-6)]).astype(np.float32)

    halves = {}
    for d in sets:
        n = len(sets[d]["val"][1])
        perm = np.random.default_rng(SEED + 1).permutation(n)
        halves[d] = (perm[: n // 2], perm[n // 2 :])
    Xa = np.vstack([meta(d, "val")[halves[d][0]] for d in sets])
    ya = np.concatenate([sets[d]["val"][1][halves[d][0]] for d in sets])
    wa = np.concatenate([np.full(len(halves[d][0]), 1.0 / len(halves[d][0])) for d in sets])
    stack = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50,
                           reg_lambda=1.0, random_state=SEED, n_jobs=8, verbose=-1)  # fmt: skip
    stack.fit(Xa, ya, sample_weight=wa * len(wa))
    scls = np.asarray(stack.classes_)
    sb = int(np.flatnonzero(scls == benign)[0])

    def stack_scores(d: str, s: str, idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        X = meta(d, s) if idx is None else meta(d, s)[idx]
        p = stack.predict_proba(X)
        att = 1.0 - p[:, sb]
        q = p.copy()
        q[:, sb] = -1
        return att, scls[q.argmax(1)]

    def fpr(alert: np.ndarray, y: np.ndarray) -> float:
        return float((alert & (y == benign)).sum()) / max(1, int((y == benign).sum()))

    thr_stack = 0.999
    for t in np.unique(np.concatenate([np.linspace(0.5, 0.99, 50), [0.995, 0.999]])):
        ok = True
        for d in ("cic-ids2017", "cic-ddos2019"):
            idx = halves[d][1]
            yb = sets[d]["val"][1][idx]
            att, _ = stack_scores(d, "val", idx)
            if fpr(att >= t, yb) > fpr(pc[d]["val"][idx] != benign, yb) + 1e-12:
                ok = False
                break
        if ok:
            thr_stack = float(t)
            break

    def stacked(d: str, s: str) -> np.ndarray:
        att, fam = stack_scores(d, s)
        return np.where(att >= thr_stack, fam, benign)

    # stack2: the stack decides alert / no alert, the champion keeps naming the family
    # when it already calls the flow an attack; threshold on the second validation half:
    # home false alarms not higher AND home recall not lower than the champion's, then the
    # best mean recall on the two new networks (falls back to the stack's threshold)
    def recall(alert: np.ndarray, y: np.ndarray) -> float:
        return float((alert & (y != benign)).sum()) / max(1, int((y != benign).sum()))

    thr_stack2, best_new = thr_stack, -1.0
    for t in np.unique(np.concatenate([np.linspace(0.05, 0.99, 95), [0.995, 0.999]])):
        ok = True
        for d in ("cic-ids2017", "cic-ddos2019"):
            idx = halves[d][1]
            yb = sets[d]["val"][1][idx]
            att, _ = stack_scores(d, "val", idx)
            champ_alert = pc[d]["val"][idx] != benign
            if fpr(att >= t, yb) > fpr(champ_alert, yb) + 1e-12 or recall(att >= t, yb) < recall(champ_alert, yb) - 1e-12:
                ok = False
                break
        if not ok:
            continue
        new = np.mean([recall(stack_scores(d, "val", halves[d][1])[0] >= t, sets[d]["val"][1][halves[d][1]])
                       for d in ("cic-ids2018", "cic-iot-diad-2024")])  # fmt: skip
        if new > best_new:
            thr_stack2, best_new = float(t), float(new)

    def stacked2(d: str, s: str) -> np.ndarray:
        att, fam = stack_scores(d, s)
        champ = pc[d][s]
        out = np.where(champ != benign, champ, fam)
        return np.where(att >= thr_stack2, out, benign)
    y17v = sets["cic-ids2017"]["val"][1]
    passed = (y17v == benign) & (pc["cic-ids2017"]["val"] == benign)
    top = float(gs["cic-ids2017"]["val"][0][passed].max()) if passed.any() else 0.5
    thr_home = float(min(max(float(np.nextafter(top, np.inf)), 0.5), 1.0))
    newb = np.concatenate([gs[d]["val"][0][(sets[d]["val"][1] == benign) & (pc[d]["val"] == benign)]
                           for d in ("cic-ids2018", "cic-iot-diad-2024")])  # fmt: skip
    nb_all = sum(int((sets[d]["val"][1] == benign).sum()) for d in ("cic-ids2018", "cic-iot-diad-2024"))
    # 1 % of all new-network validation benign flows may be added as false alarms
    k = int(0.01 * nb_all)
    thr_1 = float(np.sort(newb)[::-1][k]) if 0 < k < len(newb) else thr_home
    thr_1 = float(min(max(thr_1, 0.5), thr_home))

    def cascade(d: str, s: str, thr: float) -> np.ndarray:
        p = pc[d][s].copy()
        att, fam = gs[d][s]
        hit = (p == benign) & (att >= thr)
        p[hit] = fam[hit]
        return p

    def alone(d: str, s: str) -> np.ndarray:
        att, fam = gs[d][s]
        return np.where(att >= 0.5, fam, benign)

    variants = {"champion": lambda d, s: pc[d][s], "cascade": lambda d, s: cascade(d, s, thr_home),
                "cascade@1%": lambda d, s: cascade(d, s, thr_1), "generalist": alone,
                "stack": stacked, "stack2": stacked2}  # fmt: skip
    rows: dict[str, Any] = {}
    for v, fn in variants.items():
        rows[v] = {s: {d: metrics(fn(d, s), sets[d][s][1], benign) for d in sets} for s in ("val", "test")}
        rows[v]["test_by_native"] = {d: per_native(fn(d, "test"), raw[d]["native_test"], benign)
                                     for d in ("cic-ddos2019", "cic-ids2018", "cic-iot-diad-2024")}  # fmt: skip
    x = sets["cic-ids2017"]["test"][0][:20000]
    import torch

    class Both(torch.nn.Module):  # timing: champion + generalist on the same rows
        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            gen.predict_proba(xb.numpy())
            return champion(xb)

    rows["champion"]["flows_per_s"] = imf.speed(champion, x)
    rows["cascade"]["flows_per_s"] = imf.speed(Both(), x)
    home = ("cic-ids2017", "cic-ddos2019")
    ref = "champion"
    if base is not None:  # compare with the deployed detector (the current stack)
        dep = Detector(imf.DEPLOY)
        dsc = dep.ensemble or dep.model
        rows["deployed"] = {"test": {d: metrics(imf._proba(dsc, sets[d]["test"][0]).argmax(1), sets[d]["test"][1], benign)
                                     for d in sets}}  # fmt: skip
        ref = "deployed"
        try:  # evasion: PGD on the base MLP transferred, deployed vs candidate stack
            from scripts import robust_deployed as rd
            from src.soc.model import StackCascade

            sp = {"feature_names": list(det.feature_names), "train_min": raw["cic-ids2017"]["X_train"].min(0),
                  "train_max": raw["cic-ids2017"]["X_train"].max(0)}  # fmt: skip
            cons = rd.build_constraints(det, sp)
            cand = StackCascade(champion, gen, stack, benign, len(names), det.temperature).eval()
            Xs, ys = sets["cic-ids2017"]["test"]
            rows["deployed"]["robustness"] = imf.robustness(dsc, det.model, None, Xs, ys, benign, cons, thr_stack)
            rows["stack"]["robustness"] = imf.robustness(cand, det.model, None, Xs, ys, benign, cons, thr_stack)
        except Exception as exc:  # noqa: BLE001 - reported, the comparison still stands
            rows["stack"]["robustness_error"] = str(exc)[:200]
    verdict = {}
    for v in ("cascade", "cascade@1%", "generalist", "stack", "stack2"):
        ok = all(imf.not_worse(rows[v]["test"][d], rows[ref]["test"][d]) for d in home)
        verdict[v] = f"passes the owner rule vs {ref}" if ok else f"fails the owner rule vs {ref} on " + ", ".join(
            d for d in home if not imf.not_worse(rows[v]["test"][d], rows[ref]["test"][d]))
    CANDIDATE.mkdir(parents=True, exist_ok=True)
    joblib.dump({"stack": stack, "threshold_stack": thr_stack, "threshold_stack2": thr_stack2},
                CANDIDATE / "stack.joblib", compress=3)
    joblib.dump({"model": gen, "threshold_home": thr_home, "threshold_1pct": thr_1, "benign": benign,
                 "feature_space": "deployed scaler output (artifacts/deploy/scaler.pkl)",
                 "trained_on": list(sets), "trained": time.strftime("%Y-%m-%d %H:%M")},
                CANDIDATE / "generalist.joblib", compress=3)  # fmt: skip
    return {"what": "CICFlowMeter detector across networks: champion vs multi-dataset generalist",
            "datasets": {d: {s: int(len(sets[d][s][1])) for s in ("train", "val", "test")} for d in sets},
            "per_label": per_label, "new_blind_used": False, "train_seconds": round(train_s),
            "trees": int(gen.best_iteration_ or 600), "threshold_home": thr_home, "threshold_1pct": thr_1,
            "threshold_stack": thr_stack, "threshold_stack2": thr_stack2, "adversarial_rows": adv_rows,
            "base": str(base or imf.DEPLOY), "not_blind": ["hikari"] if use_hikari else [],
            "rows": rows, "verdict": verdict, "candidate": str(CANDIDATE / "generalist.joblib")}  # fmt: skip


TAG = ["v2"]


def write(res: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"improve_flows_{TAG[0]}.json").write_text(json.dumps(res, indent=1, default=float), "utf-8")
    lines = ["# Flow detector across networks (multi-dataset generalist)", "",
             "| Variant | Dataset | Precision | Attack recall | False alarms | Macro-F1 |", "|---|---|---|---|---|---|"]
    for v, r in res["rows"].items():
        for d, m in r["test"].items():
            lines.append(f"| {v} | {d} | {m['precision']:.4f} | {m['recall']:.4f} | "
                         f"{m['false_alarm_rate']:.4%} | {m['macro_f1']:.3f} |")  # fmt: skip
    lines += ["", "Verdicts: " + "; ".join(f"{k}: {v}" for k, v in res["verdict"].items())]
    (OUT / f"improve_flows_{TAG[0]}.md").write_text("\n".join(lines) + "\n", "utf-8")


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-label", type=int, default=PER_LABEL)
    ap.add_argument("--cache", type=Path, default=Path("D:/SplitShield-data/cache/flows_v2"))
    ap.add_argument("--out", type=Path, default=None, help="write results here instead of results/extra")
    ap.add_argument("--base", type=Path, default=None, help="detector folder the stack sits on (else the deployed one)")
    ap.add_argument("--hikari", action="store_true", help="add HIKARI (owner-released, NOT BLIND)")
    ap.add_argument("--adversarial", action="store_true", help="add PGD attack rows to the generalist's training")
    ap.add_argument("--tag", default="v2", help="results / candidate name (improve_flows_<tag>)")
    ap.add_argument("--adv-eps", type=float, nargs="+", default=[0.1], help="PGD strengths for --adversarial")
    a = ap.parse_args()
    ADV_EPS[:] = a.adv_eps
    global OUT, CANDIDATE
    if a.out:
        OUT = a.out
    TAG[0] = a.tag
    if a.tag != "v2":
        CANDIDATE = CANDIDATE.with_name(f"flow_generalist_{a.tag}")
    res = run(a.per_label, a.cache, a.base, a.hikari, a.adversarial)
    write(res)
    print(json.dumps(res["verdict"], indent=1))


if __name__ == "__main__":
    main()
