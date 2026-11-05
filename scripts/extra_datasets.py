"""Extra datasets (owner 2026-10-14): test the detector recipe on datasets the
project was not designed on, and score the log -> ATT&CK mapping on real attack logs.

Outside the registered design (results/FINAL_REPORT.md is untouched): every dataset
gets the same recipe as the deployed detector -- Random Forest, LightGBM, the MLP and
the validation-chosen ensemble -- with train / validation / test macro-F1, a fit
verdict, per-class F1 and a confusion matrix. Selection on validation, test only
reported. Results: results/extra/ and results/figures/extra_*.png (the dashboard's
Project results page shows them).

    python -m scripts.extra_datasets                    # all datasets + logs
    python -m scripts.extra_datasets --only logs        # log mapping only
    python -m scripts.extra_datasets --only nf-ton-iot-v2 --seeds 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from utils.logging_config import get_logger
from utils.paths import RAW_DIR, RESULTS_DIR

_log = get_logger(__name__)
OUT = RESULTS_DIR / "extra"
FIG = RESULTS_DIR / "figures"
DATASETS = ("nf-ton-iot-v2", "nf-unsw-nb15-v2-multi", "cic-ddos2019")
NAMES = {
    "nf-ton-iot-v2": "NF-ToN-IoT-v2",
    "nf-unsw-nb15-v2-multi": "NF-UNSW-NB15-v2 (10-class)",
    "cic-ddos2019": "CIC-DDoS2019 (03-11)",
}
MODELS = ("rf", "lgbm", "mlp", "ensemble")
MODEL_NAMES = {
    "rf": "Random Forest",
    "lgbm": "LightGBM",
    "mlp": "MLP",
    "ensemble": "Ensemble",
}
GAP = 0.05  # same over/underfitting margin as src/eval/fitcheck.py


def _cfg(ds: str, scheme: str, seed: int) -> Any:
    from config.base import build_config

    raw = {  # same caps and network size as the registered NetFlow experiment (E8)
        "id": f"X_{ds}",
        "dataset": ds,
        "label_schema": scheme,
        "split": "random",
        "subsample": {"rare_floor": 5000, "class_cap": 60000},
        "model": {"hidden_layers": 4, "hidden_width": 128, "dropout": 0.2},
    }
    return build_config(raw, seed)


def _f1(y: np.ndarray, p: np.ndarray, labels: list[int]) -> float:
    """Macro-F1 over the classes PRESENT in the data. A scheme can list classes a
    capture never contains (CIC-DDoS2019 day 03-11 has 8 of its 14); averaging
    those in as F1 = 0 would understate every model."""
    from sklearn.metrics import f1_score

    return float(f1_score(y, p, average="macro", labels=labels, zero_division=0))


def _verdict(train: float, val: float, test: float, best_val: float) -> str:
    if train - val > GAP and train - test > GAP:
        return "overfitting"
    if val < best_val - GAP:
        return "underfitting (high bias vs the best model)"
    return "good fit"


def run_seed(ds: str, raw: Any, seed: int) -> dict[str, Any]:
    import torch
    from sklearn.metrics import confusion_matrix

    from src.data.preprocess import preprocess_frame
    from src.data.sources import get_spec
    from src.eval.fitcheck import with_reference
    from src.eval.metrics import classification_metrics
    from src.models.ensemble import EnsembleModel, choose, fit_trees
    from src.training.centralized import train_centralized
    from utils.device import get_primary_device

    spec = get_spec(ds)
    cfg = _cfg(ds, spec.label_scheme, seed)
    dev = get_primary_device()
    t0 = time.time()
    data = preprocess_frame(raw, spec, cfg, seed=seed)
    n = len(data.label_names)
    present = sorted(
        int(c) for c in set(np.unique(data.y_train)) | set(np.unique(data.y_test))
    )
    trees = fit_trees(data, seed)
    mlp, _, mlp_m = train_centralized(data, cfg, device=dev)

    @torch.no_grad()
    def pred(model: Any, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba") and not isinstance(model, torch.nn.Module):
            return np.asarray(model.predict(X), dtype=int)
        model.eval()
        out = []
        for i in range(0, len(X), 65_536):
            xb = torch.as_tensor(X[i : i + 65_536], dtype=torch.float32, device=dev)
            out.append(model(xb).argmax(1).cpu().numpy())
        return np.concatenate(out)

    ens, cands = choose(
        mlp, trees, data, dev, lambda m: _f1(data.y_val, pred(m, data.X_val), present)
    )
    models = {"rf": trees["rf"], "lgbm": trees["lgbm"], "mlp": mlp, "ensemble": ens}
    res: dict[str, Any] = {
        "seed": seed,
        "classes": list(data.label_names),
        "present_classes": [data.label_names[i] for i in present],
        "class_counts": data.stats["class_counts"],
        "n_features": len(data.feature_names),
        "ensemble_members": sorted(ens.weights)
        if isinstance(ens, EnsembleModel)
        else [],
        "ensemble_candidates": cands,
        "models": {},
    }
    for name, m in models.items():
        p_test = pred(m, data.X_test)
        cm = classification_metrics(data.y_test, p_test, data.label_names)
        res["models"][name] = {
            "train": _f1(data.y_train, pred(m, data.X_train), present),
            "val": _f1(data.y_val, pred(m, data.X_val), present),
            "test": _f1(data.y_test, p_test, present),
            "attack_recall": cm["attack_recall"],
            "binary_fpr": cm["binary_fpr"],
            "per_class_f1": {
                k: v["f1"]
                for i, (k, v) in enumerate(cm["per_class"].items())
                if i in present
            },
            "confusion": confusion_matrix(
                data.y_test, p_test, labels=list(range(n))
            ).tolist(),
        }
    best_val = max(r["val"] for r in res["models"].values())
    for r in res["models"].values():
        r["fit"] = _verdict(r["train"], r["val"], r["test"], best_val)
    res["mlp_fitcheck"] = with_reference(
        mlp_m["fit"], res["models"]["rf"]["test"], "Random Forest"
    )
    res["seconds"] = round(time.time() - t0)
    return res


# --- log mapping (EVTX-ATTACK-SAMPLES) -------------------------------------------

TACTIC_DIRS = {
    "Command and Control": "command-and-control",
    "Credential Access": "credential-access",
    "Defense Evasion": "defense-evasion",
    "Discovery": "discovery",
    "Execution": "execution",
    "Lateral Movement": "lateral-movement",
    "Persistence": "persistence",
    "Privilege Escalation": "privilege-escalation",
}


def is_dev(name: str) -> bool:
    """Fixed half of the EVTX samples used to WRITE mapping rules; the other half and
    all of OTRF are held out and only scored (owner 2026-10-14: fair test)."""
    import hashlib

    return hashlib.sha1(Path(name).name.encode()).digest()[0] % 2 == 0


def _score(
    evs: list[dict[str, Any]], truth_tac: list[str], truth_tech: list[str]
) -> dict[str, Any]:
    from src.knowledge.attack_details import tactics_of

    techs = sorted({t for e in evs for t in e.get("techniques", [])})
    tactics = sorted({x for t in techs for x in tactics_of(t)})
    return {
        "events": len(evs),
        "techniques": techs,
        "tactics": tactics,
        "detected": bool(techs),
        "correct": bool(set(tactics) & set(truth_tac)),
        "correct_technique": (
            bool({t.split(".")[0] for t in techs} & set(truth_tech))
            if truth_tech
            else None
        ),
    }


def _otrf_one(z: str, tacs: list[str], techs: list[str]) -> dict[str, Any]:
    """Score one OTRF recording (runs in a worker process)."""
    import zipfile

    from src.soc.parsers import parse_line, reset_sessions

    reset_sessions()
    base = RAW_DIR / "otrf-security-datasets"
    evs: list[dict[str, Any]] = []
    err = ""
    try:
        with zipfile.ZipFile(z) as zz:
            for n in zz.namelist():
                if not n.lower().endswith(".json"):
                    continue
                with zz.open(n) as fh:  # line by line: some recordings are large
                    for raw in fh:
                        ev = parse_line(raw.decode("utf-8", "replace"))
                        if ev and ev["techniques"]:
                            evs.append({"techniques": ev["techniques"]})
    except (zipfile.BadZipFile, OSError) as exc:
        err = str(exc)[:160]
    return {
        "source": "otrf", "split": "heldout", "file": Path(z).relative_to(base).as_posix(),
        "tactic": tacs[0], "truth_tactics": tacs, "truth_techniques": techs,
        "error": err, **_score(evs, tacs, techs),
    }  # fmt: skip


def _otrf_jobs() -> list[tuple[str, list[str], list[str]]]:
    """OTRF Security-Datasets (Windows atomic) with labels from each dataset's
    metadata (ATT&CK techniques + tactics). Entirely held out."""
    import yaml

    from src.knowledge.attack_details import TACTICS

    base = RAW_DIR / "otrf-security-datasets"
    by_id = {v["id"]: k for k, v in TACTICS.items()}
    labels: dict[str, tuple[list[str], list[str]]] = {}
    for m in sorted((base / "_metadata").glob("*.yaml")):
        doc = yaml.safe_load(m.read_text("utf-8")) or {}
        maps = doc.get("attack_mappings") or []
        techs = sorted({str(x["technique"]) for x in maps if x.get("technique")})
        tacs = sorted(
            {by_id[t] for x in maps for t in (x.get("tactics") or []) if t in by_id}
        )
        for f in doc.get("files") or []:
            labels[Path(str(f.get("link", ""))).name] = (tacs, techs)
    jobs = []
    for z in sorted((base / "windows").rglob("*.zip")):
        tacs, techs = labels.get(z.name, ([], []))
        if tacs:
            jobs.append((str(z), tacs, techs))
    return jobs


def _evtx_one(f: str) -> dict[str, Any] | None:
    """Score one EVTX sample (runs in a worker process)."""
    from src.soc.parsers import read_evtx, reset_sessions

    reset_sessions()  # each file on its own (workers are reused across files)
    base = RAW_DIR / "evtx-attack-samples"
    rel = Path(f).relative_to(base).parts  # <repo>-master/<Tactic>/[...]/file.evtx
    truth = TACTIC_DIRS.get(rel[1] if len(rel) > 2 else "")
    if truth is None:  # 'Other' / tool folders carry no tactic label
        return None
    try:
        evs, err = read_evtx(f, max_events=200_000), ""  # 2026-10-25: was 3,000 (later events were never scored)
    except RuntimeError as exc:
        evs, err = [], str(exc)[:160]
    return {
        "source": "evtx", "split": "dev" if is_dev(Path(f).name) else "heldout",
        "file": Path(f).relative_to(base).as_posix(), "tactic": truth,
        "truth_tactics": [truth], "truth_techniques": [], "error": err,
        **_score(evs, [truth], []),
    }  # fmt: skip


def _all_rows(workers: int) -> list[dict[str, Any]]:
    """EVTX + OTRF rows, scored in parallel (one file per task)."""
    from concurrent.futures import ProcessPoolExecutor

    evtx = sorted(str(f) for f in (RAW_DIR / "evtx-attack-samples").rglob("*.evtx"))
    otrf = _otrf_jobs()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for r in pool.map(_evtx_one, evtx) if r]
        # biggest recordings first so no single worker is left with a long tail
        otrf.sort(key=lambda j: -Path(j[0]).stat().st_size)
        rows += list(pool.map(_otrf_one, *zip(*otrf, strict=True)))
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(rows))
    tech = [r for r in rows if r["correct_technique"] is not None]
    return {
        "files": len(rows),
        "detected_rate": sum(r["detected"] for r in rows) / n,
        "correct_tactic_rate": sum(r["correct"] for r in rows) / n,
        "correct_technique_rate": (
            sum(bool(r["correct_technique"]) for r in tech) / len(tech)
            if tech
            else None
        ),
    }


def run_logs(tag: str = "current", workers: int = 1) -> dict[str, Any]:
    """Score the log -> ATT&CK mapping on real attack logs. A file is 'detected' when
    any event maps to a technique and 'correct' when a mapped technique belongs to
    one of its labelled tactics. Headline = the HELD-OUT files only."""
    from collections import Counter

    rows = _all_rows(workers)
    held = [r for r in rows if r["split"] == "heldout"]
    per: dict[str, dict[str, Any]] = {}
    for r in held:
        for t in r["truth_tactics"][:1]:
            p = per.setdefault(t, {"files": 0, "detected": 0, "correct": 0})
            p["files"] += 1
            p["detected"] += r["detected"]
            p["correct"] += r["correct"]
    summ = {
        "heldout": _summary(held),
        "heldout_evtx": _summary([r for r in held if r["source"] == "evtx"]),
        "heldout_otrf": _summary([r for r in held if r["source"] == "otrf"]),
        "dev": _summary([r for r in rows if r["split"] == "dev"]),
    }
    path = OUT / "log_mapping.json"
    runs = {}
    if path.is_file():
        runs = json.loads(path.read_text("utf-8")).get("runs", {})
    runs[tag] = summ
    all_techs: list[str] = [t for r in held for t in r["techniques"]]
    return {
        "files": len(rows),
        "labelled_files": len(held),
        "unreadable": sum(bool(r["error"]) for r in rows),
        "detected_rate": summ["heldout"]["detected_rate"],
        "correct_tactic_rate": summ["heldout"]["correct_tactic_rate"],
        "summary": summ,
        "runs": runs,
        "per_tactic": per,
        "top_techniques": Counter(all_techs).most_common(15),
        "rows": rows,
    }


# --- figures + report ----------------------------------------------------------------


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    return float(np.mean(xs)), float(np.std(xs))


def figures(results: dict[str, Any], logs: dict[str, Any] | None) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    colors = {
        "rf": "#4C72B0",
        "lgbm": "#55A868",
        "mlp": "#C44E52",
        "ensemble": "#8172B2",
    }
    dss = [d for d in DATASETS if results.get(d, {}).get("seeds")]
    if dss:  # 1. overview: test macro-F1 per dataset and model, mean +- sd over seeds
        fig, ax = plt.subplots(figsize=(10, 4.8))
        w = 0.2
        for i, m in enumerate(MODELS):
            stats = [
                _mean_sd([s["models"][m]["test"] for s in results[d]["seeds"]])
                for d in dss
            ]
            xs = np.arange(len(dss)) + (i - 1.5) * w
            ax.bar(xs, [s[0] for s in stats], w, yerr=[s[1] for s in stats],
                   label=MODEL_NAMES[m], color=colors[m], capsize=3)  # fmt: skip
            for x, (mu, _) in zip(xs, stats, strict=True):
                ax.text(x, mu + 0.01, f"{mu:.2f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(dss)), [NAMES[d] for d in dss])
        ax.set_ylabel("Test macro-F1")
        ax.set_ylim(0, 1.08)
        ax.set_title("Extra datasets: test macro-F1 (mean ± sd over seeds)")
        ax.legend(ncol=4, loc="lower center", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "extra_overview.png", dpi=130)
        plt.close(fig)
        made.append("extra_overview.png")
    for d in dss:
        seeds = results[d]["seeds"]
        classes = seeds[0]["classes"]
        # 2. fit: train / val / test per model
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for j, (split, alpha) in enumerate(
            (("train", 0.45), ("val", 0.7), ("test", 1.0))
        ):
            vals = [np.mean([s["models"][m][split] for s in seeds]) for m in MODELS]
            ax.bar(np.arange(len(MODELS)) + (j - 1) * 0.27, vals, 0.27, alpha=alpha,
                   color=[colors[m] for m in MODELS], label=split, edgecolor="black")  # fmt: skip
        verdicts = [seeds[0]["models"][m]["fit"] for m in MODELS]
        ax.set_xticks(range(len(MODELS)), [f"{MODEL_NAMES[m]}\n({v.split(' (')[0]})"
                                            for m, v in zip(MODELS, verdicts, strict=True)])  # fmt: skip
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"{NAMES[d]}: fit check (bars: train / validation / test)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / f"extra_fit_{d}.png", dpi=130)
        plt.close(fig)
        # 3. confusion matrix of the chosen ensemble (seed 0), row-normalised
        cm = np.array(seeds[0]["models"]["ensemble"]["confusion"], dtype=float)
        rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(
            figsize=(1.0 + 0.62 * len(classes), 0.8 + 0.55 * len(classes))
        )
        im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(rown):
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else "black")  # fmt: skip
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{NAMES[d]}: confusion (ensemble, test, seed 0)", fontsize=9)
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fig.savefig(FIG / f"extra_confusion_{d}.png", dpi=130)
        plt.close(fig)
        # 4. per-class test F1
        fig, ax = plt.subplots(figsize=(8, 0.9 + 0.42 * len(classes)))
        for i, m in enumerate(MODELS):
            v = [
                np.mean([s["models"][m]["per_class_f1"].get(c, 0) for s in seeds])
                for c in classes
            ]
            ax.barh(np.arange(len(classes)) + (i - 1.5) * 0.2, v, 0.2,
                    color=colors[m], label=MODEL_NAMES[m])  # fmt: skip
        ax.set_yticks(range(len(classes)), classes)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Test F1")
        ax.set_title(f"{NAMES[d]}: per-class F1")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / f"extra_perclass_{d}.png", dpi=130)
        plt.close(fig)
        made += [
            f"extra_fit_{d}.png",
            f"extra_confusion_{d}.png",
            f"extra_perclass_{d}.png",
        ]
    if logs:  # 5. log mapping per tactic
        per = logs["per_tactic"]
        tac = sorted(per, key=lambda t: -per[t]["files"])
        fig, ax = plt.subplots(figsize=(9, 4.2))
        x = np.arange(len(tac))
        det = [per[t]["detected"] / per[t]["files"] for t in tac]
        cor = [per[t]["correct"] / per[t]["files"] for t in tac]
        ax.bar(x - 0.2, det, 0.4, label="mapped to any technique", color="#4C72B0")
        ax.bar(x + 0.2, cor, 0.4, label="mapped to the right tactic", color="#55A868")
        for i, t in enumerate(tac):
            ax.text(i, 1.02, f"n={per[t]['files']}", ha="center", fontsize=7)
        ax.set_xticks(x, [t.replace("-", "\n") for t in tac], fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Share of sample files")
        ax.set_title(
            "Log → ATT&CK mapping, held-out real attack logs (EVTX half + OTRF)"
        )
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "extra_logmap.png", dpi=130)
        plt.close(fig)
        made.append("extra_logmap.png")
    return made


def report(results: dict[str, Any], logs: dict[str, Any] | None) -> str:
    L = [
        "# 🧪 Extra datasets — detector recipe and log mapping",
        "",
        "_Outside the registered design (owner 2026-10-14). Same recipe as the deployed "
        "detector; model choice on validation, test reported only. Downloaded without "
        "sign-up (Hugging Face mirrors of the original releases; GitHub for the logs)._",
        "",
    ]
    for d in DATASETS:
        r = results.get(d)
        if not r or not r.get("seeds"):
            continue
        seeds = r["seeds"]
        s0 = seeds[0]
        L += [f"## 📦 {NAMES[d]}", "",
              f"{len(s0['classes'])} classes ({', '.join(s0['classes'])}); "
              f"{s0['n_features']} features; seeds {[s['seed'] for s in seeds]}; "
              f"ensemble chosen on validation: **{' + '.join(s0['ensemble_members'])}**.", "",
              "| Model | Train | Validation | Test (mean ± sd) | Attacks caught | False alarms | Fit |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
        for m in MODELS:
            g = lambda k, m=m, seeds=seeds: [s["models"][m][k] for s in seeds]  # noqa: E731
            mu, sd = _mean_sd(g("test"))
            fit = s0["models"][m]["fit"]
            icon = "✅" if fit == "good fit" else "⚠️"
            L.append(
                f"| {MODEL_NAMES[m]} | {np.mean(g('train')):.3f} | {np.mean(g('val')):.3f} | "
                f"**{mu:.3f}** ± {sd:.3f} | {np.mean(g('attack_recall')):.2%} | "
                f"{np.mean(g('binary_fpr')):.2%} | {icon} `{fit.upper()}` |"
            )
        absent = sorted(
            set(s0["classes"]) - set(s0.get("present_classes", s0["classes"]))
        )
        tests = [s["models"]["lgbm"]["test"] for s in seeds]
        L += [
            "",
            "_Fit notes: macro-F1 counts only the classes present in this capture"
            + (f" ({', '.join(absent)} do not occur in it)" if absent else "")
            + ". Tree models (Random Forest, and ensembles that include it) fit their "
            "training rows almost perfectly by design, so their train-test gap is "
            "expected; the check that matters is validation ≈ test, i.e. no drop "
            "between two unseen sets."
            + (
                f" LightGBM collapsed on one seed ({min(tests):.3f}; class-weight "
                "instability) and was not chosen by validation there."
                if max(tests) - min(tests) > 0.1
                else ""
            )
            + "_",
        ]
        weak = sorted(
            s0["models"]["ensemble"]["per_class_f1"].items(), key=lambda kv: kv[1]
        )[:3]
        L += ["", "Weakest classes (ensemble, test F1): "
              + ", ".join(f"{c} {v:.2f}" for c, v in weak) + ".",
              f"MLP fit check: `{s0['mlp_fitcheck']['verdict'].upper()}` "
              + ("— " + "; ".join(s0["mlp_fitcheck"]["findings"]) if s0["mlp_fitcheck"]["findings"] else ""),
              "", f"![fit](../figures/extra_fit_{d}.png) "
              f"![confusion](../figures/extra_confusion_{d}.png) "
              f"![per-class](../figures/extra_perclass_{d}.png)", ""]  # fmt: skip
    if logs:
        L += ["## 🪟 Log → ATT&CK mapping on real attack logs", "",
              "Fair test: new rules were written from a fixed **development half** of "
              "EVTX-ATTACK-SAMPLES only; scores below are on the **held-out** files: the "
              "other EVTX half plus every OTRF Security-Datasets Windows recording (a "
              "different lab, labelled with ATT&CK techniques). Sysmon's RuleName field "
              "(the sample author's own labels) is ignored.", "",
              "| Rules | Held-out files | Mapped to any technique | Right tactic | "
              "Right technique (OTRF) | Right tactic: EVTX / OTRF |",
              "|---|---|---|---|---|---|"]  # fmt: skip
        for tag, sm in logs.get("runs", {}).items():
            h, e, o = sm["heldout"], sm["heldout_evtx"], sm["heldout_otrf"]
            tech = h["correct_technique_rate"]
            L.append(f"| {tag} | {h['files']} | {h['detected_rate']:.0%} | "
                     f"**{h['correct_tactic_rate']:.0%}** | "
                     f"{'-' if tech is None else f'{tech:.0%}'} | "
                     f"{e['correct_tactic_rate']:.0%} / {o['correct_tactic_rate']:.0%} |")  # fmt: skip
        L += ["", "Per tactic (held-out, current rules):", "",
              "| Tactic | Files | Mapped | Right tactic |", "|---|---|---|---|"]  # fmt: skip
        for t, p in sorted(logs["per_tactic"].items(), key=lambda kv: -kv[1]["files"]):
            L.append(f"| {t} | {p['files']} | {p['detected'] / p['files']:.0%} | "
                     f"{p['correct'] / p['files']:.0%} |")  # fmt: skip
        L += ["", "Most mapped techniques (held-out): "
              + ", ".join(f"{t} ({n})" for t, n in logs["top_techniques"]) + ".",
              "", "![log mapping](../figures/extra_logmap.png)", ""]  # fmt: skip
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of datasets and/or 'logs'")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument(
        "--logs-tag", default="current", help="name of this log-mapping run"
    )
    ap.add_argument(
        "--workers", type=int, default=1, help="parallel log-scoring processes"
    )
    a = ap.parse_args()
    only = [x for x in a.only.split(",") if x]
    seeds = [int(s) for s in a.seeds.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    store = OUT / "extra_datasets.json"
    results = json.loads(store.read_text("utf-8")) if store.is_file() else {}
    logs_path = OUT / "log_mapping.json"
    from src.data.sources import get_spec, load_raw

    for d in DATASETS:
        if only and d not in only:
            continue
        done = {s["seed"] for s in results.get(d, {}).get("seeds", [])}
        todo = [s for s in seeds if s not in done]
        if not todo:
            continue
        raw = load_raw(get_spec(d))
        for seed in todo:
            _log.info("extra %s seed %d", d, seed)
            r = run_seed(d, raw, seed)
            results.setdefault(d, {"seeds": []})["seeds"].append(r)
            results[d]["seeds"].sort(key=lambda s: s["seed"])
            store.write_text(json.dumps(results, indent=1), "utf-8")  # resumable
            print(f"{d} seed {seed}: " + ", ".join(
                f"{m} {r['models'][m]['test']:.3f}" for m in MODELS), flush=True)  # fmt: skip
        del raw
    logs = json.loads(logs_path.read_text("utf-8")) if logs_path.is_file() else None
    if not only or "logs" in only:
        logs = run_logs(a.logs_tag, a.workers)
        logs_path.write_text(json.dumps(logs, indent=1), "utf-8")
        print(f"logs: mapped {logs['detected_rate']:.0%}, right tactic "
              f"{logs['correct_tactic_rate']:.0%}", flush=True)  # fmt: skip
    figures(results, logs)
    (OUT / "EXTRA_DATASETS.md").write_text(report(results, logs), "utf-8")
    print(f"report: {OUT / 'EXTRA_DATASETS.md'}")


if __name__ == "__main__":
    main()
