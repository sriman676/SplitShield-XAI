"""Train + evaluate the insider-threat (UEBA) area on CMU CERT r4.2 (owner 2026-10-15).

Run from the repo root:  ``python -m scripts.train_insider [--stages ...]``

Resumable stages (each skips work whose output already exists):
  1. ``download``    - r4.2 + answers from figshare (no sign-up), parallel HTTP Range
  2. ``features``    - stream the logs into one row per user-day (parquet cache)
  3. ``experiments`` - time split and user split, 3 seeds, LightGBM / IsolationForest
                       / logistic regression, plus an ablation without the HR flag
  4. ``deploy``      - freeze the champion (chosen on validation) into the model store
  5. ``report``      - ``results/extra/insider.{json,md}`` + ``results/figures/mlarea_insider_*.png``

CERT r4.1 is the blind set: it is never downloaded or used here.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import threading
import time
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.mlareas import insider as ins
from src.mlareas import insider_features as feats

MAIN = ins.main_project_root()
RAW = MAIN / "data" / "raw" / "cert"
RELEASE_DIR = RAW / "r4.2"
ANSWERS_DIR = RAW / "answers"
CACHE = MAIN / "data" / "processed" / "cert_r42"
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "extra"
RUN_VERSION = f"{feats.URL_RULES_VERSION}b"  # bump to re-run experiments
RUNS = OUT / f"insider_runs_{RUN_VERSION}"
DEPLOY_INFO = OUT / f"insider_deploy_{RUN_VERSION}.json"
FIGS = REPO / "results" / "figures"
SEEDS = (0, 1, 2)
URLS = {
    "r4.2.tar.bz2": "https://ndownloader.figshare.com/files/24856766",
    "answers.tar.bz2": "https://ndownloader.figshare.com/files/24857828",
}
STAGES = ("download", "features", "experiments", "deploy", "report")


def log(msg: str) -> None:
    print(f"[insider {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_ram_gb() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().available) / 1e9
    except ImportError:
        return float("nan")


# ----------------------------------------------------------------- download


def _get(url: str, rng: str | None = None) -> Any:
    headers = {"User-Agent": "Mozilla/5.0"}
    if rng:
        headers["Range"] = rng
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=60
    )


def parallel_download(
    url: str, out: Path, threads: int = 12, chunk: int = 32 << 20
) -> None:
    """Resumable parallel Range download (each chunk re-follows the signed redirect)."""
    if out.exists():
        return
    with _get(url, "bytes=0-0") as r:
        total = int(r.headers["Content-Range"].split("/")[1])
    parts = out.with_suffix(out.suffix + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    n = (total + chunk - 1) // chunk
    lock, done = threading.Lock(), [0]

    def work(i: int) -> None:
        s, e = i * chunk, min(total, (i + 1) * chunk) - 1
        p = parts / f"{i:05d}"
        for _ in range(30):
            if p.exists() and p.stat().st_size == e - s + 1:
                break
            try:
                with (
                    _get(url, f"bytes={s}-{e}") as r,
                    open(p.with_suffix(".tmp"), "wb") as f,
                ):
                    while b := r.read(1 << 20):
                        f.write(b)
                p.with_suffix(".tmp").replace(p)
            except OSError:
                time.sleep(3)
        with lock:
            done[0] += 1
            if done[0] % 10 == 0:
                log(f"{out.name}: {done[0]}/{n} chunks")

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(work, range(n)))
    with open(out, "wb") as f:
        for i in range(n):
            f.write((parts / f"{i:05d}").read_bytes())
    if out.stat().st_size != total:
        raise OSError(f"size mismatch for {out}")
    for i in range(n):
        (parts / f"{i:05d}").unlink()  # our own temporary chunks
    parts.rmdir()


def stage_download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        target = RELEASE_DIR if name.startswith("r4.2") else ANSWERS_DIR
        if target.exists():
            continue
        archive = RAW / name
        log(f"downloading {name}")
        parallel_download(url, archive)
        log(f"extracting {name}")
        with tarfile.open(archive, "r:bz2") as tf:
            tf.extractall(RAW, filter="data")


# ----------------------------------------------------------------- features


def stage_features() -> pd.DataFrame:
    path = CACHE / f"features_{feats.URL_RULES_VERSION}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    log(f"building user-day features (free RAM {free_ram_gb():.1f} GB)")
    df = feats.build_features(RELEASE_DIR, cache_dir=CACHE, log=log)
    mal = feats.malicious_days(ANSWERS_DIR, "4.2")
    df = feats.attach_labels(df, mal)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log(f"features: {len(df)} user-days, {int(df['label'].sum())} malicious")
    return df


# -------------------------------------------------------------- experiments


def _strip(res: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe copy of one experiment (no models / score arrays)."""
    fams = {
        f: {k: v for k, v in r.items() if k not in ("model", "test_scores")}
        for f, r in res["families"].items()
    }
    return {**{k: v for k, v in res.items() if k != "families"}, "families": fams}


def _splits(df: pd.DataFrame, kind: str, seed: int) -> pd.Series:
    if kind == "user":
        insiders = set(df.loc[df["insider"] == 1, "user"])
        return ins.user_split(df["user"], insiders, seed)
    return ins.time_split(df["day"])


def stage_experiments(df: pd.DataFrame) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    no_hr = [c for c in feats.feature_columns() if c != "hr_leaving_soon"]
    web_volume = {"http_n", "http_ah", "http_n_mean30", "http_n_z30"}
    no_web = [c for c in feats.feature_columns() if c not in web_volume]
    plan: list[tuple[str, str, int, Sequence[str] | None, Sequence[str]]] = []
    for seed in SEEDS:
        plan.append(("time", "time", seed, None, ins.FAMILIES))
        plan.append(("user", "user", seed, None, ins.FAMILIES))
        plan.append(("time_nohr", "time", seed, no_hr, ("lightgbm",)))
        plan.append(("time_noweb", "time", seed, no_web, ("lightgbm",)))
    for name, kind, seed, cols, fams in plan:
        out = RUNS / f"{name}_s{seed}.json"
        if out.exists():
            continue
        log(f"experiment {name} seed {seed} (free RAM {free_ram_gb():.1f} GB)")
        split = _splits(df, kind, seed)
        res = ins.run_experiment(df, split, seed, features=cols, families=fams)
        res["split_kind"] = kind
        res["split_sizes"] = {
            k: int((split == k).sum()) for k in ("train", "val", "test")
        }
        res["split_days"] = {
            k: [
                str(df.loc[split == k, "day"].min().date()),
                str(df.loc[split == k, "day"].max().date()),
            ]
            for k in ("train", "val", "test")
        }
        if seed == 0 and cols is None:
            test = df[split == "test"][["user", "day", "label", "insider"]].copy()
            for f, r in res["families"].items():
                test[f"s_{f}"] = r["test_scores"]
            test.to_parquet(RUNS / f"{name}_s0_test_scores.parquet", index=False)
        out.write_text(json.dumps(_strip(res), indent=2, default=float))


def _load_runs(name: str) -> list[dict[str, Any]]:
    return [json.loads((RUNS / f"{name}_s{s}.json").read_text()) for s in SEEDS]


# ------------------------------------------------------------------- deploy


def choose_family(runs: list[dict[str, Any]]) -> str:
    """Family with the best mean VALIDATION PR-AUC over seeds."""
    means = {
        f: float(np.nanmean([r["families"][f]["val"]["pr_auc"] for r in runs]))
        for f in runs[0]["families"]
    }
    return max(means, key=lambda f: means[f])


def stage_deploy(df: pd.DataFrame) -> dict[str, Any]:
    info_path = DEPLOY_INFO
    if info_path.exists() and (ins.area_dir() / "current" / "model.joblib").exists():
        loaded: dict[str, Any] = json.loads(info_path.read_text())
        return loaded
    runs = _load_runs("time")
    fam = choose_family(runs)
    params = runs[0]["families"][fam]["params"]
    cols = feats.feature_columns()
    split = ins.time_split(df["day"])
    tr, va, te = (df[split == k] for k in ("train", "val", "test"))
    X_tr, X_va = tr[cols].to_numpy(np.float32), va[cols].to_numpy(np.float32)
    model = ins.make_model(fam, 0, params)
    if fam == "isoforest":
        model.fit(X_tr)
    else:
        model.fit(X_tr, tr["label"].to_numpy())
    bundle = ins.make_bundle(
        fam, model, params, cols, X_va, va["label"].to_numpy(), X_tr
    )
    ins.save_bundle(bundle)
    ins.save_store_data(tr[[*cols, "label"]], va[[*cols, "label"]])
    # single-call latency and three example explanations from the test window
    rows = te[te["label"] == 1].head(3)
    if rows.empty:
        rows = te.head(3)
    examples = []
    ins.score_user_day(
        {c: float(rows.iloc[0][c]) for c in cols}, bundle=bundle
    )  # warm-up
    t0 = time.perf_counter()
    for _, r in rows.iterrows():
        out = ins.score_user_day({c: float(r[c]) for c in cols}, bundle=bundle)
        examples.append({"user": r["user"], "day": str(r["day"].date()), **out})
    latency_ms = (time.perf_counter() - t0) * 1000 / max(len(rows), 1)
    info = {
        "family": fam,
        "params": params,
        "threshold": bundle["threshold"],
        "val_pr_auc": bundle["val_pr_auc"],
        "rows_per_s": bundle["rows_per_s"],
        "score_user_day_ms": round(latency_ms, 2),
        "model_path": str(ins.area_dir() / "current" / "model.joblib"),
        "model_bytes": int(
            (ins.area_dir() / "current" / "model.joblib").stat().st_size
        ),
        "examples": examples,
    }
    info_path.write_text(json.dumps(info, indent=2, default=str))
    joblib.dump(bundle["features"], CACHE / "deployed_features.joblib")
    return info


# ------------------------------------------------------------------- report


def _agg(
    runs: list[dict[str, Any]], fam: str, split: str, key: str
) -> dict[str, float]:
    return ins.mean_sd([r["families"][fam][split][key] for r in runs])


def _fmt(d: dict[str, float]) -> str:
    return f"{d['mean']:.3f} ± {d['sd']:.3f}"


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "alerts_per_day",
        "recall_at_budget",
    )
    out: dict[str, Any] = {}
    for fam in runs[0]["families"]:
        out[fam] = {
            sp: {k: _agg(runs, fam, sp, k) for k in keys}
            for sp in ("train", "val", "test")
        }
        tr, va = out[fam]["train"]["pr_auc"]["mean"], out[fam]["val"]["pr_auc"]["mean"]
        out[fam]["fit"] = ins.fit_verdict(tr, va, fam in ins.TREE_FAMILIES)
        out[fam]["rows_per_s"] = ins.mean_sd(
            [r["families"][fam]["rows_per_s"] for r in runs]
        )
        out[fam]["selected_on_val_count"] = sum(r["selected"] == fam for r in runs)
    return out


def _user_table(
    runs: list[dict[str, Any]], fam: str, key: str = "users_test"
) -> list[dict[str, Any]]:
    """Per-insider detection across seeds."""
    rows: dict[str, dict[str, Any]] = {}
    for r in runs:
        for u in r["families"][fam][key]:
            d = rows.setdefault(
                u["user"],
                {"user": u["user"], "first_malicious_day": u["first_malicious_day"],
                 "malicious_days": u["malicious_days"], "detected": [], "delay": [],
                 "first_alert_delta": [], "false_alert_days": []},
            )  # fmt: skip
            d["detected"].append(u["detected"])
            d["delay"].append(u["delay_days"])
            d["first_alert_delta"].append(u["first_alert_vs_first_malicious_days"])
            d["false_alert_days"].append(u["false_alert_days"])
    return sorted(rows.values(), key=lambda d: d["first_malicious_day"])


def _scenario_map() -> dict[str, int]:
    insiders = feats.load_insiders(ANSWERS_DIR, "4.2")
    return dict(zip(insiders["user"], insiders["scenario"], strict=True))


def make_figures(fam: str, summ: dict[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    FIGS.mkdir(parents=True, exist_ok=True)
    made = []
    colors = {"lightgbm": "#2a6fdb", "isoforest": "#e07b39", "logreg": "#3a9d5d"}
    sc = pd.read_parquet(RUNS / "time_s0_test_scores.parquet")
    # 1. PR curve
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for f in ins.FAMILIES:
        if sc["label"].sum() == 0:
            break
        p, r, _ = precision_recall_curve(sc["label"], sc[f"s_{f}"])
        ap = summ["time"][f]["test"]["pr_auc"]["mean"]
        ax.plot(r, p, label=f"{f} (test PR-AUC {ap:.3f})", color=colors[f])
    ax.axhline(sc["label"].mean(), ls=":", color="grey", label="chance")
    ax.set(
        xlabel="recall (malicious user-days)",
        ylabel="precision",
        title="Insider threat - time split, test (seed 0)",
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    made.append(_save(fig, "pr_curve"))
    # 2. fit bars
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.25
    for i, sp in enumerate(("train", "val", "test")):
        vals = [summ["time"][f][sp]["pr_auc"]["mean"] for f in ins.FAMILIES]
        errs = [summ["time"][f][sp]["pr_auc"]["sd"] for f in ins.FAMILIES]
        ax.bar(
            np.arange(3) + (i - 1) * w,
            vals,
            w,
            yerr=errs,
            label=sp,
            color=["#9bb8e8", "#2a6fdb", "#123d82"][i],
        )
    ax.set_xticks(np.arange(3), list(ins.FAMILIES))
    ax.set(
        ylabel="PR-AUC (malicious user-days)",
        title="Fit check - time split, mean ± sd over 3 seeds",
        ylim=(0, 1.05),
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    made.append(_save(fig, "fit_bars"))
    # 3. risk timeline of three test-window insiders
    bundle = ins.load_bundle()
    users = list(sc.loc[sc["label"] == 1, "user"].value_counts().index[:3])
    if users:
        df = pd.read_parquet(CACHE / f"features_{feats.URL_RULES_VERSION}.parquet")
        fig, axes = plt.subplots(
            len(users), 1, figsize=(8, 2.3 * len(users)), sharex=True
        )
        axes = np.atleast_1d(axes)
        for ax, u in zip(axes, users, strict=True):
            g = df[df["user"] == u].sort_values("day")
            s = ins.model_scores(
                bundle["family"],
                bundle["model"],
                g[bundle["features"]].to_numpy(np.float32),
            )
            ax.plot(g["day"], s, lw=0.8, color="#2a6fdb")
            m = g["label"] == 1
            ax.scatter(
                g.loc[m, "day"],
                s[m.to_numpy()],
                color="#d62728",
                s=14,
                zorder=3,
                label="malicious day",
            )
            ax.axhline(
                bundle["threshold"],
                ls="--",
                color="grey",
                lw=0.8,
                label="alert threshold",
            )
            ax.set_title(f"{u} (scenario {_scenario_map().get(u, '?')})", fontsize=9)
            ax.set_ylabel("risk")
        axes[0].legend(frameon=False, fontsize=7, loc="upper left")
        fig.tight_layout()
        made.append(_save(fig, "risk_timeline"))
    # 4. feature importance
    if bundle["family"] == "lightgbm":
        imp = pd.Series(
            bundle["model"].booster_.feature_importance("gain"),
            index=bundle["features"],
        )
        title = "LightGBM gain importance (deployed model)"
    elif bundle["family"] == "logreg":
        imp = pd.Series(np.abs(bundle["model"][-1].coef_[0]), index=bundle["features"])
        title = "|logistic coefficient| (deployed model)"
    else:
        imp = pd.Series(dtype=float)
        title = ""
    if not imp.empty:
        top = imp.sort_values().tail(20)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.barh(top.index, top.to_numpy() / top.max(), color="#2a6fdb")
        ax.set(xlabel="relative importance", title=title)
        fig.tight_layout()
        made.append(_save(fig, "feature_importance"))
    return made


def _save(fig: Any, name: str) -> str:
    import matplotlib.pyplot as plt

    path = FIGS / f"mlarea_insider_{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path.relative_to(REPO))


def stage_report(df: pd.DataFrame, deploy: dict[str, Any]) -> None:
    runs = {k: _load_runs(k) for k in ("time", "user", "time_nohr", "time_noweb")}
    summ = {k: _summary(v) for k, v in runs.items()}
    fam = deploy["family"]
    scen = _scenario_map()
    first_mal = {
        u: str(g.min().date())
        for u, g in df.loc[df["label"] == 1].groupby("user")["day"]
    }
    users = {
        "time_test": _user_table(runs["time"], fam),
        "user_test": _user_table(runs["user"], fam),
        "time_val": _user_table(runs["time"], fam, "users_val"),
    }
    for tab in users.values():
        for d in tab:
            d["scenario"] = scen.get(d["user"])
            d["attack_start"] = first_mal.get(d["user"])
    figs = make_figures(fam, summ)
    data = {
        "release": "CMU CERT r4.2",
        "user_days": int(len(df)),
        "users": int(df["user"].nunique()),
        "days": int(df["day"].nunique()),
        "malicious_user_days": int(df["label"].sum()),
        "insiders": int(df.loc[df["insider"] == 1, "user"].nunique()),
        "features": len(feats.feature_columns()),
        "split_days_time": runs["time"][0]["split_days"],
        "split_sizes_time": runs["time"][0]["split_sizes"],
        "split_pos_time": {
            k: runs["time"][0]["families"][fam][k]["n_pos"]
            for k in ("train", "val", "test")
        },
        "split_pos_user": {
            f"seed{r['seed']}": {
                k: r["families"][fam][k]["n_pos"] for k in ("train", "val", "test")
            }
            for r in runs["user"]
        },
    }
    result = {
        "area": "insider",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": data,
        "summary": summ,
        "deployed": deploy,
        "users": users,
        "figures": figs,
        "seeds": list(SEEDS),
        "selection": "family, hyper-parameters, early stopping and threshold chosen on validation only",
        "blind_test": "CERT r4.1 reserved; blind_test() implemented, not run",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "insider.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    (OUT / "insider.md").write_text(render_md(result), encoding="utf-8")
    log(f"wrote {OUT / 'insider.json'} and insider.md")


def _det_cell(d: dict[str, Any]) -> tuple[str, str, str]:
    det = sum(d["detected"])
    delays = [x for x in d["delay"] if x is not None]
    first = [x for x in d["first_alert_delta"] if x is not None]
    delay = f"{np.median(delays):+.0f} d" if delays else "-"
    fa = f"{np.median(first):+.0f} d" if first else "-"
    return f"{det}/{len(d['detected'])}", delay, fa


def render_md(res: dict[str, Any]) -> str:
    d, summ, dep = res["data"], res["summary"], res["deployed"]
    fam = dep["family"]
    L = ["# Insider threat / UEBA - CMU CERT r4.2", ""]
    L += ["## 📦 Data", ""]
    L.append(
        f"- {d['user_days']:,} user-days ({d['users']} users x {d['days']} days with activity), "
        f"{d['malicious_user_days']} malicious user-days from {d['insiders']} insiders "
        f"({d['malicious_user_days'] / d['user_days']:.3%} positive)."
    )
    L.append(f"- {d['features']} features per user-day: logon, USB, file copies to removable media, email, web categories, "
             "psychometric (OCEAN), LDAP role/department/HR-leaving flag, after-hours/weekend, past-30-day baselines "
             "(mean + z, past only), role-peer z-scores, 7-day activity counts.")  # fmt: skip
    sd = d["split_days_time"]
    L.append(f"- Time split: train {sd['train'][0]}..{sd['train'][1]}, val {sd['val'][0]}..{sd['val'][1]}, "
             f"test {sd['test'][0]}..{sd['test'][1]}; malicious user-days {d['split_pos_time']}.")  # fmt: skip
    L.append("- User split: 70/15/15 of users (insiders stratified), test insiders never seen in training; "
             f"malicious user-days per seed {d['split_pos_user']}.")  # fmt: skip
    L.append(
        "- Model, hyper-parameters, early stopping and alert threshold chosen on VALIDATION only; test reported. 3 seeds (0, 1, 2), mean ± sd."
    )
    L.append("")
    for kind, title in (
        ("time", "time split (main, honest for UEBA)"),
        ("user", "user split (harder: unseen insiders)"),
        ("time_nohr", "time split, ablation without the HR-leaving flag"),
        (
            "time_noweb",
            "time split, ablation without total web-volume features "
            "(http_n, http_ah and their baselines)",
        ),
    ):
        L += [f"## 📊 Train / val / test - {title}", ""]
        L.append(
            "| model | split | precision | recall | F1 | PR-AUC | ROC-AUC | alerts/day | recall@top-5/day |"
        )
        L.append("|---|---|---|---|---|---|---|---|---|")
        for f, s in summ[kind].items():
            for sp in ("train", "val", "test"):
                m = s[sp]
                L.append(
                    f"| {f} | {sp} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | "
                    f"{_fmt(m['pr_auc'])} | {_fmt(m['roc_auc'])} | {m['alerts_per_day']['mean']:.2f} | "
                    f"{_fmt(m['recall_at_budget'])} |"
                )
        L.append("")
        L.append(
            "| model | fit (PR-AUC train vs val) | verdict | selected on val (seeds) | rows/s |"
        )
        L.append("|---|---|---|---|---|")
        for f, s in summ[kind].items():
            v = s["fit"]
            icon = "✅" if v["verdict"] == "good fit" else "⚠️"
            note = f" - {v['note']}" if v["note"] else ""
            L.append(f"| {f} | {v['train']:.3f} vs {v['val']:.3f} (gap {v['gap']:+.3f}) | {icon} {v['verdict']}{note} | "
                     f"{s['selected_on_val_count']}/3 | {s['rows_per_s']['mean']:,.0f} |")  # fmt: skip
        L.append("")
    L += ["## ✅/⚠️ Fit verdict", ""]
    for kind in ("time", "user"):
        v = summ[kind][fam]["fit"]
        icon = "✅" if v["verdict"] == "good fit" else "⚠️"
        L.append(
            f"- {icon} {kind} split, deployed family **{fam}**: {v['verdict']} (train {v['train']:.3f}, val {v['val']:.3f}, gap {v['gap']:+.3f}). {v['note']}"
        )
    L.append(
        "- Tree models (LightGBM, IsolationForest) fit train near-perfectly by design; the useful check is val vs test."
    )
    L.append("")
    L += ["## 🚀 Deployed model", ""]
    L.append(f"- Family **{fam}** (best mean validation PR-AUC on the time split), params {dep['params']}, "
             f"threshold {dep['threshold']:.4f} (max F1 on validation), val PR-AUC {dep['val_pr_auc']:.3f}.")  # fmt: skip
    L.append(
        f"- Speed: {dep['rows_per_s']:,.0f} user-days/s batch; `score_user_day` incl. explanation {dep['score_user_day_ms']:.1f} ms/call; model {dep['model_bytes'] / 1e6:.2f} MB."
    )
    L.append("")
    if dep.get("examples"):
        ex = dep["examples"][0]
        L += [
            f"Example explanation ({ex['user']}, {ex['day']}, risk {ex['risk']}, verdict {ex['verdict']}):",
            "",
        ]
        L += [f"    {w}" for w in ex["why"]]
        L.append("")
    for key, title in (
        ("time_test", "time split, test window"),
        ("time_val", "time split, validation window"),
        ("user_test", "user split, unseen test insiders"),
    ):
        L += [f"## 👤 Per-insider detection - {title} ({fam})", ""]
        L.append("Detected = at least one malicious day alerted (x/3 seeds). Delay = first alerted malicious day minus first "
                 "malicious day in the window (median). First alert = first alert of any kind for that user minus first "
                 "malicious day (negative = alerted earlier).")  # fmt: skip
        L.append("")
        L.append(
            "| insider | scenario | attack start (whole data) | first malicious day in window | malicious days in window | detected | delay | first alert | false-alert days (mean) |"
        )
        L.append("|---|---|---|---|---|---|---|---|---|")
        tab = res["users"][key]
        for u in tab:
            det, delay, fa = _det_cell(u)
            L.append(
                f"| {u['user']} | {u['scenario']} | {u.get('attack_start', '-')} | {u['first_malicious_day']} | {u['malicious_days']} | {det} | {delay} | {fa} | {np.mean(u['false_alert_days']):.1f} |"
            )
        n_det = sum(sum(u["detected"]) for u in tab)
        n_all = sum(len(u["detected"]) for u in tab)
        L.append("")
        L.append(
            f"Insider-level detection: {n_det}/{n_all} insider-seed pairs ({n_det / max(n_all, 1):.0%})."
        )
        L.append("")
    L += ["## 🖼️ Figures", ""] + [f"- `{f}`" for f in res["figures"]] + [""]
    L += ["## ⚠️ Limits", ""]
    L += [
        "- CERT r4.2 is synthetic; real organisations are noisier. Results do not transfer automatically.",
        "- Very few positives: malicious user-days are ~0.1-0.3% of rows, and the time-split test window holds only the insiders active in the last 15% of days, so test numbers swing a lot between windows.",
        "- r4.2 http.csv has no upload/download activity column; uploads are approximated by visits to WikiLeaks / cloud-sharing domains. URL categories are keyword rules.",
        "- `hr_leaving_soon` assumes HR knows about a departure about one month ahead (built from the next monthly LDAP snapshot); the ablation above shows the result without it.",
        "- Peer-group z-scores use same-day role peers, which is only available at the end of the day; the model is an end-of-day scorer, not real-time.",
        "- IsolationForest is unsupervised (labels unused); its threshold is still set on validation labels.",
        "- r4.2 is a 'dense needle' release: red-team events are injected on top of normal behaviour, so attack days also show more total web requests than the user's own past (`http_n_z30` is the 2nd most important feature). That is partly a simulation artefact; the web-volume ablation above shows the score without it. Expect lower numbers on real logs.",
        "- In the time split, 7 of the 8 test-window insiders started their attack in the validation window (attacks continue across the boundary); only JLM0364 starts inside test. Detection delay is therefore measured from the first malicious day inside the window, and the attack-start column gives the true start.",
        "- ROC-AUC shows 1.000 after rounding because the benign class is huge (~45k user-days per window); PR-AUC, precision and alerts/day are the meaningful numbers.",
        "- Logistic regression is deterministic (sd 0 across seeds on the time split).",
        "- Job-search / cloud URL rules match the host only; this was tightened once after a check on a slice of http.csv showed the word 'job' in random page paths (not tuned on test).",
        "- CERT r4.1 is the reserved blind set; `blind_test()` is implemented with the frozen model + threshold but has NOT been run.",
    ]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    args = ap.parse_args(argv)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    if "download" in args.stages:
        stage_download()
    df = stage_features() if set(args.stages) - {"download"} else pd.DataFrame()
    if "experiments" in args.stages:
        stage_experiments(df)
    deploy: dict[str, Any] = {}
    if "deploy" in args.stages:
        deploy = stage_deploy(df)
    if "report" in args.stages:
        if not deploy:
            deploy = json.loads(DEPLOY_INFO.read_text())
        stage_report(df, deploy)


if __name__ == "__main__":
    main()
