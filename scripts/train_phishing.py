"""Train + report the phishing URL detector (owner 2026-10-15).

Run: ``python -m scripts.train_phishing [--quick]``

Writes ``results/extra/phishing.json`` + ``.md`` and
``results/figures/mlarea_phishing_*.png``; deploys the URL-only (hostname) model to
``artifacts/mlareas/phishing/current`` (previous model kept as ``prev``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)

from src.mlareas import phishing  # noqa: E402

OUT = Path("results/extra")
FIG = Path("results/figures")


def _fmt(d: dict[str, float]) -> str:
    return f"{d['mean']:.4f} ± {d['sd']:.4f}"


def make_figures(plot: dict[str, Any]) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    paths = []
    soc = plot["url_host"]
    y, p = soc["y_test"], soc["p_test"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y,
        (p >= 0.5).astype(int),
        display_labels=["legit", "phishing"],
        ax=ax,
        colorbar=False,
    )
    ax.set_title("URL-only (hostname) model, test, seed 0")
    paths.append(_save(fig, "confusion"))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for variant, d in plot.items():
        fpr, tpr, _ = roc_curve(d["y_test"], d["p_test"])
        pr, rc, _ = precision_recall_curve(d["y_test"], d["p_test"])
        axes[0].plot(fpr, tpr, label=variant)
        axes[1].plot(rc, pr, label=variant)
    axes[0].set(
        xlabel="false-positive rate", ylabel="true-positive rate", title="ROC (test)"
    )
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall (test)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    paths.append(_save(fig, "roc_pr"))

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.25
    names = list(plot)
    for k, split in enumerate(("train", "val", "test")):
        means = [plot[v]["fit"][split]["f1"]["mean"] for v in names]
        sds = [plot[v]["fit"][split]["f1"]["sd"] for v in names]
        ax.bar(
            np.arange(len(names)) + (k - 1) * width, means, width, yerr=sds, label=split
        )
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylim(0.5, 1.01)
    ax.set_ylabel("F1 (phishing), mean ± sd over 3 seeds")
    ax.set_title("Fit check: selected model per variant")
    ax.legend()
    paths.append(_save(fig, "fit_bars"))
    return paths


def _save(fig: Any, name: str) -> str:
    path = FIG / f"mlarea_phishing_{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def markdown(res: dict[str, Any]) -> str:
    lines = ["# 🎣 Phishing URL detection (PhiUSIIL)", "",
             "## 📦 Data", "",
             f"- {res['dataset']}; positive class = phishing; {res['split']}; "
             f"seeds {res['seeds']}; {res['selection']}.",
             ]  # fmt: skip
    for v, vr in res["variants"].items():
        lines.append(f"- `{v}`: {vr['data']}, {vr['n_features']} features")
    lines += ["", "## 📊 Results (mean ± sd over 3 seeds)", "",
              "| variant | model | split | precision | recall | F1 | ROC-AUC | FPR legit |",
              "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for v, vr in res["variants"].items():
        for fam, fr in vr["families"].items():
            tag = f"**{fam}** (selected)" if fam == vr["selected"] else fam
            for split in ("train", "val", "test"):
                m = fr[split]
                lines.append(
                    f"| {v} | {tag} | {split} | {_fmt(m['precision'])} | {_fmt(m['recall'])} "
                    f"| {_fmt(m['f1'])} | {_fmt(m['roc_auc'])} | {_fmt(m['fpr_legit'])} |"
                )
    lines += ["", "## ✅/⚠️ Fit verdict (F1 train vs val, gap rule 0.05)", ""]
    for v, vr in res["variants"].items():
        for fam, fr in vr["families"].items():
            f = fr["fit"]
            icon = "✅" if f["verdict"] == "good fit" else "⚠️"
            note = f" ({f['note']})" if f["note"] else ""
            lines.append(f"- {icon} {v} / {fam}: train {f['train']} vs val {f['val']} "
                         f"(gap {f['gap']}) -> {f['verdict']}{note}")  # fmt: skip
    soc = res["variants"]["url_host"]
    cal = soc["selected_calibrated_test_seed0"]
    lines += ["", "## 🛡️ Deployed SOC model", "",
              f"- URL-only hostname model, `{soc['selected']}`, isotonic-calibrated on val.",
              f"- Test (seed 0, calibrated): precision {cal['precision']:.4f}, recall "
              f"{cal['recall']:.4f}, F1 {cal['f1']:.4f}, ROC-AUC {cal['roc_auc']:.4f}, "
              f"FPR on legit {cal['fpr_legit']:.4f}.",
              f"- Speed: {soc['speed_score_only_urls_per_s']} URLs/s scoring only; "
              f"{soc['speed_urls_per_s']} URLs/s with the full step-by-step explanation.",
              "", "## 🔍 Leakage and limits", ""]  # fmt: skip
    leak = res["leakage"]
    art = leak["artifacts"]
    lines += [
        f"- Every legitimate URL is a bare `https://www.<domain>` homepage: "
        f"`http://` URLs are {art['http_scheme']['phishing_rate']:.0%} phishing "
        f"(n={art['http_scheme']['n']}), URLs with any path/query are "
        f"{art['url_has_path_or_query']['phishing_rate']:.0%} phishing "
        f"(n={art['url_has_path_or_query']['n']}); legit URLs with a path: "
        f"{art['legit_urls_with_path']}.",
        "- So `url_full` scores are inflated by URL *shape*, not phishing semantics. The "
        "deployed model sees only the hostname (scheme, www, path removed) and is "
        "de-duplicated by host, which is the honest number to quote.",
        f"- Dataset features with single-feature AUC > 0.95: "
        f"{', '.join(leak['features_auc_over_0.95'])}. `URLSimilarityIndex` is derived "
        "with reference to legitimate URLs and page-content counts (LineOfCode, "
        "NoOfExternalRef...) need the page fetched; `all_features` is a reference only.",
        "- Legitimate side is homepages of popular domains; real enterprise traffic "
        "(deep links, CDNs, SaaS) will raise the false-positive rate. Re-validate on "
        "local proxy logs before trusting the FPR.",
        "- Known false positive from a manual spot check: `login.microsoftonline.com` "
        "(a real Microsoft login host) scores as phishing, because sub-domain login hosts "
        "of big brands never appear on the dataset's legitimate side. Pair the model "
        "with an allowlist of the organisation's own SaaS domains.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, 20k rows smoke run")
    args = ap.parse_args()
    seeds = (0,) if args.quick else (0, 1, 2)
    df = phishing.load_phiusiil()
    if args.quick:
        df = df.sample(20000, random_state=0)
    res = phishing.train(seeds=seeds, df=df, save=not args.quick)
    plot = res.pop("_plot")
    res["figures"] = make_figures(plot)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phishing.json").write_text(json.dumps(res, indent=2))
    (OUT / "phishing.md").write_text(markdown(res), encoding="utf-8")
    print(json.dumps({v: {"selected": r["selected"],
                          "test_f1": r["families"][r["selected"]]["test"]["f1"]}
                      for v, r in res["variants"].items()}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
