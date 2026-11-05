"""Train + report the OT/ICS anomaly detector on HAI 22.04 (owner 2026-10-15).

Run: ``python -m scripts.train_ics [--quick]``

Writes ``results/extra/ics.json`` + ``.md`` and ``results/figures/mlarea_ics_*.png``;
deploys the selected detector to ``artifacts/mlareas/ics/current`` (old one kept as
``prev``). HAI 21.03 stays untouched for the lead's blind final test.
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
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve  # noqa: E402

from src.mlareas import ics  # noqa: E402

OUT = Path("results/extra")
FIG = Path("results/figures")


def _fmt(d: dict[str, float] | None) -> str:
    return "n/a" if d is None else f"{d['mean']:.4f} ± {d['sd']:.4f}"


def _save(fig: Any, name: str) -> str:
    path = FIG / f"mlarea_ics_{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def make_figures(plot: dict[str, Any]) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    s, y, thr = plot["scores"], plot["labels"], plot["threshold"]
    alarm = (s > thr).astype(int)
    paths = []
    fig, axes = plt.subplots(
        len(plot["lengths"]), 1, figsize=(11, 2.2 * len(plot["lengths"]))
    )
    start = 0
    for k, (ax, n) in enumerate(zip(np.atleast_1d(axes), plot["lengths"], strict=True)):
        t = np.arange(n) / 3600
        ax.plot(t, s[start : start + n], lw=0.6, label="anomaly score")
        ax.axhline(thr, color="red", ls="--", lw=0.8, label="threshold (from val)")
        top = max(float(np.percentile(s[start : start + n], 99.9)), thr) * 1.5
        ax.fill_between(t, 0, top, where=y[start : start + n] > 0, color="orange",
                        alpha=0.3, label="labelled attack")  # fmt: skip
        ax.set_ylim(0, top)
        ax.set_ylabel(f"test{k + 1}")
        start += n
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("hours into file")
    paths.append(_save(fig, "timeline"))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y, alarm, display_labels=["normal", "attack"],
                                            ax=ax, colorbar=False)  # fmt: skip
    ax.set_title("Test, point-wise, seed 0")
    paths.append(_save(fig, "confusion"))

    fig, ax = plt.subplots(figsize=(5, 4))
    pr, rc, _ = precision_recall_curve(y, s)
    ax.plot(rc, pr)
    ax.set(xlabel="recall", ylabel="precision", title="Point-wise PR curve (test)")
    ax.grid(alpha=0.3)
    paths.append(_save(fig, "pr"))

    fig, ax = plt.subplots(figsize=(7, 4))
    fams = list(plot["fit"])
    for k, split in enumerate(("train", "val", "test")):
        means = [plot["fit"][f][split]["f1"]["mean"] for f in fams]
        sds = [plot["fit"][f][split]["f1"]["sd"] for f in fams]
        ax.bar(
            np.arange(len(fams)) + (k - 1) * 0.25, means, 0.25, yerr=sds, label=split
        )
    ax.set_xticks(np.arange(len(fams)), fams)
    ax.set_ylabel("F1 (train/val: injected faults; test: real attacks)")
    ax.set_title("Fit check, mean ± sd over 3 seeds")
    ax.legend()
    paths.append(_save(fig, "fit_bars"))
    return paths


def markdown(res: dict[str, Any]) -> str:
    lines = ["# 🏭 OT/ICS anomaly detection (HAI 22.04)", "", "## 📦 Data", "",
             f"- {res['dataset']}; {res['sensors_used']} non-constant sensors; rows "
             f"{res['rows']}; test attack rows {res['test_attack_rows']}.",
             "- Validation = last 15% of the normal period (chronological). No attacks "
             "exist there, so model choice and the threshold use a fixed copy with "
             "injected synthetic faults (offset / drift / freeze on 1-3 sensors).",
             f"- Blind hold-out: {res['blind_holdout_version']}.",
             f"- Selection: {res['selection']}. Seeds {res['seeds']}.",
             "", "## 📊 Results (mean ± sd over 3 seeds)", "",
             "| model | train F1 (inj.) | val F1 (inj.) | test precision | test recall "
             "| test F1 | episodes detected | median delay (s) | false alarms/day |",
             "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for fam, fr in res["families"].items():
        tag = f"**{fam}** (selected)" if fam == res["selected"] else fam
        te = fr["test"]
        lines.append(
            f"| {tag} | {_fmt(fr['train']['f1'])} | {_fmt(fr['val']['f1'])} | "
            f"{_fmt(te['precision'])} | {_fmt(te['recall'])} | {_fmt(te['f1'])} | "
            f"{_fmt(te['episodes_detected'])} / {int(te['episodes']['mean'])} | "
            f"{_fmt(te.get('delay_s_median'))} | {_fmt(te['false_alarms_per_day'])} |"
        )
    lines += [
        "",
        "## ✅/⚠️ Fit verdict (injected-fault F1, train slice vs val, gap 0.05)",
        "",
    ]
    for fam, fr in res["families"].items():
        f = fr["fit"]
        icon = "✅" if f["verdict"] == "good fit" else "⚠️"
        lines.append(f"- {icon} {fam}: train {f['train']} vs val {f['val']} (gap {f['gap']}) "
                     f"-> {f['verdict']}; clean-val false alarms/day "
                     f"{_fmt(fr['val'].get('clean_false_alarms_per_day'))}")  # fmt: skip
    lines += ["", "## ⚡ Speed", "",
              f"- Batch scoring: {res['speed_rows_per_s_batch']} rows (seconds of plant "
              f"data)/s; SOC `score_window` with explanation: "
              f"{res['speed_score_window_per_s']} windows/s.",
              "", "## 🔍 Notes and limits", "",
              "- Train/val F1 are on synthetic faults, test F1 on real HAI attacks; they "
              "measure different things, so the train-val gap is the fit check and test "
              "is the honest outcome.",
              "- Autoencoder/forecaster score = max over sensors of the normalised error, "
              "picked in a seed-0 pilot on injected-fault validation F1 (max beat mean and "
              "top-3 mean); thresholds come from clean-validation percentiles.",
              "- Point-wise F1 penalises late detection and alarms that outlast the label; "
              "the episode and delay columns are closer to how a SOC experiences it.",
              "- False alarms/day: alarm events (merged when < 60 s apart) starting outside "
              "attacks and outside 5 min after an attack, per day of normal test time.",
              "- ⚠️ The injected-fault F1 criterion picks low thresholds (95th-98th "
              "clean-val percentile), so false alarms/day are high. Not retuned after "
              "seeing test (that would leak); a validation false-alarm budget "
              "(e.g. <= 5/day on clean val) is the owner's call for the next run.",
              "- Absolute train/val F1 on synthetic faults is low by construction (freeze "
              "faults are near-undetectable), so 'underfitting' labels there mean "
              "'synthetic faults are hard', not that a bigger model is needed.",
              "- HAI is one testbed; a real plant needs its own normal baseline and threshold."]  # fmt: skip
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quick", action="store_true", help="1 seed, short training smoke run"
    )
    args = ap.parse_args()
    kw: dict[str, Any] = (
        {"seeds": (0,), "epochs": 1, "per_epoch": 20_000, "save": False}
        if args.quick
        else {}
    )
    res = ics.train(**kw)
    plot = res.pop("_plot")
    res["figures"] = make_figures(plot)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ics.json").write_text(json.dumps(res, indent=2))
    (OUT / "ics.md").write_text(markdown(res), encoding="utf-8")
    print(json.dumps({f: {"val_f1": r["val"]["f1"], "test_f1": r["test"]["f1"]}
                      for f, r in res["families"].items()}, indent=1))  # fmt: skip
    print("selected:", res["selected"])


if __name__ == "__main__":
    main()
