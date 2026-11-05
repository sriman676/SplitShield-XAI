"""E3 batch matrix driver — training scheme × label skew × seeds, resumable.

The research pipeline (``run_pipeline.py``) runs one config point per invocation.
E3 (06_EXPERIMENTS §2) needs a *matrix*: the four training arms × Dirichlet α ×
seeds, aggregated into macro-F1 vs α per arm (H4, figure F4). This driver sweeps
that matrix and is ``--resume``-safe: each cell writes a deterministic run dir
(``<exp>__<scheme>__a<α>__seed<n>__<confighash>``) plus a ``DONE`` marker, so a
re-run with ``--resume`` skips finished cells and only fills the gaps.

    python run_matrix.py --experiment E3_alpha_skew            # full matrix
    python run_matrix.py --experiment E3_alpha_skew --fast     # smoke on the fixture
    python run_matrix.py --experiment E3_alpha_skew --resume   # fill in missing cells

Data is preprocessed once per seed and reused across every (scheme, α) cell for
that seed — α changes only the Dirichlet partition *inside* the distributed arms,
not the split. ponytail: no cross-seed cache; add one only if real preprocessing
dominates wall-clock.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from config.base import Config, build_config, config_hash, load_experiment
from utils.logging_config import configure_logging, get_logger
from utils.manifest import write_manifest
from utils.paths import RESULTS_DIR, RUNS_DIR, run_dir
from utils.seed import set_seed

_log = get_logger(__name__)

_DEFAULT_SCHEMES = ["centralized", "split_sequential", "fedavg", "splitfed"]
_DEFAULT_ALPHAS = [10.0, 1.0, 0.1]


def _key(alpha: float) -> str:
    return str(alpha).replace(".", "p").replace("-", "m")


def _rid(cfg: Config, scheme: str, alpha: float) -> str:
    return f"{cfg.experiment_id}__{scheme}__a{_key(alpha)}__seed{cfg.seed}__{config_hash(cfg)}"


def _train_arm(scheme: str, data: Any, cfg: Config, device: str) -> float:
    """Train one arm and return its test macro-F1."""
    from src.training.centralized import train_centralized
    from src.training.fedavg import train_fedavg
    from src.training.split_sequential import train_split_sequential
    from src.training.splitfed import train_splitfed

    if scheme == "centralized":
        return float(train_centralized(data, cfg, device)[2]["macro_f1"])
    if scheme == "split_sequential":
        return float(train_split_sequential(data, cfg, device)[1]["test"]["macro_f1"])
    if scheme == "fedavg":
        return float(train_fedavg(data, cfg, device)[1]["test"]["macro_f1"])
    if scheme == "splitfed":
        return float(train_splitfed(data, cfg, device)[1]["test"]["macro_f1"])
    raise ValueError(f"unknown scheme {scheme!r}")


def _write_arm(rid: str, scheme: str, alpha: float, seed: int, f1: float) -> None:
    d = run_dir(rid)
    payload = {
        "run_id": rid,
        "metrics": {
            "macro_f1": {
                "value": f1,
                "scheme": scheme,
                "dirichlet_alpha": alpha,
                "seed": seed,
            }
        },
        "preliminary": True,
    }
    (d / "metrics" / "e3_arm.json").write_text(json.dumps(payload, indent=2), "utf-8")
    (d / "DONE").write_text("ok", "utf-8")


def _read_f1(rid: str) -> float:
    payload = json.loads(
        (RUNS_DIR / rid / "metrics" / "e3_arm.json").read_text("utf-8")
    )
    return float(payload["metrics"]["macro_f1"]["value"])


def _summarize(results: dict[tuple[str, float], list[float]]) -> dict[str, Any]:
    import numpy as np

    out: dict[str, Any] = {}
    for (scheme, alpha), vals in results.items():
        out[f"{scheme}|a{alpha}"] = {
            "scheme": scheme,
            "dirichlet_alpha": alpha,
            "macro_f1_mean": float(np.mean(vals)),
            "macro_f1_std": float(np.std(vals)),
            "n_seeds": len(vals),
        }
    return out


def _plot_f4(experiment: str, results: dict[tuple[str, float], list[float]]) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schemes = sorted({s for s, _ in results})
    fig, ax = plt.subplots(figsize=(7, 5))
    for scheme in schemes:
        pts = sorted(
            (a, np.mean(v), np.std(v)) for (s, a), v in results.items() if s == scheme
        )
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=scheme)
    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet α (label skew; smaller = more non-IID →)")
    ax.set_ylabel("test macro-F1")
    ax.set_title(f"F4 Training scheme × skew ({experiment})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "figures" / "F4_alpha_skew.png", dpi=140)
    plt.close(fig)


def run_matrix(
    experiment: str,
    fast: bool = False,
    resume: bool = False,
    matrix_override: dict[str, Any] | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Sweep the (scheme × α × seed) matrix; return run/skip counts and the summary."""
    from src.data.preprocess import preprocess_frame
    from src.data.sources import get_spec, load_raw
    from utils.device import get_primary_device

    raw = dict(load_experiment(f"{experiment}.yaml"))
    if dataset:
        raw["dataset"] = dataset
    mat = matrix_override or raw.get("matrix") or {}
    schemes = list(mat.get("scheme", _DEFAULT_SCHEMES))
    alphas = [float(a) for a in mat.get("dirichlet_alpha", _DEFAULT_ALPHAS)]
    seeds = list(mat.get("seeds", raw.get("seeds", [0, 1, 2, 3, 4])))
    if fast and "seeds" not in mat:
        seeds = seeds[:1]  # --fast smoke: one seed (preliminary, non-result)
    device = get_primary_device()

    results: dict[tuple[str, float], list[float]] = {}
    n_run = n_skip = 0

    for seed in seeds:
        plan = []  # cells that still need training for this seed
        for scheme in schemes:
            for alpha in alphas:
                cfg = build_config(
                    raw,
                    seed=seed,
                    fast=fast,
                    overrides={
                        "training": {"scheme": scheme, "dirichlet_alpha": alpha}
                    },
                )
                rid = _rid(cfg, scheme, alpha)
                done = RUNS_DIR / rid / "DONE"
                metrics = RUNS_DIR / rid / "metrics" / "e3_arm.json"
                if resume and done.exists() and metrics.exists():
                    results.setdefault((scheme, alpha), []).append(_read_f1(rid))
                    n_skip += 1
                else:
                    plan.append((scheme, alpha, cfg, rid))
        if not plan:
            continue

        set_seed(seed)
        try:
            spec = get_spec(raw["dataset"])
            data = preprocess_frame(load_raw(spec), spec, plan[0][2], seed=seed)
        except (FileNotFoundError, KeyError) as exc:
            _log.warning("E3 seed %d skipped (%s); dataset unavailable", seed, exc)
            continue

        for scheme, alpha, cfg, rid in plan:
            f1 = _train_arm(scheme, data, cfg, device)
            write_manifest(cfg, rid, {})
            _write_arm(rid, scheme, alpha, seed, f1)
            results.setdefault((scheme, alpha), []).append(f1)
            n_run += 1
            _log.info(
                "E3 %s alpha=%s seed=%d -> macro_f1=%.3f", scheme, alpha, seed, f1
            )

    summary = _summarize(results)
    if results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"{experiment}_summary.json").write_text(
            json.dumps(summary, indent=2), "utf-8"
        )
        _plot_f4(experiment, results)

    return {
        "experiment": experiment,
        "n_run": n_run,
        "n_skipped": n_skip,
        "n_points": len(schemes) * len(alphas) * len(seeds),
        "summary": summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="E3 scheme×skew matrix driver")
    ap.add_argument("--experiment", default="E3_alpha_skew")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dataset", default=None, help="override the YAML dataset")
    args = ap.parse_args()

    configure_logging()
    res = run_matrix(
        args.experiment, fast=args.fast, resume=args.resume, dataset=args.dataset
    )
    print(
        f"[matrix] {res['experiment']}: ran {res['n_run']}, skipped {res['n_skipped']} "
        f"of {res['n_points']} cells -> results/{res['experiment']}_summary.json"
    )


if __name__ == "__main__":
    main()
