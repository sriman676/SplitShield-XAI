"""Self-improvement entrypoint.

    python improve.py --dataset cicids2017-improved --trials 6
    python improve.py --fast --report          # quick smoke + attack/defence report

Runs one self-improvement pass (search -> distil -> quantise -> gate -> promote),
prints the leaderboard row, and optionally writes the attack/defence report for the
winning compact model. Training-heavy: intended for the GPU box, smoke-able with
--fast on CPU.
"""

from __future__ import annotations

import argparse
import json

from config.base import build_config, experiment_for_dataset
from src.selfimprove.loop import improve
from utils.device import get_primary_device
from utils.logging_config import configure_logging
from utils.seed import set_seed


def _report_for(dataset: str, fast: bool, device: str, seed: int) -> None:
    """Train the promoted (auto.yaml) arch and write the attack/defence report."""
    from src.data.preprocess import preprocess_frame
    from src.data.sources import get_spec, load_raw
    from src.explain.attack_report import build_report
    from src.selfimprove.loop import auto_model
    from src.training.centralized import train_centralized
    from utils.paths import RESULTS_DIR

    raw = experiment_for_dataset(dataset, fast)
    arch = auto_model(dataset)
    if arch:
        raw["model"] = {**raw.get("model", {}), **arch}
    cfg = build_config(raw, seed=seed, fast=fast)
    data = preprocess_frame(
        load_raw(get_spec(dataset)), get_spec(dataset), cfg, seed=seed
    )
    model, _, _ = train_centralized(data, cfg, device)
    out = RESULTS_DIR / "attack_defense" / dataset  # one report per dataset
    out.mkdir(parents=True, exist_ok=True)
    build_report(model, data, device, out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-improvement loop")
    ap.add_argument("--dataset", default="cicids2017-improved")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--budget", type=int, default=2_000_000, help="size budget (bytes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--report", action="store_true", help="also write attack/defence report"
    )
    ap.add_argument(
        "--usl", action="store_true", help="run the unsupervised-pretrain hook first"
    )
    ap.add_argument(
        "--continual", action="store_true", help="run the continual-learning hook after"
    )
    ap.add_argument(
        "--hpo", action="store_true", help="hyperparameter fine-tune around the winner"
    )
    args = ap.parse_args()

    configure_logging()
    set_seed(args.seed)
    device = get_primary_device()

    row = improve(
        dataset=args.dataset,
        n_trials=args.trials,
        fast=args.fast,
        device=device,
        size_budget_bytes=args.budget,
        seed=args.seed,
        use_usl=args.usl,
        use_continual=args.continual,
        use_hpo=args.hpo,
    )
    print("[self-improve] " + json.dumps(row))

    if args.report:
        _report_for(args.dataset, args.fast, device, args.seed)
        print(
            f"[self-improve] attack/defence report -> results/attack_defense/{args.dataset}/"
        )


if __name__ == "__main__":
    main()
