"""Train + test + validate the NLP stack: intent model, ShieldLM, RLCD.

    python -m src.nlp.train            # full (GPU box)
    python -m src.nlp.train --fast     # minutes on CPU (preliminary)

Artifacts: artifacts/nlp/ (tokenizer, ShieldLM base + RLCD-aligned, reward model,
intent model). Metrics: results/nlp_metrics.json (also rendered in FINAL_REPORT).
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import torch

from src.nlp import intent, rlcd
from src.nlp.corpus import build_corpus
from src.nlp.lm import ShieldLM, Tokenizer, perplexity, save, train_lm
from utils.device import get_primary_device
from utils.logging_config import configure_logging, get_logger
from utils.paths import PROJECT_ROOT, RESULTS_DIR
from utils.seed import set_seed

_log = get_logger(__name__)
NLP_DIR = PROJECT_ROOT / "artifacts" / "nlp"

SIZES = {
    "fast": {
        "incidents": 600,
        "steps": 400,
        "batch": 32,
        "d": 96,
        "layers": 2,
        "pairs": 500,
        "rm_epochs": 8,
        "rl_steps": 25,
        "rl_batch": 16,
        "eval": 40,
        "distill_steps": 150,
    },
    "full": {
        "incidents": 12000,
        "steps": 4000,
        "batch": 64,
        "d": 192,
        "layers": 4,
        "pairs": 3000,
        "rm_epochs": 4,
        "rl_steps": 300,
        "rl_batch": 32,
        "eval": 300,
        "distill_steps": 1500,
    },
}


def run(fast: bool = False, seed: int = 0) -> dict:
    set_seed(seed)
    dev = get_primary_device()
    z = SIZES["fast" if fast else "full"]
    NLP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    metrics: dict = {"preliminary": fast, "device": dev, "sizes": z}

    # 1. Supervised question understanding.
    clf, m = intent.train(seed)
    joblib.dump(clf, NLP_DIR / "intent.joblib")
    metrics["intent"] = m
    _log.info("intent model: %s", m)

    # 2. ShieldLM pre-training on the security corpus (held-out 10% for perplexity).
    data = build_corpus(z["incidents"], seed)
    texts = [t for _, t in data]
    tok = Tokenizer.build(texts)
    enc = [tok.encode(t, eos=True) for t in texts]
    cut = int(len(enc) * 0.9)
    lm = ShieldLM(len(tok.itos), d=z["d"], layers=z["layers"]).to(dev)
    losses = train_lm(
        lm, enc[:cut], z["steps"], z["batch"], dev, seed=seed, log=_log.info
    )
    metrics["shieldlm"] = {
        "params": lm.n_params(),
        "vocab": len(tok.itos),
        "train_sequences": cut,
        "final_train_loss": losses[-1],
        "heldout_perplexity": perplexity(lm, enc[cut:], dev),
    }
    tok.save(NLP_DIR / "tokenizer.json")
    save(lm, NLP_DIR / "shieldlm_base.pt")

    # 3. RLCD: contrastive pairs -> reward model -> policy-gradient alignment.
    pairs = rlcd.make_pairs(lm, tok, z["pairs"], dev, seed + 1)
    rm, rm_acc = rlcd.train_reward(pairs, tok, dev, z["rm_epochs"], seed, log=_log.info)
    torch.save({"cfg": rm.cfg, "state": rm.state_dict()}, NLP_DIR / "reward.pt")
    policy = rlcd.reinforce(
        lm, rm, tok, dev, z["rl_steps"], z["rl_batch"], seed + 2, log=_log.info
    )
    save(policy, NLP_DIR / "shieldlm_rlcd.pt")
    ev = rlcd.evaluate(
        {"before_rlcd": lm, "after_rlcd": policy}, rm, tok, dev, z["eval"], seed + 3
    )
    # Promotion gate: a higher reward is not enough (a weak reward model can be
    # gamed). The aligned model ships only if the independent faithfulness check
    # does not get worse; otherwise the base model is deployed and this is reported.
    promoted = (
        ev["after_rlcd"]["faithfulness_score"]
        >= ev["before_rlcd"]["faithfulness_score"] - 0.01
    )
    aligned = policy if promoted else lm
    save(aligned, NLP_DIR / "shieldlm_full.pt")  # uncompressed: rollback + self-improve
    metrics["rlcd"] = {
        "pairs": len(pairs),
        "reward_model_heldout_pair_accuracy": rm_acc,
        "promoted": promoted,
        "deployed": "rlcd" if promoted else "base",
        **ev,
    }

    # 4. Compression: prune -> distil -> int8, deployed only if it passes the gate
    # against the aligned full model (faithfulness -1pt, perplexity +5%, RM h2h).
    from src.nlp.compress import compress

    deploy, best_float, report = compress(
        aligned, rm, tok, enc[:cut], enc[cut:], dev,
        z["distill_steps"], z["batch"], z["eval"], seed + 4, log=_log.info,
    )  # fmt: skip
    save(best_float, NLP_DIR / "shieldlm_student.pt")
    save(deploy, NLP_DIR / "shieldlm_deployed.pt")
    metrics["compression"] = report
    _log.info(
        "compression: deployed %s, %.0f%% of the original size",
        report["deployed"],
        100 * report["size_ratio"],
    )
    metrics["minutes"] = (time.time() - t0) / 60
    (RESULTS_DIR / "nlp_metrics.json").write_text(
        json.dumps(metrics, indent=2), "utf-8"
    )
    (NLP_DIR / "meta.json").write_text(
        json.dumps({"trained": time.time(), "fast": fast}), "utf-8"
    )
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train intent model + ShieldLM + RLCD")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    configure_logging()
    m = run(args.fast, args.seed)
    r = m["rlcd"]
    print(
        f"[nlp] intent acc {m['intent']['intent_accuracy_heldout_templates']:.3f} | "
        f"ShieldLM ppl {m['shieldlm']['heldout_perplexity']:.2f} ({m['shieldlm']['params']:,} params) | "
        f"RM acc {r['reward_model_heldout_pair_accuracy']:.3f} | faithfulness "
        f"{r['before_rlcd']['faithfulness_score']:.3f} -> {r['after_rlcd']['faithfulness_score']:.3f} | "
        f"win rate {r.get('win_rate_after_vs_before', float('nan')):.2f} | deployed "
        f"{m['compression']['deployed']} at {100 * m['compression']['size_ratio']:.0f}% size"
    )


if __name__ == "__main__":
    main()
