"""ShieldLM self-improvement on real SOC incidents (analyst-triggered, gated).

One cycle (``improve``), run from the dashboard's Learning page:

1. Unsupervised self-training: for each real incident the current model writes
   several narratives; those that pass the rule-based faithfulness check with a
   high score become new training text (pseudo-labels; no human input). Mixed with
   a replay of the original corpus so earlier skills are not forgotten.
2. Supervised analyst feedback: narratives an analyst marked 👍 are training
   targets; 👎 narratives become preference pairs (the reference summary is
   preferred over the disliked one) for the reward model.
3. RLCD: fresh contrastive pairs (<good> vs <bad> prompts) on the real incidents
   plus the analyst pairs continue training the reward model; the self-trained
   policy is then optimised against it with a KL penalty (rlcd.reinforce).
4. Gate + rollback: the candidate must pass the same gate as compression against
   the current model on held-out real incidents and random ones (faithfulness -1pt,
   perplexity +5 %, not losing the reward-model head-to-head). On a pass, the
   current files are kept as *.prev.pt (``rollback`` restores them), the new full
   model is saved and re-compressed to the deployed shape (int8 if it was), and
   that compressed copy must pass the gate too or the full model is deployed.

History: results/nlp_selfimprove.json. Feedback: artifacts/nlp/feedback.jsonl.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from typing import Any

import torch

from src.nlp.corpus import build_corpus, faithfulness, good_summary, prompt_text
from src.nlp.lm import EOS, ShieldLM, Tokenizer, generate, load, save, train_lm
from utils.logging_config import get_logger
from utils.paths import PROJECT_ROOT, RESULTS_DIR

_log = get_logger(__name__)
NLP_DIR = PROJECT_ROOT / "artifacts" / "nlp"
FEEDBACK = NLP_DIR / "feedback.jsonl"
HISTORY = RESULTS_DIR / "nlp_selfimprove.json"
MODEL_FILES = (
    "shieldlm_full.pt",
    "shieldlm_student.pt",
    "shieldlm_deployed.pt",
    "reward.pt",
)
PSEUDO_MIN = 0.95  # faithfulness needed for a self-generated narrative to be kept

SIZES = {
    "fast": {"samples": 2, "lm_steps": 40, "rm_epochs": 1, "rl_steps": 5, "batch": 16,
             "replay": 200, "distill_steps": 40, "eval_random": 20},
    "full": {"samples": 4, "lm_steps": 300, "rm_epochs": 2, "rl_steps": 60, "batch": 32,
             "replay": 2000, "distill_steps": 600, "eval_random": 150},
}  # fmt: skip


def record_feedback(facts: dict[str, Any], delex: str, good: bool) -> None:
    """Store an analyst's 👍/👎 on a narrative (delexicalised text + its facts)."""
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": time.time(), "facts": facts, "text": delex, "good": bool(good)}
    with open(FEEDBACK, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _feedback() -> list[dict[str, Any]]:
    if not FEEDBACK.is_file():
        return []
    out = []
    for line in FEEDBACK.read_text("utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # a half-written line from a concurrent writer
    return out


def real_facts() -> list[dict[str, Any]]:
    """Distinct facts of the SOC's correlated incidents that carry techniques."""
    from src.nlp.corpus import facts_from_incident
    from src.soc import engine

    seen, out = set(), []
    for inc in engine.incidents():
        if not inc.get("techniques"):
            continue
        f = facts_from_incident(inc)
        key = prompt_text(f) + f["ip"]
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def ready() -> dict[str, Any]:
    """What a cycle would use (shown on the dashboard before the button)."""
    need = [n for n in ("shieldlm_full.pt", "tokenizer.json", "reward.pt")]
    missing = [n for n in need if not (NLP_DIR / n).is_file()]
    fb = _feedback()
    return {
        "missing": missing,
        "incidents": len(real_facts()) if not missing else 0,
        "feedback_up": sum(1 for r in fb if r["good"]),
        "feedback_down": sum(1 for r in fb if not r["good"]),
    }


def _seq(tok: Tokenizer, f: dict[str, Any], text: str) -> list[int]:
    return tok.encode(prompt_text(f, "<sum>")) + tok.encode(text, bos=False) + [EOS]


def improve(
    device: str = "cpu", fast: bool = False, seed: int = 0, log: Any = None
) -> dict[str, Any]:
    """Run one gated self-improvement cycle. Never replaces a model that fails."""
    from src.nlp import compress, rlcd

    log = log or _log.info
    z = SIZES["fast" if fast else "full"]
    st = ready()
    if st["missing"]:
        return {"error": f"train the NLP stack first (missing {st['missing']})"}
    facts = real_facts()
    fb = _feedback()
    if len(facts) < 4 and not fb:
        return {
            "error": "need at least 4 incidents with ATT&CK techniques, or feedback"
        }

    t0 = time.time()
    rng = random.Random(seed)
    tok = Tokenizer.load(NLP_DIR / "tokenizer.json")
    current = load(NLP_DIR / "shieldlm_full.pt", device)
    ck = torch.load(NLP_DIR / "reward.pt", map_location=device, weights_only=True)
    rm = rlcd.RewardModel(**ck["cfg"]).to(device)
    rm.load_state_dict(ck["state"])
    rng.shuffle(facts)
    n_hold = len(facts) // 4
    hold, train_f = facts[:n_hold], facts[n_hold:]
    report: dict[str, Any] = {
        "ts": time.time(), "incidents": len(facts), "heldout_incidents": n_hold,
        "feedback_up": st["feedback_up"], "feedback_down": st["feedback_down"],
    }  # fmt: skip

    # 1. unsupervised self-training on pseudo-labelled narratives (+ corpus replay)
    pseudo: list[list[int]] = []
    for _ in range(z["samples"]):
        prompts = [tok.encode(prompt_text(f, "<sum>")) for f in train_f]
        for f, c in zip(train_f, generate(current, tok, prompts, device), strict=True):
            if faithfulness(tok.decode(c), f)["score"] >= PSEUDO_MIN:
                pseudo.append(tok.encode(prompt_text(f, "<sum>")) + c + [EOS])
    # 2. supervised: analyst-approved narratives are targets
    sft = [_seq(tok, r["facts"], r["text"]) for r in fb if r["good"]]
    replay = [
        tok.encode(t, eos=True) for _, t in build_corpus(z["replay"] // 2, seed + 5)
    ]
    report |= {"pseudo_labels": len(pseudo), "analyst_targets": len(sft)}
    log(f"self-improve: {len(pseudo)} pseudo-labels, {len(sft)} analyst targets")
    policy = load(NLP_DIR / "shieldlm_full.pt", device)
    if pseudo or sft:
        train_lm(policy, pseudo + sft * 3 + replay, z["lm_steps"], z["batch"],
                 device, lr=2e-4, seed=seed, log=log)  # fmt: skip
        policy.eval()

    # 3. RLCD on real incidents + analyst preference pairs
    pairs = rlcd.make_pairs(policy, tok, 0, device, seed + 1, facts=train_f)
    for r in fb:
        if not r["good"]:  # reference preferred over the disliked narrative
            ref = good_summary(r["facts"], random.Random(seed))
            pairs.append(
                (
                    r["facts"],
                    tok.encode(ref, bos=False),
                    tok.encode(r["text"], bos=False),
                )
            )
    pairs += rlcd.make_pairs(policy, tok, max(32, len(pairs)), device, seed + 2)
    rng.shuffle(pairs)
    rm, rm_acc = rlcd.train_reward(
        pairs, tok, device, z["rm_epochs"], seed, log=log, init=rm
    )
    candidate = rlcd.reinforce(
        policy, rm, tok, device, z["rl_steps"], z["batch"], seed + 3, log=log
    )
    report |= {"rlcd_pairs": len(pairs), "reward_model_pair_accuracy": rm_acc}

    # 4. gate on held-out real incidents + random incidents
    heldout = [tok.encode(t, eos=True) for _, t in build_corpus(100, seed + 9)]
    from src.nlp.corpus import random_facts

    gate_facts = hold + [random_facts(rng) for _ in range(z["eval_random"])]
    gate = compress.compare(
        current, candidate, rm, tok, heldout, device, len(gate_facts), seed + 11,
        facts=gate_facts,
    )  # fmt: skip
    report["gate"] = gate
    if not gate["passed"]:
        report["promoted"] = False
        log(f"self-improve: candidate rejected by the gate {gate['checks']}")
        return _finish(report, t0)

    for name in MODEL_FILES:  # rollback point
        if (NLP_DIR / name).is_file():
            shutil.copy2(NLP_DIR / name, NLP_DIR / name.replace(".pt", ".prev.pt"))
    torch.save({"cfg": rm.cfg, "state": rm.state_dict()}, NLP_DIR / "reward.pt")
    save(candidate, NLP_DIR / "shieldlm_full.pt")
    report["promoted"] = True
    report["deployed"] = _recompress(
        candidate, rm, tok, replay, heldout, device, z, seed, log
    )
    return _finish(report, t0)


def _recompress(
    full: ShieldLM,
    rm: torch.nn.Module,
    tok: Tokenizer,
    train: list[list[int]],
    heldout: list[list[int]],
    device: str,
    z: dict[str, int],
    seed: int,
    log: Any,
) -> str:
    """Re-apply the deployed compression (same shape, int8 if it was) to the new
    full model; deploy the compressed copy only if it passes the gate."""
    from src.nlp import compress
    from src.nlp.lm import quantize_int8

    student_ck = NLP_DIR / "shieldlm_student.prev.pt"
    deployed_ck = NLP_DIR / "shieldlm_deployed.prev.pt"
    shape: dict[str, Any] = (
        torch.load(student_ck, map_location="cpu", weights_only=True)["cfg"]
        if student_ck.is_file()
        else full.cfg
    )
    was_int8 = deployed_ck.is_file() and "int8" in torch.load(
        deployed_ck, map_location="cpu", weights_only=True
    )
    float_model = full
    dims = ("d", "ffn", "layers")
    if any(shape[k] != full.cfg[k] for k in dims):
        student = compress.prune(full, train[:256], device, *(shape[k] for k in dims))
        float_model = compress.distill(
            student, full, train, z["distill_steps"], z["batch"], device,
            seed=seed, log=log,
        )  # fmt: skip
    save(float_model, NLP_DIR / "shieldlm_student.pt")
    model: torch.nn.Module = quantize_int8(float_model) if was_int8 else float_model
    if model is not full:
        g = compress.compare(
            full, model, rm, tok, heldout, device, z["eval_random"], seed + 13
        )
        dev = "cpu" if was_int8 else device
        compress._speed(  # same speed rule as the original compression
            g,
            compress.latency_ms(model, tok, dev),
            compress.latency_ms(full, tok, device),
        )
        if not g["passed"]:
            log(
                "self-improve: compressed copy failed the gate (quality or speed); deploying the full model"
            )
            save(full, NLP_DIR / "shieldlm_deployed.pt")
            return "full (compressed copy failed the gate)"
    save(model, NLP_DIR / "shieldlm_deployed.pt")
    return ("int8 " if was_int8 else "") + (
        "student" if float_model is not full else "full"
    )


def rollback() -> bool:
    """Restore the models from before the last promoted cycle."""
    prev = [NLP_DIR / n.replace(".pt", ".prev.pt") for n in MODEL_FILES]
    if not any(p.is_file() for p in prev):
        return False
    for name, p in zip(MODEL_FILES, prev, strict=True):
        if p.is_file():
            shutil.copy2(p, NLP_DIR / name)
    _append({"ts": time.time(), "rollback": True})
    return True


def _append(row: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hist = json.loads(HISTORY.read_text("utf-8")) if HISTORY.is_file() else []
    hist.append(row)
    HISTORY.write_text(json.dumps(hist, indent=2), "utf-8")


def _finish(report: dict[str, Any], t0: float) -> dict[str, Any]:
    report["minutes"] = round((time.time() - t0) / 60, 2)
    _append(report)
    return report


__all__ = ["improve", "ready", "record_feedback", "rollback"]
