"""RLCD — Reinforcement Learning from Contrastive Distillation (Yang et al., 2023).

1. Contrastive distillation: for each incident, ShieldLM writes one summary under the
   positive prompt (<good>) and one under the negative prompt (<bad>). The pair's
   preference label comes from which prompt produced it — no human labelling.
2. Reward model: a small transformer scores (neutral prompt, summary); trained with
   the Bradley-Terry loss  -log sigmoid(r(y+) - r(y-)).
3. Policy optimisation: the unaligned <sum> policy is fine-tuned with REINFORCE on the
   reward, minus a KL penalty to the frozen base model (keeps fluency, limits
   reward hacking). PPO in the paper; REINFORCE with a batch baseline here.
Evaluation uses the reward model AND an independent rule-based faithfulness score,
so "the reward went up" is never the only evidence.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from src.nlp.corpus import faithfulness, prompt_text, random_facts
from src.nlp.lm import PAD, ShieldLM, Tokenizer, generate, pad_batch, seq_logprob


class RewardModel(nn.Module):
    def __init__(
        self, vocab: int, d: int = 128, layers: int = 2, heads: int = 4, ctx: int = 200
    ) -> None:
        super().__init__()
        self.cfg = {
            "vocab": vocab,
            "d": d,
            "layers": layers,
            "heads": heads,
            "ctx": ctx,
        }
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(ctx, d)
        block = nn.TransformerEncoderLayer(
            d, heads, 4 * d, 0.1, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = x.eq(PAD)
        h = self.enc(
            self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device)),
            src_key_padding_mask=pad,
        )
        keep = (~pad).unsqueeze(-1).float()
        return self.out((h * keep).sum(1) / keep.sum(1).clamp(min=1)).squeeze(-1)


def _neutral(tok: Tokenizer, f: dict[str, Any]) -> list[int]:
    return tok.encode(prompt_text(f, "<sum>"))


def rm_score(
    rm: RewardModel,
    tok: Tokenizer,
    facts: list[dict],
    conts: list[list[int]],
    device: str,
) -> torch.Tensor:
    seqs = [_neutral(tok, f) + c for f, c in zip(facts, conts, strict=True)]
    return rm(pad_batch(seqs, rm.ctx, device))


def make_pairs(
    lm: ShieldLM,
    tok: Tokenizer,
    n: int,
    device: str,
    seed: int,
    facts: list[dict] | None = None,
) -> list[tuple[dict, list[int], list[int]]]:
    """Contrastive pairs on ``facts`` (default: ``n`` random incidents)."""
    rng = random.Random(seed)
    facts = facts if facts is not None else [random_facts(rng) for _ in range(n)]
    pos = generate(
        lm, tok, [tok.encode(prompt_text(f, "<good>")) for f in facts], device
    )
    neg = generate(
        lm, tok, [tok.encode(prompt_text(f, "<bad>")) for f in facts], device
    )
    # drop ties: identical outputs carry no preference signal
    return [(f, p, q) for f, p, q in zip(facts, pos, neg, strict=True) if p != q]


def train_reward(
    pairs: list,
    tok: Tokenizer,
    device: str,
    epochs: int,
    seed: int,
    log: Any = None,
    init: RewardModel | None = None,
) -> tuple[RewardModel, float]:
    """Bradley-Terry reward model; ``init`` continues training an existing one."""
    torch.manual_seed(seed)
    cut = max(1, int(len(pairs) * 0.8))
    tr, te = pairs[:cut], pairs[cut:] or pairs[-1:]
    rm = init if init is not None else RewardModel(len(tok.itos)).to(device)
    opt = torch.optim.AdamW(rm.parameters(), lr=3e-4, weight_decay=0.01)
    bs = 32
    for ep in range(epochs):
        rm.train()
        random.Random(seed + ep).shuffle(tr)
        for i in range(0, len(tr), bs):
            b = tr[i : i + bs]
            f = [x[0] for x in b]
            loss = -nn.functional.logsigmoid(
                rm_score(rm, tok, f, [x[1] for x in b], device)
                - rm_score(rm, tok, f, [x[2] for x in b], device)
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if log:
            log(f"reward model epoch {ep + 1}/{epochs} loss {float(loss):.3f}")
    rm.eval()
    with torch.no_grad():
        f = [x[0] for x in te]
        acc = float(
            (
                rm_score(rm, tok, f, [x[1] for x in te], device)
                > rm_score(rm, tok, f, [x[2] for x in te], device)
            )
            .float()
            .mean()
        )
    return rm, acc


def reinforce(
    base: ShieldLM,
    rm: RewardModel,
    tok: Tokenizer,
    device: str,
    steps: int,
    batch: int,
    seed: int,
    beta: float = 0.05,
    lr: float = 2e-5,
    log: Any = None,
) -> ShieldLM:
    policy = copy.deepcopy(base)
    ref = copy.deepcopy(base).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    rng = random.Random(seed)
    for step in range(steps):
        facts = [random_facts(rng) for _ in range(batch)]
        prompts = [tok.encode(prompt_text(f, "<sum>")) for f in facts]
        conts = generate(policy, tok, prompts, device, temperature=1.0, top_k=40)
        policy.eval()  # no dropout: sampling and scoring use the same network
        lp = seq_logprob(policy, prompts, conts, device)
        with torch.no_grad():
            lp_ref = seq_logprob(ref, prompts, conts, device)
            r = rm_score(rm, tok, facts, conts, device) - beta * (lp.detach() - lp_ref)
            adv = (r - r.mean()) / (r.std() + 1e-6)
        loss = -(adv * lp).mean()
        if not torch.isfinite(loss):
            continue  # skip a degenerate batch rather than corrupt the weights
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if log and (step % max(1, steps // 10) == 0 or step == steps - 1):
            log(f"RLCD step {step + 1}/{steps} reward {float(r.mean()):.3f}")
    return policy.eval()


@torch.no_grad()
def evaluate(
    models: dict[str, ShieldLM],
    rm: RewardModel,
    tok: Tokenizer,
    device: str,
    n: int,
    seed: int,
) -> dict[str, Any]:
    """Same held-out incidents for every model: RM reward, faithfulness, win rate."""
    rng = random.Random(seed)
    facts = [random_facts(rng) for _ in range(n)]
    prompts = [tok.encode(prompt_text(f, "<sum>")) for f in facts]
    out: dict[str, Any] = {}
    rewards = {}
    for name, m in models.items():
        conts = generate(m, tok, prompts, device)
        rewards[name] = rm_score(rm, tok, facts, conts, device).cpu().numpy()
        faith = [
            faithfulness(tok.decode(c), f) for c, f in zip(conts, facts, strict=True)
        ]
        out[name] = {
            "reward_mean": float(rewards[name].mean()),
            **{
                f"faithfulness_{k}": float(np.mean([x[k] for x in faith]))
                for k in faith[0]
            },
            "example": tok.decode(conts[0]),
        }
    if "before_rlcd" in rewards and "after_rlcd" in rewards:
        out["win_rate_after_vs_before"] = float(
            np.mean(rewards["after_rlcd"] > rewards["before_rlcd"])
        )
    return out
