"""ShieldLM: a small decoder-only transformer (GPT-style) trained from scratch.

Word-level tokenizer with slot/control tokens, causal self-attention, learned
positions. Generation can only emit slot tokens present in the prompt, so it can
never refer to a technique or address the incident does not contain.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.nlp.corpus import CONTROL, MAX_T, SLOTS

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>", "<sep>", *CONTROL, *SLOTS]
PAD, BOS, EOS, UNK, SEP = range(5)
_TOKEN = re.compile(r"<[a-zA-Z0-9]+>|[a-z0-9]+(?:[-'&.][a-z0-9]+)*|[^\sa-z0-9]", re.I)


class Tokenizer:
    def __init__(self, itos: list[str]) -> None:
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}

    @classmethod
    def build(cls, texts: list[str], min_count: int = 1) -> Tokenizer:
        counts = Counter(t for x in texts for t in cls.split(x) if t not in SPECIAL)
        words = sorted(w for w, c in counts.items() if c >= min_count)
        return cls([*SPECIAL, *words])

    @staticmethod
    def split(text: str) -> list[str]:
        out = []
        for tok in _TOKEN.findall(text):
            out.append(
                tok if tok.startswith("<") and tok.endswith(">") else tok.lower()
            )
        return out

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> list[int]:
        ids = [self.stoi.get(t, UNK) for t in self.split(text)]
        return ([BOS] if bos else []) + ids + ([EOS] if eos else [])

    def decode(self, ids: list[int]) -> str:
        toks = [self.itos[i] for i in ids if i not in (PAD, BOS, EOS)]
        return " ".join(toks)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.itos), "utf-8")

    @classmethod
    def load(cls, path: Path) -> Tokenizer:
        return cls(json.loads(path.read_text("utf-8")))


class ShieldLM(nn.Module):
    def __init__(
        self,
        vocab: int,
        d: int = 192,
        layers: int = 4,
        heads: int = 4,
        ctx: int = 160,
        dropout: float = 0.1,
        ffn: int | None = None,  # feed-forward width; None = 4*d (the original)
    ) -> None:
        super().__init__()
        ffn = ffn or 4 * d
        self.cfg: dict[str, Any] = {
            "vocab": vocab,
            "d": d,
            "layers": layers,
            "heads": heads,
            "ctx": ctx,
            "dropout": dropout,
            "ffn": ffn,
        }
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(ctx, d)
        block = nn.TransformerEncoderLayer(
            d,
            heads,
            ffn,
            dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        # GPT-style small init: N(0,1) embeddings + tied head give huge initial
        # logits and training stalls (observed: perplexity ~5000 after 150 steps).
        nn.init.normal_(self.tok.weight, 0.0, 0.02)
        nn.init.normal_(self.pos.weight, 0.0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "int8", False):
            # the fused eval fast path reads float Linear weights; int8 has none
            prev = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            try:
                return self._forward(x)
            finally:
                torch.backends.mha.set_fastpath_enabled(prev)
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
        h = self.blocks(h, mask=mask, is_causal=True, src_key_padding_mask=x.eq(PAD))
        return self.head(self.norm(h))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def pad_batch(seqs: list[list[int]], ctx: int, device: str) -> torch.Tensor:
    L = min(ctx, max(len(s) for s in seqs))
    out = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:L]
        out[i, : len(s)] = torch.tensor(s)
    return out.to(device)


def lm_loss(model: ShieldLM, batch: torch.Tensor) -> torch.Tensor:
    logits = model(batch[:, :-1])
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), ignore_index=PAD
    )


def train_lm(
    model: ShieldLM,
    train: list[list[int]],
    steps: int,
    batch: int,
    device: str,
    lr: float = 1e-3,
    seed: int = 0,
    log: Any = None,
) -> list[float]:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.1
    )
    model.train()
    losses = []
    for step in range(steps):
        idx = torch.randint(len(train), (batch,), generator=g).tolist()
        loss = lm_loss(model, pad_batch([train[i] for i in idx], model.ctx, device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if log and (step % max(1, steps // 10) == 0 or step == steps - 1):
            log(f"ShieldLM step {step + 1}/{steps} loss {loss:.3f}")
    return losses


@torch.no_grad()
def perplexity(
    model: ShieldLM, data: list[list[int]], device: str, batch: int = 64
) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(data), batch):
        b = pad_batch(data[i : i + batch], model.ctx, device)
        logits = model(b[:, :-1])
        nll = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            b[:, 1:].reshape(-1),
            ignore_index=PAD,
            reduction="sum",
        )
        tot += float(nll)
        n += int(b[:, 1:].ne(PAD).sum())
    return math.exp(tot / max(n, 1))


def _banned(tok: Tokenizer, prompt: list[int]) -> torch.Tensor:
    """Token ids the continuation may not use: slots absent from the prompt,
    control/structure tokens."""
    present = set(prompt)
    ban = [tok.stoi[t] for t in SLOTS if tok.stoi[t] not in present]
    ban += [tok.stoi[t] for t in ["<pad>", "<bos>", "<unk>", "<sep>", *CONTROL]]
    return torch.tensor(ban, dtype=torch.long)


@torch.no_grad()
def generate(
    model: ShieldLM,
    tok: Tokenizer,
    prompts: list[list[int]],
    device: str,
    max_new: int = 90,
    temperature: float = 0.8,
    top_k: int = 20,
    greedy: bool = False,
) -> list[list[int]]:
    """Batched sampling. Prompts of equal length are batched together."""
    model.eval()
    out: list[list[int]] = [[] for _ in prompts]
    groups: dict[int, list[int]] = {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)
    for L, idxs in groups.items():
        x = torch.tensor([prompts[i] for i in idxs], dtype=torch.long, device=device)
        bans = [_banned(tok, prompts[i]).to(device) for i in idxs]
        done = torch.zeros(len(idxs), dtype=torch.bool, device=device)
        for _ in range(min(max_new, model.ctx - L)):
            logits = model(x[:, -model.ctx :])[:, -1, :] / max(temperature, 1e-4)
            for r, b in enumerate(bans):
                logits[r, b] = float("-inf")
            if greedy:
                nxt = logits.argmax(-1)
            else:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
                nxt = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            x = torch.cat([x, nxt[:, None]], 1)
            done |= nxt.eq(EOS)
            if bool(done.all()):
                break
        for r, i in enumerate(idxs):
            cont = x[r, L:].tolist()
            out[i] = (
                cont[: cont.index(EOS)]
                if EOS in cont
                else [c for c in cont if c != PAD]
            )
    return out


def seq_logprob(
    model: ShieldLM, prompts: list[list[int]], conts: list[list[int]], device: str
) -> torch.Tensor:
    """Sum of log p(continuation | prompt) per sample (differentiable)."""
    seqs = [p + c + [EOS] for p, c in zip(prompts, conts, strict=True)]
    x = pad_batch(seqs, model.ctx, device)
    logp = torch.log_softmax(model(x[:, :-1]), -1)
    tgt = x[:, 1:]
    lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(lp)
    for i, p in enumerate(prompts):
        end = min(len(p) + len(conts[i]) + 1, x.size(1)) - 1
        mask[i, len(p) - 1 : end] = 1.0
    return (lp * mask).sum(1)


def quantize_int8(model: ShieldLM) -> nn.Module:
    """Dynamic int8 for the Linear layers (feed-forward, attention out, head), CPU.

    The fused attention in-projection is a raw parameter, not an nn.Linear, so it
    stays float32; the head is untied from the embedding in the quantized copy."""
    import copy

    q = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).cpu().eval(), {nn.Linear}, dtype=torch.qint8
    )
    q.int8 = True  # read by forward()
    return q


def save(model: nn.Module, path: Path) -> None:
    """Float ShieldLM -> {cfg, state}. An int8 copy (from ``quantize_int8``) stores
    each quantized Linear as plain int8 values + scale + zero point, so the file is
    small AND still loads with ``weights_only=True`` (no pickled objects)."""
    cfg = model.cfg  # ShieldLM or its int8 copy
    qlin = torch.ao.nn.quantized.dynamic.Linear
    q = {
        name: {
            "int": mod.weight().int_repr(),
            "scale": torch.tensor(mod.weight().q_scale()),
            "zp": torch.tensor(mod.weight().q_zero_point()),
        }
        for name, mod in model.named_modules()
        if isinstance(mod, qlin)
    }
    if not q:
        torch.save({"cfg": cfg, "state": model.state_dict()}, path)
        return
    state = {}
    for name, mod in model.named_modules():
        if isinstance(mod, qlin):
            if mod.bias() is not None:
                state[f"{name}.bias"] = mod.bias()
        elif not list(mod.children()):
            state |= {
                f"{name}.{k}" if name else k: v for k, v in mod.state_dict().items()
            }
    for name, mod in model.named_modules():  # MHA in_proj_* live on the MHA module
        if isinstance(mod, nn.MultiheadAttention):
            for k in ("in_proj_weight", "in_proj_bias"):
                state[f"{name}.{k}"] = getattr(mod, k).detach()
    torch.save({"cfg": cfg, "state": state, "int8": q}, path)


def load(path: Path, device: str = "cpu") -> ShieldLM:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = ShieldLM(**ck["cfg"])
    if "int8" not in ck:
        m.load_state_dict(ck["state"])
        return m.to(device).eval()
    m.load_state_dict(ck["state"], strict=False)  # quantized weights come next
    qm = quantize_int8(m)
    for name, w in ck["int8"].items():
        mod = qm.get_submodule(name)
        qw = torch._make_per_tensor_quantized_tensor(
            w["int"], float(w["scale"]), int(w["zp"])
        )
        mod.set_weight_bias(qw, mod.bias())  # type: ignore[operator]
    return qm.eval()  # type: ignore[return-value]  # same call interface, CPU only


__all__ = [
    "MAX_T",
    "ShieldLM",
    "Tokenizer",
    "generate",
    "load",
    "perplexity",
    "quantize_int8",
    "save",
    "seq_logprob",
    "train_lm",
]
