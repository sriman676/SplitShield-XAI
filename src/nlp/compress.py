"""Shrink ShieldLM without losing quality: prune -> distil -> int8, behind a gate.

Recipe (Minitron: Muralidharan et al., NeurIPS 2024, "Compact Language Models via
Pruning and Knowledge Distillation"; Sreenivas et al., 2024, "LLM Pruning and
Distillation in Practice"): estimate structure importance on a calibration batch,
cut the least important structure, then recover by distilling the original model's
token distribution into the pruned student. They report width pruning beating
depth pruning, so the search prefers width cuts. Finally dynamic int8 quantization
of the Linear layers (the fused attention in-projection stays float32).

Structures pruned (importance from activations, never from the test set):
  * hidden width d    - residual channels by mean |activation| at every LayerNorm
                        input; per head, the q/k dims and the v dims with the largest
                        weight norms, so every head keeps its own sub-space;
  * feed-forward width - neurons by mean |GELU output|;
  * depth             - the block whose removal raises calibration loss least.

Acceptance gate (owner-set, 2026-10-13): a candidate replaces the deployed model
only if, on held-out data, faithfulness drops at most 1 point (0.01), perplexity
rises at most 5 %, it does not lose the reward-model head-to-head (win rate not
significantly below 0.5, one-sided binomial p < 0.05), and generation is at most
5 % slower (int8: than the float model it was made from). The smallest passing
candidate wins; if none passes, the original stays deployed.
"""

from __future__ import annotations

import copy
import io
import random
import time
from typing import Any

import torch
from scipy.stats import binomtest
from torch import nn

from src.nlp.corpus import faithfulness, prompt_text, random_facts
from src.nlp.lm import (
    PAD,
    ShieldLM,
    Tokenizer,
    generate,
    pad_batch,
    perplexity,
    quantize_int8,
    save,
)

FAITH_DROP = 0.01
PPL_RISE = 1.05
LATENCY_RISE = 1.05  # owner: a compressed model must not make narratives slower
WIN_ALPHA = 0.05


# --- importance --------------------------------------------------------------------


@torch.no_grad()
def _importance(model: ShieldLM, calib: list[list[int]], device: str) -> dict[str, Any]:
    """Channel / neuron importance from activations on a calibration batch."""
    model.eval()
    d = model.cfg["d"]
    chan = torch.zeros(d, device=device)
    neurons: list[torch.Tensor] = []
    hooks = []
    layers = list(model.blocks.layers)

    def ln_hook(_m: nn.Module, inp: tuple[torch.Tensor, ...]) -> None:
        chan.add_(inp[0].abs().mean(dim=(0, 1)))

    def ffn_hook(i: int) -> Any:
        def h(_m: nn.Module, inp: tuple[torch.Tensor, ...]) -> None:
            neurons[i].add_(inp[0].abs().mean(dim=(0, 1)))

        return h

    for i, layer in enumerate(layers):
        neurons.append(torch.zeros(layer.linear1.out_features, device=device))
        hooks += [
            layer.norm1.register_forward_pre_hook(ln_hook),
            layer.norm2.register_forward_pre_hook(ln_hook),
            layer.linear2.register_forward_pre_hook(ffn_hook(i)),
        ]
    hooks.append(model.norm.register_forward_pre_hook(ln_hook))
    # the fused fast path bypasses the submodules the hooks sit on
    prev = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        for i in range(0, len(calib), 32):
            model(pad_batch(calib[i : i + 32], model.ctx, device))
    finally:
        torch.backends.mha.set_fastpath_enabled(prev)
        for h in hooks:
            h.remove()
    return {"channels": chan, "neurons": neurons}


@torch.no_grad()
def _calib_loss(model: ShieldLM, calib: list[list[int]], device: str) -> float:
    model.eval()
    b = pad_batch(calib, model.ctx, device)
    logits = model(b[:, :-1])
    return float(
        nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), b[:, 1:].reshape(-1), ignore_index=PAD
        )
    )


# --- pruning -----------------------------------------------------------------------


def _top(score: torch.Tensor, k: int) -> torch.Tensor:
    return torch.sort(torch.topk(score, k).indices).values


@torch.no_grad()
def prune(
    teacher: ShieldLM,
    calib: list[list[int]],
    device: str,
    d: int,
    ffn: int,
    layers: int,
) -> ShieldLM:
    """Student with the given (d, ffn, layers), initialised from the teacher's most
    important structure (see module docstring)."""
    t = teacher.cfg
    heads = t["heads"]
    if d % heads or d > t["d"] or ffn > t["ffn"] or layers > t["layers"]:
        raise ValueError(f"invalid student shape d={d} ffn={ffn} layers={layers}")
    imp = _importance(teacher, calib, device)

    # depth: drop the blocks whose removal hurts calibration loss least, one by one
    keep_layers = list(range(t["layers"]))
    while len(keep_layers) > layers:
        best, best_loss = keep_layers[0], float("inf")
        for cand in keep_layers:
            trial = copy.deepcopy(teacher)
            trial.blocks.layers = nn.ModuleList(
                [teacher.blocks.layers[i] for i in keep_layers if i != cand]
            )
            loss = _calib_loss(trial, calib, device)
            if loss < best_loss:
                best, best_loss = cand, loss
        keep_layers.remove(best)

    C = _top(imp["channels"], d)  # kept residual channels
    hd_t, hd = t["d"] // heads, d // heads
    student = ShieldLM(
        t["vocab"], d=d, layers=layers, heads=heads, ctx=t["ctx"],
        dropout=t["dropout"], ffn=ffn,
    ).to(device)  # fmt: skip
    student.tok.weight.copy_(teacher.tok.weight[:, C])
    student.pos.weight.copy_(teacher.pos.weight[:, C])
    student.norm.weight.copy_(teacher.norm.weight[C])
    student.norm.bias.copy_(teacher.norm.bias[C])

    for s_l, ti in zip(student.blocks.layers, keep_layers, strict=True):
        tl = teacher.blocks.layers[ti]
        W, B = tl.self_attn.in_proj_weight, tl.self_attn.in_proj_bias
        Wq, Wk, Wv = W[: t["d"]], W[t["d"] : 2 * t["d"]], W[2 * t["d"] :]
        bq, bk, bv = B[: t["d"]], B[t["d"] : 2 * t["d"]], B[2 * t["d"] :]
        qk_rows, v_rows = [], []
        for h in range(heads):
            rows = torch.arange(h * hd_t, (h + 1) * hd_t, device=W.device)
            # q.k score contribution of a dim ~ |w_q|*|w_k| (same dims for q and k)
            qk = Wq[rows][:, C].norm(dim=1) * Wk[rows][:, C].norm(dim=1)
            qk_rows.append(rows[_top(qk, hd)])
            v = Wv[rows][:, C].norm(dim=1) * tl.self_attn.out_proj.weight[C][
                :, rows
            ].norm(dim=0)
            v_rows.append(rows[_top(v, hd)])
        QK, V = torch.cat(qk_rows), torch.cat(v_rows)
        s_att = s_l.self_attn
        s_att.in_proj_weight.copy_(torch.cat([Wq[QK][:, C], Wk[QK][:, C], Wv[V][:, C]]))
        s_att.in_proj_bias.copy_(torch.cat([bq[QK], bk[QK], bv[V]]))
        # attention scale 1/sqrt(head_dim) changes with hd: rescale q to keep logits
        s_att.in_proj_weight[:d].mul_((hd / hd_t) ** 0.5)
        s_att.in_proj_bias[:d].mul_((hd / hd_t) ** 0.5)
        s_att.out_proj.weight.copy_(tl.self_attn.out_proj.weight[C][:, V])
        s_att.out_proj.bias.copy_(tl.self_attn.out_proj.bias[C])

        N = _top(imp["neurons"][ti], ffn)
        s_l.linear1.weight.copy_(tl.linear1.weight[N][:, C])
        s_l.linear1.bias.copy_(tl.linear1.bias[N])
        s_l.linear2.weight.copy_(tl.linear2.weight[C][:, N])
        s_l.linear2.bias.copy_(tl.linear2.bias[C])
        for n in ("norm1", "norm2"):
            getattr(s_l, n).weight.copy_(getattr(tl, n).weight[C])
            getattr(s_l, n).bias.copy_(getattr(tl, n).bias[C])
    return student.eval()


# --- distillation ------------------------------------------------------------------


def distill(
    student: ShieldLM,
    teacher: ShieldLM,
    train: list[list[int]],
    steps: int,
    batch: int,
    device: str,
    lr: float = 5e-4,
    temperature: float = 2.0,
    alpha: float = 0.9,
    seed: int = 0,
    log: Any = None,
) -> ShieldLM:
    """Logit distillation: alpha * T^2 * KL(teacher || student) + (1-alpha) * CE."""
    g = torch.Generator().manual_seed(seed)
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.1
    )
    T = temperature
    for step in range(steps):
        student.train()
        idx = torch.randint(len(train), (batch,), generator=g).tolist()
        b = pad_batch([train[i] for i in idx], student.ctx, device)
        x, y = b[:, :-1], b[:, 1:]
        keep = y.ne(PAD).reshape(-1)
        with torch.no_grad():
            soft = torch.softmax(teacher(x) / T, -1).reshape(-1, teacher.cfg["vocab"])
        logits = student(x).reshape(-1, student.cfg["vocab"])
        kl = nn.functional.kl_div(
            torch.log_softmax(logits[keep] / T, -1), soft[keep], reduction="batchmean"
        )
        ce = nn.functional.cross_entropy(logits[keep], y.reshape(-1)[keep])
        loss = alpha * T * T * kl + (1 - alpha) * ce
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        if log and (step % max(1, steps // 5) == 0 or step == steps - 1):
            log(f"distil step {step + 1}/{steps} loss {float(loss):.3f}")
    return student.eval()


# --- measurement + gate ------------------------------------------------------------


def size_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


@torch.no_grad()
def latency_ms(
    model: nn.Module, tok: Tokenizer, device: str, n: int = 8, repeats: int = 3
) -> float:
    """Per-narrative generation time; best of ``repeats`` to resist CPU noise."""
    rng = random.Random(7)
    prompts = [tok.encode(prompt_text(random_facts(rng), "<sum>")) for _ in range(n)]
    best = float("inf")
    for _ in range(repeats):
        torch.manual_seed(0)
        t0 = time.perf_counter()
        generate(model, tok, prompts, device, greedy=True)  # type: ignore[arg-type]
        best = min(best, (time.perf_counter() - t0) * 1e3 / n)
    return best


def _speed(gate: dict[str, Any], latency: float, reference: float) -> None:
    """Add the speed check to a quality gate result (in place)."""
    gate["checks"]["latency_rise_le_5pct"] = latency <= reference * LATENCY_RISE
    gate["passed"] = all(gate["checks"].values())


@torch.no_grad()
def compare(
    reference: nn.Module,
    candidate: nn.Module,
    rm: nn.Module,
    tok: Tokenizer,
    heldout: list[list[int]],
    device: str,
    n_eval: int,
    seed: int,
    facts: list[dict] | None = None,
) -> dict[str, Any]:
    """Same held-out incidents for both: faithfulness, perplexity, RM head-to-head.

    ``facts`` (e.g. real SOC incidents) replaces the random held-out incidents."""
    from src.nlp.rlcd import rm_score

    rng = random.Random(seed)
    facts = facts or [random_facts(rng) for _ in range(n_eval)]
    n_eval = len(facts)
    prompts = [tok.encode(prompt_text(f, "<sum>")) for f in facts]
    res: dict[str, Any] = {}
    rewards = {}
    for name, m in (("reference", reference), ("candidate", candidate)):
        dev = device if not getattr(m, "int8", False) else "cpu"
        torch.manual_seed(seed)  # identical sampling noise for both models
        conts = generate(m, tok, prompts, dev)  # type: ignore[arg-type]
        rewards[name] = rm_score(rm, tok, facts, conts, device)  # type: ignore[arg-type]
        faith = [
            faithfulness(tok.decode(c), f)["score"]
            for c, f in zip(conts, facts, strict=True)
        ]
        res[name] = {
            "faithfulness": float(sum(faith) / len(faith)),
            "perplexity": perplexity(m, heldout, dev),  # type: ignore[arg-type]
        }
    wins = float((rewards["candidate"] > rewards["reference"]).float().sum())
    ties = float((rewards["candidate"] == rewards["reference"]).float().sum())
    k = int(wins + ties / 2)
    res["win_rate"] = k / n_eval
    res["win_p_value_less"] = float(
        binomtest(k, n_eval, 0.5, alternative="less").pvalue
    )
    ref, cand = res["reference"], res["candidate"]
    res["checks"] = {
        "faithfulness_drop_le_1pt": cand["faithfulness"]
        >= ref["faithfulness"] - FAITH_DROP,
        "perplexity_rise_le_5pct": cand["perplexity"] <= ref["perplexity"] * PPL_RISE,
        "not_losing_head_to_head": res["win_p_value_less"] >= WIN_ALPHA,
    }
    res["passed"] = all(res["checks"].values())
    return res


def candidate_shapes(cfg: dict[str, int]) -> list[tuple[str, int, int, int]]:
    """(name, d, ffn, layers), smallest first; width cuts before depth cuts."""
    d, f, L, H = cfg["d"], cfg["ffn"], cfg["layers"], cfg["heads"]

    def r(x: float) -> int:  # round down to a multiple of the head count
        return max(H, int(x) // H * H)

    shapes = [
        ("width d/2, ffn/4", r(d / 2), max(H, f // 4), L),
        ("width 2d/3, ffn/3", r(2 * d / 3), max(H, f // 3), L),
        ("width 3d/4, ffn/2", r(3 * d / 4), max(H, f // 2), L),
        ("ffn/2", d, max(H, f // 2), L),
        ("depth -1", d, f, max(1, L - 1)),
    ]
    seen, out = set(), []
    for s in shapes:
        if s[1:] not in seen and s[1:] != (d, f, L):
            seen.add(s[1:])
            out.append(s)
    return out


def compress(
    teacher: ShieldLM,
    rm: nn.Module,
    tok: Tokenizer,
    train: list[list[int]],
    heldout: list[list[int]],
    device: str,
    steps: int,
    batch: int,
    n_eval: int,
    seed: int = 0,
    log: Any = None,
) -> tuple[nn.Module, ShieldLM, dict[str, Any]]:
    """Search smallest-first for a pruned+distilled student that passes the gate,
    then try int8 on top. Returns (deployable model, best float model, report).
    The deployable model is the teacher itself when nothing passes."""
    log = log or (lambda _m: None)
    calib = train[: min(256, len(train))]
    base = {
        "params": teacher.n_params(),
        "bytes": size_bytes(teacher),
        "latency_ms": latency_ms(teacher, tok, device),
    }
    report: dict[str, Any] = {"teacher": {**teacher.cfg, **base}, "candidates": []}
    best_float: ShieldLM = teacher
    for name, d, ffn, layers in candidate_shapes(teacher.cfg):
        log(f"compress: trying {name} (d={d} ffn={ffn} layers={layers})")
        student = prune(teacher, calib, device, d, ffn, layers)
        student = distill(
            student, teacher, train, steps, batch, device, seed=seed, log=log
        )
        gate = compare(teacher, student, rm, tok, heldout, device, n_eval, seed + 11)
        lat = latency_ms(student, tok, device)
        _speed(gate, lat, base["latency_ms"])
        row = {
            "name": name, "d": d, "ffn": ffn, "layers": layers,
            "params": student.n_params(), "bytes": size_bytes(student),
            "latency_ms": lat, **gate,
        }  # fmt: skip
        report["candidates"].append(row)
        log(
            f"compress: {name} params {row['params']:,} faith "
            f"{gate['candidate']['faithfulness']:.3f} vs {gate['reference']['faithfulness']:.3f} "
            f"ppl {gate['candidate']['perplexity']:.3f} vs {gate['reference']['perplexity']:.3f} "
            f"win {gate['win_rate']:.2f} -> {'PASS' if gate['passed'] else 'fail'}"
        )
        if gate["passed"]:
            best_float = student
            break

    # int8 on top of the chosen float model, same gate against the ORIGINAL teacher
    # int8 must not slow down the float model it was made from (it only saves disk)
    q = quantize_int8(best_float)
    qgate = compare(teacher, q, rm, tok, heldout, device, n_eval, seed + 11)
    q_lat = latency_ms(q, tok, "cpu")
    float_lat = (
        base["latency_ms"]
        if best_float is teacher
        else latency_ms(best_float, tok, device)
    )
    _speed(qgate, q_lat, float_lat)
    report["int8"] = {
        "of": "student" if best_float is not teacher else "teacher",
        "bytes": size_bytes(q),
        "latency_ms": q_lat,
        "float_latency_ms": float_lat,
        **qgate,
    }
    deploy: nn.Module = q if qgate["passed"] else best_float
    report["deployed"] = ("int8 " if qgate["passed"] else "") + (
        "student" if best_float is not teacher else "teacher"
    )
    report["deployed_bytes"] = size_bytes(deploy)
    report["size_ratio"] = report["deployed_bytes"] / base["bytes"]
    return deploy, best_float, report


__all__ = ["candidate_shapes", "compare", "compress", "distill", "prune", "save"]
