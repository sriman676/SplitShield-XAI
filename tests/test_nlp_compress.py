"""ShieldLM compression (prune/distil/int8/gate) and self-improvement plumbing."""

from __future__ import annotations

import random

import pytest
import torch

from src.nlp import compress, selfimprove
from src.nlp.lm import ShieldLM, load, quantize_int8, save


def _model(seed: int = 0) -> ShieldLM:
    torch.manual_seed(seed)
    return ShieldLM(40, d=32, layers=3, heads=4, ctx=24, ffn=64).eval()


def _data(n: int = 64) -> list[list[int]]:
    r = random.Random(0)
    return [
        [1] + [r.randrange(5, 40) for _ in range(r.randrange(6, 20))] + [2]
        for _ in range(n)
    ]


def test_prune_to_same_shape_is_exact() -> None:
    # selecting every channel/head-dim/neuron/layer must reproduce the teacher
    # bit-for-bit up to float error: proves the head-preserving slicing is correct
    t = _model()
    s = compress.prune(t, _data(), "cpu", d=32, ffn=64, layers=3)
    x = torch.tensor([[1, 7, 9, 11, 13, 2]])
    assert torch.allclose(t(x), s(x), atol=1e-5)


def test_prune_shapes_and_distil_runs() -> None:
    t = _model()
    s = compress.prune(t, _data(), "cpu", d=16, ffn=32, layers=2)
    assert (s.cfg["d"], s.cfg["ffn"], s.cfg["layers"]) == (16, 32, 2)
    assert s.n_params() < t.n_params()
    s = compress.distill(s, t, _data(), steps=3, batch=8, device="cpu")
    assert torch.isfinite(s(torch.tensor([[1, 5, 6]]))).all()
    with pytest.raises(ValueError):
        compress.prune(t, _data(), "cpu", d=18, ffn=32, layers=2)  # d % heads != 0


def test_candidates_smallest_first_and_never_the_original() -> None:
    cfg = {"d": 192, "ffn": 768, "layers": 4, "heads": 4}
    shapes = compress.candidate_shapes(cfg)
    sizes = [d * d * L + d * f * L for _, d, f, L in shapes]
    assert sizes[0] == min(sizes) and all(s[1:] != (192, 768, 4) for s in shapes)
    assert all(d % 4 == 0 for _, d, _, _ in shapes)


def test_int8_roundtrip_exact_and_smaller(tmp_path) -> None:
    m = ShieldLM(400, d=64, layers=2).eval()
    q = quantize_int8(m)
    save(m, tmp_path / "f.pt")
    save(q, tmp_path / "q.pt")
    x = torch.tensor([[1, 50, 60, 70]])
    assert torch.equal(load(tmp_path / "q.pt")(x), q(x))
    assert (tmp_path / "q.pt").stat().st_size < (tmp_path / "f.pt").stat().st_size
    assert torch.equal(load(tmp_path / "f.pt")(x), m(x))


def test_feedback_and_rollback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(selfimprove, "NLP_DIR", tmp_path)
    monkeypatch.setattr(selfimprove, "FEEDBACK", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(selfimprove, "HISTORY", tmp_path / "hist.json")
    assert selfimprove.rollback() is False  # nothing promoted yet
    selfimprove.record_feedback({"ip": "x"}, "a <IP> b", True)
    selfimprove.record_feedback({"ip": "x"}, "bad", False)
    fb = selfimprove._feedback()
    assert [r["good"] for r in fb] == [True, False]
    (tmp_path / "shieldlm_full.pt").write_text("new")
    (tmp_path / "shieldlm_full.prev.pt").write_text("old")
    assert selfimprove.rollback() is True
    assert (tmp_path / "shieldlm_full.pt").read_text() == "old"


def test_selfimprove_cycle_end_to_end(tmp_path, monkeypatch) -> None:
    """One full cycle on tiny artifacts: self-training, analyst feedback, RLCD, gate;
    a promotion leaves a rollback point, a rejection leaves the models untouched."""
    from src.nlp import rlcd
    from src.nlp.corpus import build_corpus, random_facts
    from src.nlp.lm import Tokenizer, train_lm

    texts = [t for _, t in build_corpus(60, 0)]
    tok = Tokenizer.build(texts)
    lm = ShieldLM(len(tok.itos), d=32, layers=1, ctx=160)
    train_lm(lm, [tok.encode(t, eos=True) for t in texts], 20, 8, "cpu")
    rm = rlcd.RewardModel(len(tok.itos), d=32, layers=1)
    tok.save(tmp_path / "tokenizer.json")
    save(lm.eval(), tmp_path / "shieldlm_full.pt")
    save(lm, tmp_path / "shieldlm_student.pt")
    save(quantize_int8(lm), tmp_path / "shieldlm_deployed.pt")
    torch.save({"cfg": rm.cfg, "state": rm.state_dict()}, tmp_path / "reward.pt")

    r = random.Random(1)
    facts = [random_facts(r) for _ in range(8)]
    for name, val in (
        ("NLP_DIR", tmp_path), ("FEEDBACK", tmp_path / "fb.jsonl"),
        ("HISTORY", tmp_path / "hist.json"), ("real_facts", lambda: facts),
    ):  # fmt: skip
        monkeypatch.setattr(selfimprove, name, val)
    selfimprove.record_feedback(facts[0], "<IP> did <T1> .", False)
    before = (tmp_path / "shieldlm_full.pt").read_bytes()

    res = selfimprove.improve(fast=True)
    assert "error" not in res and set(res["gate"]["checks"]) == {
        "faithfulness_drop_le_1pt", "perplexity_rise_le_5pct", "not_losing_head_to_head",
    }  # fmt: skip
    assert res["rlcd_pairs"] >= 1 and res["feedback_down"] == 1
    if res["promoted"]:
        assert (tmp_path / "shieldlm_full.prev.pt").read_bytes() == before
        assert "int8" in torch.load(
            tmp_path / "shieldlm_deployed.pt", weights_only=True
        ) or (res["deployed"].startswith("full"))
        load(tmp_path / "shieldlm_deployed.pt")  # deployable file loads
    else:
        assert (tmp_path / "shieldlm_full.pt").read_bytes() == before
