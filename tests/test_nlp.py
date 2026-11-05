"""NLP stack: faithfulness metric, slot filling, constrained generation, grounding."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from src.nlp.corpus import (
    bad_summary,
    faithfulness,
    fill,
    good_summary,
    prompt_text,
    random_facts,
)
from src.nlp.lm import ShieldLM, Tokenizer, generate


def test_reference_summaries_are_fully_faithful_and_negatives_are_not() -> None:
    rng = random.Random(0)
    good, bad = [], []
    for _ in range(200):
        f = random_facts(rng)
        good.append(faithfulness(good_summary(f, rng), f)["score"])
        bad.append(faithfulness(bad_summary(f, rng), f)["score"])
    assert min(good) == 1.0
    assert sum(bad) / len(bad) < 0.85  # corruptions are detectable on average


def test_fill_leaves_no_slots_and_uses_real_values() -> None:
    f = random_facts(random.Random(3))
    text = fill(good_summary(f, random.Random(3)), f)
    assert "<" not in text and f["ip"] in text and f["techniques"][0] in text


def test_generation_cannot_invent_absent_slots() -> None:
    rng = random.Random(1)
    f = random_facts(rng)
    f["techniques"] = f["techniques"][:1]  # only <T1> exists
    tok = Tokenizer.build(
        [prompt_text(f, c) + " " + good_summary(f, rng) for c in ("<good>", "<sum>")]
    )
    torch.manual_seed(0)
    lm = ShieldLM(len(tok.itos), d=32, layers=1, heads=2)  # untrained: worst case
    prompt = tok.encode(prompt_text(f, "<sum>"))
    for out in generate(
        lm, tok, [prompt] * 20, "cpu", max_new=40, temperature=2.0, top_k=50
    ):
        text = tok.decode(out)
        assert not any(f"<T{i}>" in text for i in range(2, 6))


def test_assistant_answers_from_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.nlp.assistant import ShieldAssistant
    from src.soc import engine
    from src.soc.parsers import parse_text
    from src.soc.samples import make_samples

    monkeypatch.setattr(engine, "EVENTS", tmp_path / "events.jsonl")
    monkeypatch.setattr(engine, "ALERTS", tmp_path / "alerts.jsonl")
    for p in make_samples(tmp_path / "s"):
        if p.suffix != ".csv":
            engine.record(parse_text(p.read_text("utf-8"), p.name), send=False)
    bot = ShieldAssistant()
    ip = bot.answer("what did 203.0.113.45 do?")
    assert (
        ip["intent"] == "ip"
        and "203.0.113.45" in ip["text"]
        and "Brute Force" in ip["text"]
    )
    assert "How to defend" in bot.answer("how do I stop port scan")["text"]
    assert bot.answer("how many alerts")["intent"] == "count"
    assert (
        "198.51.100.7" not in bot.answer("what did 192.0.2.10 do")["text"]
    )  # no leakage across IPs
    assert bot.answer("")["intent"] == "help"


ROUTING = [
    ("give me a quick tour", "help"), ("what should I type", "help"),
    ("hello assistant", "help"), ("how many critical incidents", "count"),
    ("number of failed logins in the last hour", "count"),
    ("were the hypotheses confirmed", "results"), ("is h3 supported", "results"),
    ("what is going on right now", "incidents"), ("are we under attack right now", "incidents"),
    ("what attacks are happening today?", "incidents"),
    ("what has 192.168.1.20 been doing", "ip"), ("which incidents involve T1046", "mitre"),
    ("tell me about heartbleed", "explain"), ("explain generic attack", "explain"),
    ("how can we block botnets", "defend"), ("stop port scans", "defend"),
    ("top attackers", "top"), ("who is hitting us hardest", "top"),
]  # fmt: skip


def test_question_routing_on_unseen_paraphrases() -> None:
    """End-to-end routing (entities -> strong cues -> classifier -> keywords) on
    phrasings that appear in no training template."""
    from src.nlp import intent
    from src.nlp.assistant import ShieldAssistant

    bot = ShieldAssistant.__new__(ShieldAssistant)
    bot.clf = intent.train(0)[0]
    wrong = [(q, t, bot.intent(q)) for q, t in ROUTING if bot.intent(q) != t]
    assert not wrong, wrong


def test_assistant_scope_and_stop_wording(tmp_path):
    """Owner 2026-10-22: off-topic questions are said to be out of scope (not
    "corrected" into security words); "stop" is not "top"; password spraying is a
    brute-force type; "what is ransomware" explains the technique."""
    from src.soc import sandbox

    sandbox.isolate(tmp_path)
    from src.nlp.assistant import ShieldAssistant

    a = ShieldAssistant()
    for q in ("what is the weather today", "tell me a joke", "how do i cook pasta"):
        assert a.answer(q)["intent"] == "off_topic", q
    assert a.answer("how do I stop password spraying")["intent"] == "defend"
    assert a.answer("what is ransomware")["intent"] == "explain_technique"
    for q in ("what happened today", "who is attacking us", "brut forse", "make a report of today"):
        assert a.answer(q)["intent"] != "off_topic", q
