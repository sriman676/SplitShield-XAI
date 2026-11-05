"""Assistant wording (owner 2026-10-15): answer first, numbered steps, adaptive
style, typo tolerance, next-question suggestions, grammar of every answer path."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from src.nlp import feedback, style, typo

# --- typo tolerance -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("q", "want"),
    [
        ("is there any brut force happning?", "is there any brute force happening?"),
        ("whats t1059", "what is T1059"),
        ("explain pishing", "explain phishing"),
        ("what is ransomwear", "what is ransomware"),
        ("how do i stop credntial dumping", "how do i stop credential dumping"),
        ("privesc on host-12?", "privilege escalation on host-12?"),
        ("ddos?", "ddos?"),
    ],
)
def test_typos_are_corrected(q: str, want: str) -> None:
    assert typo.correct(q)[0] == want


@pytest.mark.parametrize(
    "q",
    ["what did 203.0.113.45 do", "what did Alice do on the server", "is my laptop safe",
     "how many alerts in the last hour", "is the mimikatz thing still going on"],
)  # fmt: skip
def test_names_ips_and_plain_words_are_left_alone(q: str) -> None:
    out, changes = typo.correct(q)
    assert out == q and changes == []


def test_did_you_mean_only_when_unsure() -> None:
    sure = [{"from": "brut", "to": "brute", "sure": True}]
    unsure = [{"from": "lateal", "to": "lateral", "sure": False}]
    assert typo.did_you_mean("x", sure) is None
    line = typo.did_you_mean("lateral movement", unsure)
    assert line and line.startswith("🔤 Did you mean") and "lateral" in line


# --- whole answers ----------------------------------------------------------------------


@pytest.fixture
def bot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    from src.nlp.assistant import ShieldAssistant
    from src.soc import engine
    from src.soc.parsers import parse_text
    from src.soc.samples import make_samples

    monkeypatch.setattr(engine, "EVENTS", tmp_path / "events.jsonl")
    monkeypatch.setattr(engine, "ALERTS", tmp_path / "alerts.jsonl")
    for p in make_samples(tmp_path / "s"):
        if p.suffix != ".csv":
            engine.record(parse_text(p.read_text("utf-8"), p.name), send=False)
    now = time.time()
    row = {"ts": now - 120, "src_ip": "203.0.113.45", "source": "winsec",
           "kind": "failed_logon", "detail": "4625", "techniques": ["T1110"],
           "dst_ip": None, "dst_port": None}  # fmt: skip
    with engine.EVENTS.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(row) for _ in range(6)) + "\n")
    return ShieldAssistant()


QUESTIONS = [
    "is there any brute force happening?", "is there any defense evasion happening?",
    "is there any kerberoasting happening?", "what is a brute force attack?",
    "what is lateral movement", "explain T1059", "what is ransomware",
    "how do I stop port scan", "what attacks are happening today?",
    "what did 203.0.113.45 do?", "what did 192.0.2.99 do?", "which incidents used T1110?",
    "how many alerts", "who is attacking us?", "what are the project results",
    "hello", "", "explain", "is there any heartbleed in the last hour?",
]  # fmt: skip

_SLOT = re.compile(r"<(IP|LEVEL|N|SENSORS|T\d)>|\{[a-z_]+\}")


def _check_grammar(a: dict, q: str) -> None:
    text = a["text"]
    lines = text.split("\n")
    assert lines[0] == a["headline"], "first line must be the answer line"
    head = re.sub(r"[*_`\s]+$", "", lines[0])
    assert head.endswith(
        (".", "?", "!", ")")
    ), f"answer line lacks punctuation: {lines[0]!r}"
    assert not _SLOT.search(text), f"unfilled placeholder in: {text[:200]}"
    for ln in lines:
        body = ln.rstrip()  # markdown line breaks ("  \n") are fine
        assert "  " not in body.lstrip(), f"double space: {ln!r}"
        assert not re.search(r"\b(\w{2,}) \1\b", body, re.I), f"doubled word: {ln!r}"
    for s in a["steps"]:
        assert re.match(r"^\d\. \*\*\w[\w ]*:\*\* ", s) and s.rstrip().endswith(
            (".", "…")
        ), s
    fu = a["follow_up"]
    assert 1 <= len(fu) <= 3 and len(set(fu)) == len(fu)
    assert all(f.lower().rstrip("?") != q.strip().lower().rstrip("?") for f in fu)


@pytest.mark.parametrize("mode", ["auto", "beginner", "expert"])
def test_every_answer_path_reads_well(bot, mode: str) -> None:  # noqa: ANN001
    for q in QUESTIONS:
        a = bot.answer(q, mode=mode)
        assert a["style"] in ("urgent", "learning", "expert", "normal")
        _check_grammar(a, q)


def test_active_attack_puts_steps_first(bot) -> None:  # noqa: ANN001
    a = bot.answer("is there any brute force happening?")
    assert a["style"] == "urgent" and a["intent"] == "status"
    lines = a["text"].split("\n")
    assert lines[0].startswith("🚨 `ONGOING` **Yes - brute force is happening now.**")
    assert lines[1] == "🧭 **What to do now:**" and lines[2].startswith(
        "1. **Contain:**"
    )
    assert "203.0.113.45" in lines[2] and "blocking stays your decision" in lines[2]
    assert [s[:2] for s in a["steps"]] == ["1.", "2.", "3.", "4."]
    assert "MFA" in a["text"]  # a fix step from ATT&CK / incident-response knowledge


def test_learning_question_stays_an_explanation_with_a_live_note(bot) -> None:  # noqa: ANN001
    a = bot.answer("what is a brute force attack?")
    assert a["style"] == "learning"
    lines = a["text"].split("\n")
    assert lines[0].startswith("📖 **Brute force**")
    assert lines[1].startswith("🚨 This is happening in your logs right now")
    assert a["text"].index("Types:") < a["text"].index("🧭 **What to do now:**")


def test_expert_phrasing_and_modes(bot) -> None:  # noqa: ANN001
    a = bot.answer("show 4625 event ids and a splunk query for T1110")
    assert "Expert detail" in a["text"] and "EventCode=4625" in a["text"]
    b = bot.answer("how many alerts", mode="expert")
    assert b["style"] == "expert"
    c = bot.answer("how do I stop port scan", mode="beginner")
    assert c["style"] == "learning" and "In plain words" in c["text"]
    d = bot.answer("how do I stop port scan", mode="auto")
    assert "Expert detail" not in d["text"]


def test_typo_question_is_answered_and_reported(bot) -> None:  # noqa: ANN001
    a = bot.answer("is there any brut force happning?")
    assert (
        a["intent"] == "status"
        and a["corrected"] == "is there any brute force happening?"
    )
    assert a["text"].startswith("🚨 `ONGOING`")


def test_results_question_is_not_mistaken_for_a_technique(bot) -> None:  # noqa: ANN001
    a = bot.answer("what are the project results")
    assert a["intent"] == "results" and "T1053" not in a["text"]


def test_steps_grammar_singular_and_plural() -> None:
    one = style.steps(
        "brute force", ["1.2.3.4"], ["T1110"], ["winsec"], "ongoing", "brute force"
    )
    two = style.steps("brute force", ["1.2.3.4", "5.6.7.8", "1.2.3.4"], ["T1110"], [],
                      "happened", "brute force")  # fmt: skip
    assert "1.2.3.4 is still the source. Consider blocking it" in one[0]
    assert "1.2.3.4 and 5.6.7.8 were the sources. Consider blocking them" in two[0]
    possible = style.steps("x", [], [], [], "possible", "x")
    assert possible[0].startswith("1. **Confirm first:**")


def test_thumbs_up_on_rule_based_answers_does_not_crash(tmp_path: Path) -> None:
    fb = tmp_path / "f.jsonl"
    assert feedback.record("any brute force?", "status", True, path=fb)["label"] is None
    assert (
        feedback.record("explain T1059", "explain_technique", True, path=fb)["label"]
        == "explain"
    )


def test_narratives_capitalise_acronyms() -> None:
    from src.nlp.corpus import fill

    f = {"ip": "1.2.3.4", "n": 3, "sensors": ["winsec"], "level": "high",
         "techniques": ["Brute Force (T1110)"]}  # fmt: skip
    out = fill("recommended action : block <IP> and reset the exposed credentials and "
               "enforce mfa and forward logs to a central siem .", f)  # fmt: skip
    assert "MFA" in out and "SIEM" in out and "1.2.3.4" in out
