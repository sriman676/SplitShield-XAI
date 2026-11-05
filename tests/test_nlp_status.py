"""Full-sentence questions: detailed explanations and 'is it happening?' log checks
(owner 2026-10-14: ongoing = activity within the last hour)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.knowledge.attack_details import DETAILS, TACTICS, TECHNIQUE_TACTICS
from src.nlp.intent import entities, status_question
from src.nlp.status import check, describe, follow_ups
from src.soc.correlate import CLASS_TECHNIQUES, TECHNIQUES

NOW = 1_800_000_000.0


def _ev(ts: float, **kw) -> dict:  # noqa: ANN003
    return {
        "ts": ts,
        "src_ip": "203.0.113.9",
        "source": "winsec",
        "dst_ip": None,
        "dst_port": None,
        "kind": "log",
        "detail": "",
        "techniques": [],
        **kw,
    }


def test_knowledge_is_complete_and_consistent() -> None:
    for fam in CLASS_TECHNIQUES:
        if fam == "ATTACK":
            continue
        d = DETAILS[fam]
        assert len(d["types"]) >= 1 and len(d["steps"]) >= 3 and d["signs"], fam
    assert set(TECHNIQUE_TACTICS) == set(TECHNIQUES)  # every technique has tactics
    for t, tactics in TECHNIQUE_TACTICS.items():
        assert TECHNIQUES[t][1] in tactics  # the correlator's tactic is among them
        assert all(x in TACTICS for x in tactics)
    assert TACTICS["defense-evasion"]["id"] == "TA0005"


@pytest.mark.parametrize(
    ("q", "attack", "tactic", "status"),
    [
        ("is there any defense evasion attack done", None, "defense-evasion", True),
        ("what is brute force attack", "BruteForce", None, False),
        (
            "has anyone tried to brute force our servers and is it still going on?",
            "BruteForce",
            None,
            True,
        ),
        ("explain pgd evasion", "PGD", None, False),
        ("what is the impact of a ddos", "DDoS", None, False),
        ("any c2 activity?", None, "command-and-control", True),
    ],
)
def test_entities_and_status_cues(q, attack, tactic, status) -> None:  # noqa: ANN001
    e = entities(q)
    assert (e["attack"], e["tactic"], status_question(q)) == (attack, tactic, status)


def test_status_ongoing_happened_possible_not_seen() -> None:
    brute = [_ev(NOW - 600, techniques=["T1110"]) for _ in range(5)]
    r = check(brute, ["T1110"], "BruteForce", now=NOW)
    assert r["status"] == "ongoing" and r["strong"] == 5 and not r["needs_follow_up"]
    r = check([_ev(NOW - 7200, techniques=["T1110"])] * 5, ["T1110"], now=NOW)
    assert r["status"] == "happened"  # last activity 2 h ago > 1 h window
    low = [_ev(NOW - 60, source="detector", tier="low", label="BruteForce",
               candidate_techniques=["T1110"])]  # fmt: skip
    r = check(low, ["T1110"], "BruteForce", now=NOW)
    assert r["status"] == "possible" and r["needs_follow_up"]
    assert any("low-confidence" in q for q in follow_ups("brute force", r, False))
    r = check([_ev(NOW, techniques=["T1046"])], ["T1070"], now=NOW)
    assert r["status"] == "not_seen" and r["sources_checked"] == ["winsec"]
    assert "`NOT SEEN` **No" in describe("Defense Evasion", r, now=NOW)[0]


def test_assistant_answers_full_sentences(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    from src.nlp.assistant import ShieldAssistant
    from src.soc import engine

    events = tmp_path / "events.jsonl"
    now = time.time()
    rows = [_ev(now - 300, techniques=["T1070"], kind="audit_log_cleared")] * 3
    events.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(engine, "EVENTS", events)
    monkeypatch.setattr(engine, "ALERTS", tmp_path / "alerts.jsonl")
    bot = ShieldAssistant()
    a = bot.answer("is there any defense evasion happening?")
    assert a["intent"] == "status"
    assert a["text"].startswith(
        "🚨 `ONGOING` **Yes - Defense Evasion (TA0005) is happening now.**"
    )
    assert "T1070" in a["text"] and "winsec" in a["text"]
    b = bot.answer("what is a brute force attack?")
    assert "Password spraying" in b["text"] and "1. " in b["text"]
    assert "T1110" in b["text"] and "In your logs right now" in b["text"]
    c = bot.answer("what is lateral movement")
    assert "TA0008" in c["text"] and "T1021 Remote Services" in c["text"]
    d = bot.answer("what is resource development")  # no log source shows it
    assert "TA0042" in d["text"] and "Sysmon" in d["text"]


def test_review_fixes_grammar_dedupe_heartbleed_and_techniques() -> None:
    from src.nlp.assistant import _dedupe_recs
    from src.nlp.status import _is, check

    assert _is("web application attacks") == "are" and _is("brute force") == "is"
    rec = "reset the exposed credentials and enforce mfa"
    txt = f"recommended action : block <IP> and {rec} and {rec} ."
    assert _dedupe_recs(txt).count(rec) == 1
    web = {"ts": NOW, "source": "suricata", "detail": "ET WEB_SERVER SQL injection",
           "kind": "ids_alert", "techniques": ["T1190"]}  # fmt: skip
    assert check([web], ["T1190"], "Heartbleed", now=NOW)["status"] == "not_seen"
    assert check([web], ["T1190"], "WebAttack", now=NOW)["status"] == "ongoing"
    e = entities("any credential dumping?")
    assert e["tids"] == ["T1003"] and status_question("any credential dumping?")
