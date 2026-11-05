"""Unknown log formats, the self-evolving loop and the test-any-dataset tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.soc import evolve, unknown


def test_structure_detection() -> None:
    js = "\n".join(json.dumps({"t": i, "msg": "ok"}) for i in range(10))
    kv = "\n".join(f"time={i} user=bob action=login src=10.0.0.{i}" for i in range(10))
    cs = "time,user,action\n" + "\n".join(f"{i},bob,login" for i in range(10))
    assert unknown.detect_structure(js) == "json"
    assert unknown.detect_structure(kv) == "kv"
    assert unknown.detect_structure(cs) == "csv"
    assert unknown.detect_structure("free text here\nand more") == "text"


def test_unknown_log_gets_indicators_and_a_parser_suggestion(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    monkeypatch.setattr(unknown, "SUGGESTIONS", tmp_path / "sug.jsonl")
    lines = [
        f"ts={1700000000 + i} srcip=10.0.0.{i % 9} user=u{i % 3} action=open file=/a/b{i}"
        for i in range(300)
    ]
    lines.append(
        "ts=1700000400 srcip=203.0.113.9 user=evil action=run cmd='cat /etc/shadow'"
    )
    evs = unknown.parse_unknown("\n".join(lines), "custom.log")
    hit = [e for e in evs if e["techniques"]]
    assert (
        len(hit) == 1
        and hit[0]["techniques"] == ["T1003"]
        and hit[0]["src_ip"] == "203.0.113.9"
    )
    sug = json.loads((tmp_path / "sug.jsonl").read_text().splitlines()[0])
    assert sug["structure"] == "kv" and sug["fields"]["src_ip"] == "srcip"
    assert sug["status"] == "pending owner approval"


def test_evolution_keeps_only_improvements(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(evolve, "DIR", tmp_path)
    monkeypatch.setattr(evolve, "BENCH", tmp_path / "b.jsonl")
    monkeypatch.setattr(evolve, "STATE", tmp_path / "s.json")
    monkeypatch.setattr(evolve, "HISTORY", tmp_path / "h.jsonl")
    rows = [{"line": "GET /x?cmd=whoami HTTP", "attack": True, "step": "webshell"}]
    rows += [
        {
            "line": "Dec 10 06:55:46 h sshd[1]: Accepted publickey for bob",
            "attack": False,
            "step": "",
        }
    ] * 50
    (tmp_path / "b.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    base = {
        "steps_caught": 1.0,
        "attack_lines_caught": 1.0,
        "false_alarm_rate": 0.0,
        "events_per_second": 100,
    }
    assert evolve._better({**base, "steps_caught": 0.9}, base)[0] is False
    assert evolve._better({**base, "false_alarm_rate": 0.01}, base)[0] is False
    assert evolve._better({**base, "events_per_second": 200}, base) == (True, "faster")
    d = evolve.iterate()
    assert "champion" in d and evolve.HISTORY.is_file()


@pytest.mark.parametrize("name", ["auth.log"])
def test_test_any_dataset_on_logs(tmp_path: Path, name: str) -> None:
    from scripts.test_dataset import test_path

    f = tmp_path / name
    f.write_text(
        "\n".join(
            f"Dec 10 06:55:{i:02d} h sshd[1]: Failed password for root from 203.0.113.7 port 22"
            for i in range(10)
        )  # fmt: skip
    )
    card = test_path(f)[0]
    assert card["type"] == "logs" and card["alerts"] == 10 and card["incidents"] >= 1
