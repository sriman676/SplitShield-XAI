"""Self-improvement controller, code proposals, per-prompt learning, NetFlow/Zeek
update helpers (owner 2026-10-15)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pandas as pd
import pytest

from src.nlp import feedback
from src.soc import codeopt, controller
from src.soc.model_updates import _binary_labels, zeek_rows

# --- controller guard ------------------------------------------------------------------


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """One fake model whose file content and speed the test controls."""
    f = tmp_path / "m" / "model.bin"
    f.parent.mkdir()
    f.write_text("champion")
    box = {"speed": [100.0]}
    model = controller.Model(
        "fake", "Fake model", [f.parent], lambda: box["speed"].pop(0)
    )
    monkeypatch.setattr(controller, "registry", lambda: {"fake": model})
    monkeypatch.setattr(controller, "HOME", tmp_path / "ctl")
    monkeypatch.setattr(controller, "HISTORY", tmp_path / "ctl" / "h.jsonl")
    return {"file": f, "speed": box}


def _write_and(res: dict, f: Path):  # noqa: ANN202
    def update() -> dict:
        f.write_text("challenger")
        return res

    return update


def test_promoted_when_better_and_not_slower(fake: dict) -> None:
    fake["speed"]["speed"] = [100.0, 95.0]  # 5 % slower: within timing noise
    out = controller.guarded("fake", _write_and({"promoted": True}, fake["file"]), "t")
    assert out["decision"] == "promoted"
    assert fake["file"].read_text() == "challenger"


def test_rolled_back_when_slower(fake: dict) -> None:
    fake["speed"]["speed"] = [100.0, 50.0]
    out = controller.guarded("fake", _write_and({"promoted": True}, fake["file"]), "t")
    assert out["decision"].startswith("rolled back")
    assert fake["file"].read_text() == "champion"


def test_rolled_back_when_score_dips(fake: dict) -> None:
    res = {"kept": True, "macro_f1_before": 0.90, "macro_f1_after": 0.89}
    out = controller.guarded("fake", _write_and(res, fake["file"]), "t")
    assert out["decision"] == "rolled back: held-out score dipped"
    assert fake["file"].read_text() == "champion"


def test_nothing_trained_is_not_called_a_champion_win(fake: dict) -> None:
    # 2026-10-16: ShieldLM's nightly cycle without incidents said "kept champion
    # (challenger not better)" although no challenger was ever trained
    res = {"error": "need at least 4 incidents with ATT&CK techniques, or feedback"}
    out = controller.guarded("fake", lambda: res, "nightly cycle")
    assert out["decision"].startswith("not trained: need at least 4 incidents")
    assert fake["file"].read_text() == "champion"


def test_failed_update_restores_champion(fake: dict) -> None:
    def boom() -> dict:
        fake["file"].write_text("half-written")
        raise RuntimeError("disk full")

    out = controller.guarded("fake", boom, "t")
    assert out["decision"] == "failed" and "disk full" in out["error"]
    assert fake["file"].read_text() == "champion"
    assert controller.history()[-1]["decision"] == "failed"


@pytest.mark.parametrize(
    ("res", "ok"),
    [
        ({"promoted": True}, True),
        ({"promoted": False}, False),
        ({"kept": "challenger"}, True),
        ({"kept": "champion"}, False),
        ({"status": "promoted"}, True),
        ({"status": "rejected (would get worse)"}, False),
        ({"decision": "challenger promoted"}, True),
        ({"kept": True, "macro_f1_before": 0.5, "macro_f1_after": 0.5}, True),
    ],
)
def test_promotion_keys_normalised(res: dict, ok: bool) -> None:
    assert controller.promoted(res) is ok


def test_registry_covers_nine_models() -> None:
    names = set(controller.registry())
    assert names == {"flow", "netflow", "zeek", "phishing", "malware", "ics",
                     "insider", "intent", "shieldlm"}  # fmt: skip


# --- code proposals --------------------------------------------------------------------

HOT = """import re


def hot(s, flags):
    a = re.search(r"ab+c", s, re.I)
    b = re.findall("[0-9]+", s)
    c = re.sub(r"\\s+", " ", s)
    d = re.search("x", s, flags)  # flags from a variable: must stay
    return a, b, c, d
"""


def _load(path: Path):  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_regex_patch_same_behaviour(tmp_path: Path) -> None:
    src = tmp_path / "hotmod.py"
    src.write_text(HOT, "utf-8")
    patched = codeopt._regex_patch(src, {"hot"})
    assert patched is not None
    new_src, desc = patched
    assert "3 regex" in desc
    assert new_src.count("re.compile(") == 3
    assert 're.search("x", s, flags)' in new_src  # variable flags untouched
    out = tmp_path / "hotmod2.py"
    out.write_text(new_src, "utf-8")
    old, new = _load(src), _load(out)
    for s in ("xxABBBC 12 34   y", "nothing here", "abc\tabc 7"):
        a0, b0, c0, d0 = old.hot(s, re.I)
        a1, b1, c1, d1 = new.hot(s, re.I)
        assert (a0 and a0.group()) == (a1 and a1.group())
        assert (b0, c0) == (b1, c1)
        assert (d0 and d0.group()) == (d1 and d1.group())


def test_regex_patch_ignores_cold_functions(tmp_path: Path) -> None:
    src = tmp_path / "cold.py"
    src.write_text(HOT, "utf-8")
    assert codeopt._regex_patch(src, {"other"}) is None


# --- per-prompt learning ----------------------------------------------------------------


def test_feedback_learns_a_correction(tmp_path: Path) -> None:
    fb, model = tmp_path / "fb.jsonl", tmp_path / "intent.joblib"
    q = "zorblat the quux"  # nonsense the templates never contain
    target = feedback.intents()[0]
    feedback.record(q, "help", False, target, path=fb)
    res = feedback.quick_update(path=fb, model_path=model)
    assert res["status"] == "promoted"  # no champion yet
    import joblib

    assert joblib.load(model).predict([q])[0] == target
    again = feedback.quick_update(path=fb, model_path=model)
    assert again["status"] == "promoted"  # same data: not worse -> kept
    assert (tmp_path / "intent_prev.joblib").is_file()


def test_feedback_rejects_unknown_intent(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        feedback.record("q", "help", False, "not-an-intent", path=tmp_path / "f.jsonl")


def test_thumbs_down_without_correction_is_review_only(tmp_path: Path) -> None:
    fb = tmp_path / "f.jsonl"
    feedback.record("q", "help", False, None, path=fb)
    assert feedback.labelled(fb) == ([], [])
    assert (
        feedback.quick_update(path=fb, model_path=tmp_path / "m.joblib")["status"]
        == "nothing to learn"
    )


# --- NetFlow / Zeek update helpers -------------------------------------------------------


def test_binary_labels() -> None:
    assert _binary_labels(pd.Series([0, 1, 3])).tolist() == [0, 1, 1]
    assert _binary_labels(pd.Series(["BENIGN", "DDoS", "normal"])).tolist() == [0, 1, 0]


def test_zeek_rows_iot23_layout(tmp_path: Path) -> None:
    log = tmp_path / "conn.log.labeled"
    log.write_text(
        "#separator \\x09\n"
        "#fields\tts\tuid\tid.orig_h\tproto\ttunnel_parents   label   detailed-label\n"
        "1.0\tC1\t10.0.0.1\ttcp\t-   Malicious   PartOfAHorizontalPortScan\n"
        "2.0\tC2\t10.0.0.2\tudp\t-   Benign   -\n",
        "utf-8",
    )
    rows = zeek_rows(log)
    assert [r["label"] for r in rows] == ["Malicious", "Benign"]
    assert rows[0]["proto"] == "tcp"
