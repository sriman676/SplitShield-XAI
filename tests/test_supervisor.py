"""Unattended-run rules (scripts/supervisor.py): OOM detection, shrink, judgement."""

from __future__ import annotations

import json

import scripts.supervisor as wd


def _wire(tmp_path, monkeypatch, engine: str, run: str) -> None:
    for name, val in (
        ("ENGINE_LOG", tmp_path / "ai_engineer.log"),
        ("RUN_LOG", tmp_path / "full_run.log"),
        ("LOG", tmp_path / "watchdog.log"),
        ("STATE", tmp_path / "state.json"),
        ("STATUS_MD", tmp_path / "STATUS.md"),
        ("MEM_JSON", tmp_path / "mem.json"),
        ("MEM_MD", tmp_path / "MEM.md"),
        ("QUEUE", tmp_path / "queue.json"),
    ):
        monkeypatch.setattr(wd, name, val)
    (tmp_path / "ai_engineer.log").write_text(engine, "utf-8")
    (tmp_path / "full_run.log").write_text(run, "utf-8")


def _st() -> dict:
    return {"status": "running", "attempts": {}, "idle_launches": 0, "launches": 0}


def test_dataset_of_and_oom_regex() -> None:
    assert wd.dataset_of("train:E4_cut_depth:seed0") == "cicids2017-improved"
    assert wd.dataset_of("e3_matrix") == "cicids2017-improved"
    assert wd.dataset_of("improve:cic-iot-2023") == "cic-iot-2023"
    assert wd.dataset_of("nlp") is None
    assert wd.dataset_of("followup:C_H5_original:seed3") == "cicids2017-original"
    assert wd.OOM.search("numpy.core._exceptions._ArrayMemoryError: Unable to allocate")
    assert wd.OOM.search("FAIL  x (exit 3221225477)")
    assert not wd.OOM.search("Stage 4 FAILED: ValueError: bad shape")


def test_oom_failure_shrinks_then_retries(tmp_path, monkeypatch) -> None:
    eng = "t begin (full)\nt FAILED phases (re-run to retry only these): train:R1_ciciot2023:seed0\nt finished in 9 min\n"
    run = "[ai-engineer] begin (full)\n[ai-engineer] START train:R1_ciciot2023:seed0\nMemoryError\n"
    _wire(tmp_path, monkeypatch, eng, run)
    st = _st()
    assert wd.judge(st) is True  # retry warranted
    assert json.loads((tmp_path / "mem.json").read_text())["datasets"] == {
        "cic-iot-2023": 0.5
    }
    assert st["attempts"]["train:R1_ciciot2023:seed0"] == 2
    assert wd.judge(st) is True and wd.judge(st) is False  # 3 attempts max (R4)
    assert (
        json.loads((tmp_path / "mem.json").read_text())["datasets"]["cic-iot-2023"]
        == 0.25
    )
    assert st["status"] == "finished_with_failures"


def test_fatal_halts_and_crash_resumes(tmp_path, monkeypatch) -> None:
    _wire(
        tmp_path,
        monkeypatch,
        "t begin (full)\nt check: tests\n",
        "[ai-engineer] begin (full)\n[ai-engineer] tests check failed - not starting training.\n",
    )
    st = _st()
    assert wd.judge(st) is False and st["status"] == "halted"
    _wire(
        tmp_path,
        monkeypatch,
        "t begin (full)\nt START train:x:seed0\n",
        "[ai-engineer] begin (full)\n",
    )
    assert wd.judge(_st()) is True  # interrupted run -> resume
    _wire(
        tmp_path, monkeypatch, "t begin (full)\nt finished in 1 min\n", "begin (full)\n"
    )
    st = _st()
    assert wd.judge(st) is False and st["status"] == "done"


def test_queued_work_relaunches_until_done(tmp_path, monkeypatch) -> None:
    """R9: a clean finish with queued phases still open -> relaunch, then done."""
    eng = "t begin (full)\nt finished in 1 min\n"
    _wire(tmp_path, monkeypatch, eng, "begin (full)\n")
    queue = {"phases": ["followup:C_x:seed0"]}
    (tmp_path / "queue.json").write_text(json.dumps(queue))
    done: set[str] = set()
    monkeypatch.setattr(wd, "_done_phases", lambda: done)
    st = _st()
    assert wd.judge(st) is True and st["status"] == "running"
    done.add("followup:C_x:seed0")
    assert wd.judge(st) is False and st["status"] == "done"
