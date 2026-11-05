"""Fast mode overrides the batch size in child processes without editing research
files; normal mode is untouched; fast-mode runs get their own config hash."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from scripts import fast_mode

ROOT = Path(__file__).resolve().parents[1]
PROBE = ("from config.base import build_config, load_experiment, config_hash;"
         "c = build_config(load_experiment('T1_temporal.yaml'), seed=0);"
         "print(c.runtime.batch_size, config_hash(c))")  # fmt: skip


def _probe(env: dict[str, str]) -> tuple[int, str]:
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=300, check=True).stdout.split()  # fmt: skip
    return int(out[-2]), out[-1]


def test_fast_mode_changes_batch_and_hash_only_when_on() -> None:
    plain = {k: v for k, v in os.environ.items() if k not in ("SPLITSHIELD_FAST_BATCH",)}
    b0, h0 = _probe(plain)
    b1, h1 = _probe(fast_mode.env_for(512))
    assert b0 == 256 and b1 == 512
    assert h0 != h1  # never mixes with registered runs
    b2, h2 = _probe(plain)
    assert (b2, h2) == (b0, h0)  # normal mode unchanged


def test_refuses_the_original_project(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """The original is recognised by its git history; a copy (no .git) may run. Nothing is
    ever started here (subprocess.run is replaced), in the original or in a copy."""
    started: list[list[str]] = []

    def fake_run(cmd: list[str], **_kw: object) -> object:
        started.append(cmd)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(fast_mode.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["fast_mode", "--batch", "512", "--", "run_pipeline.py"])
    original = tmp_path / "original"
    (original / ".git").mkdir(parents=True)
    monkeypatch.chdir(original)
    try:
        fast_mode.main()
    except SystemExit as e:
        assert "copy" in str(e)
    else:
        raise AssertionError("ran in the original project")
    assert not started
    copy = tmp_path / "copy"
    copy.mkdir()
    monkeypatch.chdir(copy)
    assert fast_mode.main() == 0 and started and started[0][-1] == "run_pipeline.py"
