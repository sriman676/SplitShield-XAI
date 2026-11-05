"""Power-cut recovery: run_pipeline --resume continues the unfinished run in place."""

from __future__ import annotations

import json
from pathlib import Path

import run_pipeline as rp
import utils.paths
from config.base import build_config, config_hash, load_experiment


def test_resumable_finds_unfinished_run_and_done_stages(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    monkeypatch.setattr(utils.paths, "RUNS_DIR", tmp_path)
    cfg = build_config(load_experiment("E4_cut_depth.yaml"), seed=2)
    h = config_hash(cfg)
    old = tmp_path / f"E4_cut_depth__seed2__20261013T0100Z__{h}"
    new = tmp_path / f"E4_cut_depth__seed2__20261014T0100Z__{h}"
    other = tmp_path / f"E4_cut_depth__seed3__20261014T0200Z__{h}"
    for d in (old, new, other):
        (d / "metrics").mkdir(parents=True)
    (old / "run_complete.json").write_text("{}")  # finished: never reused
    for f in ("stage1_baseline.json", "stage2_arms.json", "stage4_failed.json"):
        (new / "metrics" / f).write_text("{}")
    found = rp._resumable(cfg, "")
    assert found is not None
    rid, done = found
    assert rid == new.name and done == {1, 2}  # failed stage 4 is redone
    (new / "run_complete.json").write_text("{}")
    assert rp._resumable(cfg, "") is None


def test_note_resume_records_in_manifest(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(rp, "run_dir", lambda rid: tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "x"}))
    rp._note_resume("x", [1, 2, 3])
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["resumed"][0]["stages_already_done"] == [1, 2, 3]
