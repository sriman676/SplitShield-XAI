"""make_portable vs the re-run copy: wait while it runs; removing the copy never
touches the datasets its junctions point at."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import make_portable as mp


def test_busy_until_done(tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    assert not mp.rerun_busy(copy)
    copy.mkdir()
    assert mp.rerun_busy(copy)
    (copy / "RERUN_DONE.json").write_text("{}")
    assert not mp.rerun_busy(copy)


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
def test_remove_copy_keeps_junction_target(tmp_path: Path) -> None:
    import _winapi

    real = tmp_path / "real_raw"
    real.mkdir()
    (real / "flows.csv").write_text("a,b\n")
    copy = tmp_path / "copy"
    (copy / "data").mkdir(parents=True)
    _winapi.CreateJunction(str(real), str(copy / "data" / "raw"))
    (copy / "REPRODUCIBILITY.md").write_text("# report")
    docs = tmp_path / "docs"
    docs.mkdir()
    mp.remove_rerun_copy(copy, keep_in=docs)
    assert not copy.exists()
    assert (real / "flows.csv").is_file()
    assert (docs / "rerun_REPRODUCIBILITY.md").read_text() == "# report"
