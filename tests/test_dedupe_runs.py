"""Retried runs: the newest attempt per stage is aggregated, once."""

from __future__ import annotations

from scripts.dedupe_runs import dedupe


def _run(root, name: str, files: list[str]) -> None:
    m = root / name / "metrics"
    m.mkdir(parents=True)
    for f in files:
        (m / f).write_text("{}", "utf-8")


def test_newest_attempt_wins_per_stage(tmp_path) -> None:
    _run(
        tmp_path,
        "H5__seed0__20261013T1651Z__abc",
        ["stage1.json", "stage5.json", "stage4_failed.json"],
    )
    _run(tmp_path, "H5__seed0__20261014T0300Z__abc", ["stage1.json", "stage4.json"])
    _run(tmp_path, "H5__seed1__20261013T1700Z__abc", ["stage1.json"])  # other seed
    _run(tmp_path, "E3__fedavg__a1p0__seed0__abc", ["e3_arm.json"])  # matrix cell
    _run(tmp_path, "E3__splitfed__a1p0__seed0__abc", ["e3_arm.json"])
    out = dedupe(tmp_path)
    assert out == ["H5__seed0__20261013T1651Z__abc/stage1.json"]
    old = tmp_path / "H5__seed0__20261013T1651Z__abc" / "metrics"
    assert (old / "stage1.json.superseded").is_file() and (
        old / "stage5.json"
    ).is_file()
    assert dedupe(tmp_path) == []  # idempotent
