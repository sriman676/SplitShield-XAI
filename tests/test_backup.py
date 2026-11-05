"""Daily backup (scripts/backup.py): copies the right folders, keeps the newest 7,
never touches the project or foreign folders on the backup drive."""

from __future__ import annotations

from pathlib import Path

from scripts import backup as b


def test_backup_copies_and_rotates(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    root = tmp_path / "proj"
    for d in ("artifacts/deploy", "results", "data/raw", ".venv"):
        (root / d).mkdir(parents=True)
    (root / "artifacts/deploy/model.pt").write_text("m")
    (root / "results/FINAL_REPORT.md").write_text("r")
    (root / "data/raw/huge.csv").write_text("x")
    (root / "run_pipeline.py").write_text("code")
    dest = tmp_path / "bk"
    for day in ("2026-10-01", "2026-10-02", "2026-10-03"):
        (dest / day).mkdir(parents=True)
    (dest / "my-own-files").mkdir()
    t = b.backup(dest, root, keep=2)
    assert (t / "artifacts/deploy/model.pt").read_text() == "m"
    assert (t / "results/FINAL_REPORT.md").is_file() and (
        t / "run_pipeline.py"
    ).is_file()
    assert not (t / "data").exists() and not (t / ".venv").exists()
    kept = sorted(p.name for p in dest.iterdir())
    assert kept == ["2026-10-03", t.name, "my-own-files"] or kept == sorted(
        [t.name, "2026-10-03", "my-own-files"]
    )
    assert (root / "data/raw/huge.csv").is_file()  # project untouched
