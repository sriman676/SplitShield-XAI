"""Dataset intake: signature identification, nested zips, zip-slip, corrupt archives."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.data import intake


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


IOT_CSV = b"Header_Length,Protocol Type,Tot size,Label\n1,6,60,DDoS-ICMP\n"


def test_identify_by_columns() -> None:
    assert (
        intake.identify(["Header_Length", "Protocol Type", "Tot size"], "x.csv")
        == "cic-iot-2023"
    )
    assert (
        intake.identify(["Destination_Port", "Flow_Duration"], "x.csv")
        == "cicids2017-improved"
    )
    assert (
        intake.identify([" Destination Port", " Flow Duration", " Label"], "x.csv")
        == "cicids2017-original"
    )
    nf = ["L4_SRC_PORT", "L4_DST_PORT", "IN_BYTES"]
    assert intake.identify(nf, "NF-UNSW-NB15-v2.parquet") == "nf-unsw-nb15-v2"
    assert intake.identify(nf, "netflow.parquet") is None  # ambiguous -> never guessed
    assert intake.identify(["a", "b"], "x.csv") is None


def test_renamed_and_nested_zip_lands_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    monkeypatch.setattr(intake, "RAW_DIR", raw)
    monkeypatch.setattr(intake, "REPORT", raw / ".intake.json")
    inner = _zip(tmp_path / "inner.zip", {"MERGED_CSV/Merged01.csv": IOT_CSV})
    src = tmp_path / "src"
    src.mkdir()
    outer = _zip(
        src / "my_download.zip",
        {"pack/inner.zip": inner.read_bytes(), "pack/Merged02.csv": IOT_CSV},
    )
    rows = intake.run([src])
    assert rows[0]["dataset"] == "cic-iot-2023" and rows[0]["status"] == "ok (unpacked)"
    found = sorted(p.name for p in (raw / "cic-iot-2023").rglob("*.csv"))
    assert found == ["Merged01.csv", "Merged02.csv"]  # nested zip expanded too
    again = intake.run([src])  # unchanged archive -> skipped, not re-extracted
    assert again[0]["status"] == "ok (already unpacked)"
    assert outer.exists()


def test_zip_slip_and_corrupt_are_refused_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    monkeypatch.setattr(intake, "RAW_DIR", raw)
    monkeypatch.setattr(intake, "REPORT", raw / ".intake.json")
    src = tmp_path / "src"
    src.mkdir()
    _zip(src / "evil.zip", {"../../escape.csv": IOT_CSV})
    (src / "broken.zip").write_bytes(b"PK\x03\x04 not really a zip")
    _zip(src / "good.zip", {"Merged01.csv": IOT_CSV})
    rows = {Path(r["source"]).name: r for r in intake.run([src])}
    assert (
        rows["evil.zip"]["status"].startswith("FAILED")
        and "unsafe" in rows["evil.zip"]["status"]
    )
    assert rows["broken.zip"]["status"].startswith(("FAILED", "skipped"))
    assert rows["good.zip"]["status"] == "ok (unpacked)"  # others still processed
    assert not (tmp_path / "escape.csv").exists()


def test_several_archives_of_one_dataset_are_all_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    monkeypatch.setattr(intake, "RAW_DIR", raw)
    monkeypatch.setattr(intake, "REPORT", raw / ".intake.json")
    src = tmp_path / "src"
    src.mkdir()
    _zip(src / "part1.zip", {"Merged01.csv": IOT_CSV})
    _zip(src / "part2.zip", {"Merged02.csv": IOT_CSV})
    # archive.zip is Kaggle's default name: identified by its columns, not its name
    _zip(src / "archive.zip", {"Merged03.csv": IOT_CSV})
    rows = intake.run([src])
    assert {r["status"] for r in rows} == {"ok (unpacked)"}
    found = sorted(p.name for p in (raw / "cic-iot-2023").rglob("*.csv"))
    assert found == ["Merged01.csv", "Merged02.csv", "Merged03.csv"]
    assert not (raw / "cic-iot-diad-2024").exists()
