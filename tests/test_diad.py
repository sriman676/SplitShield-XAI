"""CIC IoT-DIAD 2024 wiring: header-less read, folder labels, ciot mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data import labels as L
from src.data.sources import (
    _CICFLOWMETER_84,
    DIAD_CLASS_DIRS,
    _label_from_path,
    _read_frame,
    get_spec,
)


def test_spec_registered() -> None:
    spec = get_spec("cic-iot-diad-2024")
    assert spec.read_header == 0  # headered; names overridden
    assert spec.label_from_path and spec.label_scheme == "ciot"
    assert len(_CICFLOWMETER_84) == 84 and _CICFLOWMETER_84[-1] == "NeedManualLabel"


def test_folder_dirs_map_to_ciot() -> None:
    # every DIAD top-level class dir resolves under the ciot scheme
    expected = {
        "Benign": "BENIGN",
        "Brute Force": "BruteForce",
        "DDOS": "DDoS",
        "DOS": "DoS",
        "Mirai": "Mirai",
        "Recon": "Recon",
        "Spoofing": "Spoofing",
        "Web-Based": "WebBased",
    }
    for d in DIAD_CLASS_DIRS:
        assert L.to_ciot(d) == expected[d]


def test_label_from_path_matches_nested_class_dir() -> None:
    base = Path("data/raw/cic-iot-diad-2024")
    p = base / "IoT dataset" / "DDOS" / "DDOS ICMP FLOOD" / "x.pcap_Flow.csv"
    assert _label_from_path(p, base) == "DDOS"


def test_label_from_path_unknown_fails_loud() -> None:
    base = Path("data/raw/cic-iot-diad-2024")
    with pytest.raises(ValueError):
        _label_from_path(base / "Mystery" / "x.csv", base)


def test_headered_read_overrides_names(tmp_path: Path) -> None:
    """Real DIAD CSVs carry a header (last column literally 'Label'); the header
    row is consumed, names are overridden, and a wrong width fails loud."""
    header = ",".join([*_CICFLOWMETER_84[:-1], "Label"])
    row = ",".join(["1"] * 83 + ["NeedManualLabel"])
    f = tmp_path / "one.csv"
    f.write_text(f"{header}\n{row}\n", encoding="utf-8")
    df = _read_frame(f, get_spec("cic-iot-diad-2024"))
    assert list(df.columns) == list(_CICFLOWMETER_84)
    assert len(df) == 1  # header consumed, one data row
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 84"):
        _read_frame(bad, get_spec("cic-iot-diad-2024"))
