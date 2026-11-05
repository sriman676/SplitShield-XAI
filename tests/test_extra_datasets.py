"""Extra datasets (owner 2026-10-14): native label schemes, class-capped streaming
reads, and the .evtx reader used by the log-mapping test."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import labels as L
from src.data import sources


@pytest.mark.parametrize(
    ("raw", "scheme", "want"),
    [
        ("scanning", "ton", "Scanning"),
        ("Benign", "ton", "BENIGN"),
        ("password", "ton", "Password"),
        ("Reconnaissance", "unsw", "Reconnaissance"),
        ("Benign", "unsw", "BENIGN"),
        ("DrDoS_LDAP", "ddos19", "LDAP"),
        ("UDP-lag", "ddos19", "UDPLag"),
        ("UDPLag", "ddos19", "UDPLag"),
        ("BENIGN", "ddos19", "BENIGN"),
    ],
)
def test_native_schemes(raw: str, scheme: str, want: str) -> None:
    assert L.to_native(raw, scheme) == want
    assert L.order_for(scheme)[0] == "BENIGN"  # BENIGN is class 0 everywhere


def test_native_unknown_fails_loud_and_old_schemes_unchanged() -> None:
    with pytest.raises(ValueError):
        L.to_native("Totally Novel", "ton")
    assert L.order_for("family") == L.FAMILY_ORDER
    assert L.order_for("binary") == L.BINARY_ORDER
    enc = L.encode_scheme(L.resolve_labels(pd.Series(["xss", "Benign"]), "ton"), "ton")
    assert enc.tolist() == [L.NATIVE_ORDERS["ton"].index("XSS"), 0]


def test_stream_sample_caps_each_class_keeps_rare(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    raw = tmp_path / "x"
    raw.mkdir()
    n = 3000
    df = pd.DataFrame({"a": np.arange(n + 40), " Label": ["Syn"] * n + ["BENIGN"] * 40})
    df.to_csv(raw / "f.csv", index=False)
    monkeypatch.setattr(sources, "RAW_DIR", tmp_path)
    spec = replace(
        sources.get_spec("cic-ddos2019"),
        id="x",
        id_columns=(),
        max_rows_per_file=500,
    )
    out = sources.load_raw(spec)
    counts = out["Label"].value_counts()
    assert counts["BENIGN"] == 40  # rare class kept whole
    assert 350 < counts["Syn"] < 650  # majority class capped near 500
    again = sources.load_raw(spec)
    assert out.equals(again)  # seeded: same rows every time


def test_unsw_multi_reads_the_e8_folder() -> None:
    spec = sources.get_spec("nf-unsw-nb15-v2-multi")
    assert spec.raw_id == "nf-unsw-nb15-v2" and spec.label_col == "Attack"
    assert sources.get_spec("nf-unsw-nb15-v2").label_scheme == "binary"  # E8 intact


@pytest.mark.skipif(sys.platform != "win32", reason="Get-WinEvent is Windows-only")
def test_read_evtx_missing_file_reports(tmp_path: Path) -> None:
    from src.soc.parsers import read_evtx

    with pytest.raises(RuntimeError):
        read_evtx(tmp_path / "nope.evtx")
