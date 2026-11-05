"""Column matching across CICFlowMeter naming styles + tolerant label mapping."""

from __future__ import annotations

import pandas as pd

from src.data.labels import FAMILY_ORDER
from src.soc.columns import canon, explain, match
from src.soc.engine import map_labels

NEED = [
    "Destination Port", "Total Fwd Packets", "Total Length of Fwd Packets",
    "Flow Bytes/s", "Fwd Avg Bytes/Bulk", "Init_Win_bytes_forward",
    "min_seg_size_forward", "Min Packet Length", "CWE Flag Count", "Average Packet Size",
]  # fmt: skip


def test_three_naming_styles_match() -> None:
    v2018 = ["Dst Port", "Tot Fwd Pkts", "TotLen Fwd Pkts", "Flow Byts/s",
             "Fwd Byts/b Avg", "Init Fwd Win Byts", "Fwd Seg Size Min", "Pkt Len Min",
             "CWE Flag Count", "Pkt Size Avg"]  # fmt: skip
    v4 = ["Dst Port", "Total Fwd Packet", "Total Length of Fwd Packet", "Flow Bytes/s",
          "Fwd Bytes/Bulk Avg", "FWD Init Win Bytes", "Fwd Seg Size Min",
          "Packet Length Min", "CWR Flag Count", "Average Packet Size"]  # fmt: skip
    for cols in (v2018, v4, [f" {c} " for c in NEED]):
        m = match(cols, NEED)
        assert m["missing"] == [] and m["share"] == 1.0, m["missing"]


def test_different_features_never_collide() -> None:
    assert canon("Packet Length Mean") != canon("Average Packet Size")
    assert canon("Fwd Header Length") != canon("Fwd Header Length.1")
    assert canon("Flow Bytes/s") != canon("Flow Packets/s")


def test_netflow_is_recognised_and_explained() -> None:
    m = match(["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "L4_DST_PORT"], NEED)
    assert m["format"].startswith("NetFlow") and "NetFlow" in explain(m)
    assert m["share"] == 0


def test_unknown_labels_are_skipped_not_fatal() -> None:
    raw = pd.Series(["BENIGN", "DoS Hulk", "Totally Novel", "BENIGN"])
    mapped, unknown = map_labels(raw, list(FAMILY_ORDER))
    assert mapped.tolist()[:2] == ["BENIGN", "DoS"] and pd.isna(mapped.iloc[2])
    assert unknown == ["Totally Novel"]
