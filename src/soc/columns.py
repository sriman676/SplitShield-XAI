"""Match a flow table's columns to the detector's features (owner 2026-10-14).

CICFlowMeter has been released with several naming styles for the same features:
CIC-IDS2017 ("Total Length of Fwd Packets", "Init_Win_bytes_forward"), CIC-IDS2018
("TotLen Fwd Pkts", "Init Fwd Win Byts") and CICFlowMeter v4 ("Total Length of Fwd
Packet", "FWD Init Win Bytes"). Each name is reduced to a set of canonical words
(abbreviations expanded, plural and word order ignored), so all three styles land on
the same key. NetFlow exports (IN_BYTES, OUT_PKTS, ...) describe flows differently
(no inter-arrival timing) and cannot be converted; they are recognised and explained.
"""

from __future__ import annotations

import re
from typing import Any

# abbreviation -> canonical word(s)
_WORDS: dict[str, tuple[str, ...]] = {
    "tot": ("total",),
    "totlen": ("total", "length"),
    "len": ("length",),
    "pkt": ("packet",),
    "pkts": ("packet",),
    "byt": ("byte",),
    "byts": ("byte",),
    "b": ("bulk",),  # "Fwd Byts/b Avg" = bytes per bulk
    "blk": ("bulk",),
    "cnt": ("count",),
    "var": ("variance",),
    "seg": ("segment",),
    "average": ("avg",),
    "forward": ("fwd",),
    "backward": ("bwd",),
    "destination": ("dst",),
    "source": ("src",),
    "cwe": ("cwr",),  # CIC-IDS2017 typo of the CWR flag
    "of": (),
    "the": (),
}

NETFLOW_HINTS = {"IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "L4_DST_PORT"}
FLOWMETER_HINTS = {"fwd_pkts_tot", "bwd_pkts_tot", "flow_iat.avg", "fwd_init_window_size", "flow_duration"}

# Zeek FlowMeter (github.com/zeek-flowmeter, e.g. the HIKARI datasets) -> CIC-IDS2017 names.
# Units from the FlowMeter documentation (owner 2026-10-24): flow_duration in SECONDS
# (CICFlowMeter: microseconds); inter-arrival, active and idle times in microseconds (same
# as CICFlowMeter); sizes in payload bytes; rates per second. A str is one source column;
# a tuple is a sum; ("sq", col) is the square (variance from std).
FLOWMETER: dict[str, Any] = {
    "Destination Port": "responp", "Total Fwd Packets": "fwd_pkts_tot", "Total Backward Packets": "bwd_pkts_tot",
    "Total Length of Fwd Packets": "fwd_pkts_payload.tot", "Total Length of Bwd Packets": "bwd_pkts_payload.tot",
    "Fwd Packet Length Max": "fwd_pkts_payload.max", "Fwd Packet Length Min": "fwd_pkts_payload.min",
    "Fwd Packet Length Mean": "fwd_pkts_payload.avg", "Fwd Packet Length Std": "fwd_pkts_payload.std",
    "Bwd Packet Length Max": "bwd_pkts_payload.max", "Bwd Packet Length Min": "bwd_pkts_payload.min",
    "Bwd Packet Length Mean": "bwd_pkts_payload.avg", "Bwd Packet Length Std": "bwd_pkts_payload.std",
    "Flow Bytes/s": "payload_bytes_per_second", "Flow Packets/s": "flow_pkts_per_sec",
    "Flow IAT Mean": "flow_iat.avg", "Flow IAT Std": "flow_iat.std", "Flow IAT Max": "flow_iat.max",
    "Flow IAT Min": "flow_iat.min", "Fwd IAT Total": "fwd_iat.tot", "Fwd IAT Mean": "fwd_iat.avg",
    "Fwd IAT Std": "fwd_iat.std", "Fwd IAT Max": "fwd_iat.max", "Fwd IAT Min": "fwd_iat.min",
    "Bwd IAT Total": "bwd_iat.tot", "Bwd IAT Mean": "bwd_iat.avg", "Bwd IAT Std": "bwd_iat.std",
    "Bwd IAT Max": "bwd_iat.max", "Bwd IAT Min": "bwd_iat.min", "Fwd PSH Flags": "fwd_PSH_flag_count",
    "Bwd PSH Flags": "bwd_PSH_flag_count", "Fwd URG Flags": "fwd_URG_flag_count", "Bwd URG Flags": "bwd_URG_flag_count",
    "Fwd Header Length": "fwd_header_size_tot", "Bwd Header Length": "bwd_header_size_tot",
    "Fwd Packets/s": "fwd_pkts_per_sec", "Bwd Packets/s": "bwd_pkts_per_sec",
    "Min Packet Length": "flow_pkts_payload.min", "Max Packet Length": "flow_pkts_payload.max",
    "Packet Length Mean": "flow_pkts_payload.avg", "Packet Length Std": "flow_pkts_payload.std",
    "Packet Length Variance": ("sq", "flow_pkts_payload.std"), "FIN Flag Count": "flow_FIN_flag_count",
    "SYN Flag Count": "flow_SYN_flag_count", "RST Flag Count": "flow_RST_flag_count",
    "PSH Flag Count": ("fwd_PSH_flag_count", "bwd_PSH_flag_count"), "ACK Flag Count": "flow_ACK_flag_count",
    "URG Flag Count": ("fwd_URG_flag_count", "bwd_URG_flag_count"), "CWE Flag Count": "flow_CWR_flag_count",
    "ECE Flag Count": "flow_ECE_flag_count", "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "flow_pkts_payload.avg", "Avg Fwd Segment Size": "fwd_pkts_payload.avg",
    "Avg Bwd Segment Size": "bwd_pkts_payload.avg", "Fwd Avg Bytes/Bulk": "fwd_bulk_bytes",
    "Fwd Avg Packets/Bulk": "fwd_bulk_packets", "Fwd Avg Bulk Rate": "fwd_bulk_rate",
    "Bwd Avg Bytes/Bulk": "bwd_bulk_bytes", "Bwd Avg Packets/Bulk": "bwd_bulk_packets",
    "Bwd Avg Bulk Rate": "bwd_bulk_rate", "Subflow Fwd Packets": "fwd_subflow_pkts",
    "Subflow Fwd Bytes": "fwd_subflow_bytes", "Subflow Bwd Packets": "bwd_subflow_pkts",
    "Subflow Bwd Bytes": "bwd_subflow_bytes", "Init_Win_bytes_forward": "fwd_init_window_size",
    "Init_Win_bytes_backward": "bwd_init_window_size", "act_data_pkt_fwd": "fwd_data_pkts_tot",
    "min_seg_size_forward": "fwd_header_size_min", "Active Mean": "active.avg", "Active Std": "active.std",
    "Active Max": "active.max", "Active Min": "active.min", "Idle Mean": "idle.avg", "Idle Std": "idle.std",
    "Idle Max": "idle.max", "Idle Min": "idle.min",
}  # fmt: skip


def is_flowmeter(columns: list[str]) -> bool:
    return len({str(c).strip() for c in columns} & FLOWMETER_HINTS) >= 4


def _seconds(v: Any) -> Any:  # noqa: ANN401 - pandas Series in / out
    """flow_duration as seconds: a number, or a timedelta text ('0 days 00:00:00.000060')."""
    import pandas as pd

    num = pd.to_numeric(v, errors="coerce")
    if num.notna().mean() > 0.5:
        return num
    return pd.to_timedelta(v, errors="coerce").dt.total_seconds()


def from_flowmeter(df: Any) -> Any:  # noqa: ANN401 - pandas DataFrame in / out
    """A Zeek FlowMeter table -> the same flows with CICFlowMeter (CIC-IDS2017) columns;
    the original columns stay (labels, addresses)."""
    import pandas as pd

    df = df.rename(columns=lambda c: str(c).strip())
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(float("nan"), index=df.index)  # noqa: E731
    out = {"Flow Duration": _seconds(df["flow_duration"]) * 1e6 if "flow_duration" in df.columns
           else pd.Series(float("nan"), index=df.index)}  # fmt: skip
    for cic, src in FLOWMETER.items():
        if isinstance(src, str):
            out[cic] = num(src)
        elif src[0] == "sq":
            out[cic] = num(src[1]) ** 2
        else:
            out[cic] = sum(num(s) for s in src)
    conv = pd.DataFrame(out, index=df.index)
    keep = df[[c for c in df.columns if c not in conv.columns]]
    return pd.concat([keep, conv], axis=1)


def _singular(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def canon(name: str) -> frozenset[str]:
    """Order-free word set for a column name ("Tot Fwd Pkts" == "Total Fwd Packets")."""
    words: list[str] = []
    for w in re.split(r"[^a-z0-9]+", str(name).strip().lower()):
        if not w:
            continue
        w = w if w in _WORDS else _singular(w)
        words += list(_WORDS.get(w, (w,)))
    return frozenset(_singular(w) for w in words)


def detect_format(columns: list[str]) -> str:
    cols = {str(c).strip() for c in columns}
    if len(cols & NETFLOW_HINTS) >= 3:
        return "NetFlow v9 (NF-* datasets)"
    if is_flowmeter(list(cols)):
        return "Zeek FlowMeter (converted to CICFlowMeter units)"
    if "Init_Win_bytes_forward" in cols or "Total Length of Fwd Packets" in cols:
        return "CICFlowMeter (CIC-IDS2017 names)"
    if "TotLen Fwd Pkts" in cols or "Tot Fwd Pkts" in cols:
        return "CICFlowMeter (CIC-IDS2018 names)"
    if "Total Fwd Packet" in cols or "FWD Init Win Bytes" in cols:
        return "CICFlowMeter v4 (2023/2024 names)"
    return "unknown"


def match(columns: list[str], needed: list[str]) -> dict[str, Any]:
    """Map each needed feature to a column of the table (exact name first, then the
    canonical word set). Returns the rename map, what is missing, and the format."""
    cols = [str(c).strip() for c in columns]
    by_key: dict[frozenset[str], str] = {}
    for c in cols:
        by_key.setdefault(canon(c), c)  # first copy wins ("Fwd Header Length.1")
    rename: dict[str, str] = {}
    missing: list[str] = []
    for n in needed:
        if n in cols:
            continue
        hit = by_key.get(canon(n))
        if hit is not None and hit not in rename and hit not in needed:
            rename[hit] = n
        else:
            missing.append(n)
    found = len(needed) - len(missing)
    return {
        "format": detect_format(cols),
        "rename": rename,
        "missing": missing,
        "matched": found,
        "needed": len(needed),
        "share": found / max(1, len(needed)),
    }


def align(df: Any, needed: list[str]) -> tuple[Any, dict[str, Any]]:
    """``df`` with stripped, renamed columns that line up with ``needed`` (a Zeek
    FlowMeter table is converted first)."""
    df = df.rename(columns=lambda c: str(c).strip())
    if is_flowmeter(list(df.columns)):
        df = from_flowmeter(df)
    info = match(list(df.columns), needed)
    return df.rename(columns=info["rename"]), info


def explain(info: dict[str, Any]) -> str:
    """Plain-English verdict for the dashboard."""
    if info["format"].startswith("NetFlow"):
        return (
            "This is NetFlow data. The deployed detector reads CICFlowMeter features "
            "(packet timing, flags, bulk rates) that NetFlow does not record, so these "
            "rows cannot be scored by it. Export the traffic with CICFlowMeter instead, "
            "or train a NetFlow model (python -m scripts.extra_datasets)."
        )
    if info["share"] >= 0.8:
        extra = (
            f" {len(info['missing'])} missing feature(s) are filled with typical "
            "training values."
            if info["missing"]
            else ""
        )
        return f"Usable: {info['matched']} of {info['needed']} features matched.{extra}"
    return (
        f"Only {info['matched']} of {info['needed']} features matched (80% needed). "
        "Export the flows with CICFlowMeter; missing e.g.: "
        + ", ".join(info["missing"][:6])
    )
