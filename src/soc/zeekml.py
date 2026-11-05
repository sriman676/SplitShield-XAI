"""Machine-learning detector for Zeek connection logs (owner 2026-10-15).

Per-connection features (protocol, service, bytes, packets, duration, TCP state and
history flags, destination-port class) plus what the SAME source did in the
previous 60 s (connections, distinct hosts / ports, failed attempts). Only past
events are used, so it works on a live stream; IP addresses themselves are never
features (no memorising infected hosts).

Trained and tested on IoT-23 (Stratosphere Lab) with a split BY CAPTURE: the test
captures are never seen in training (scripts/extra_logs.py --area zeek).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from utils.paths import PROJECT_ROOT

MODEL = PROJECT_ROOT / "artifacts" / "zeekml" / "model.joblib"
STATES = (
    "S0",
    "S1",
    "SF",
    "REJ",
    "S2",
    "S3",
    "RSTO",
    "RSTR",
    "RSTOS0",
    "RSTRH",
    "SH",
    "SHR",
    "OTH",
)
PROTOS = ("tcp", "udp", "icmp")
SERVICES = ("-", "dns", "http", "ssl", "ssh", "irc", "dhcp", "ntp", "ftp", "telnet")
HIST = "ShADadFfRrTtWw^"
PORTS = (
    22,
    23,
    53,
    80,
    123,
    443,
    445,
    1900,
    2323,
    5555,
    6667,
    7547,
    8080,
    8081,
    37215,
    52869,
)
WINDOW = 60.0
FEATURES = (
    [f"proto_{p}" for p in PROTOS] + [f"service_{s}" for s in SERVICES]
    + [f"state_{s}" for s in STATES] + [f"hist_{c}" for c in HIST]
    + [f"port_{p}" for p in PORTS]
    + ["port_wellknown", "port_registered", "port_ephemeral", "duration", "orig_bytes",
       "resp_bytes", "missed_bytes", "orig_pkts", "resp_pkts", "orig_ip_bytes",
       "resp_ip_bytes", "history_len", "bytes_ratio", "src_conns_60s",
       "src_hosts_60s", "src_ports_60s", "src_failed_60s"]
)  # fmt: skip


def _num(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


class Context:
    """Rolling 60-s per-source state (connections, hosts, ports, failures)."""

    def __init__(self) -> None:
        self.q: dict[str, deque[tuple[float, str, str, bool]]] = defaultdict(deque)

    def update(self, row: dict[str, Any]) -> tuple[float, float, float, float]:
        src, ts = str(row.get("id.orig_h", "")), _num(row.get("ts"))
        q = self.q[src]
        while q and ts - q[0][0] > WINDOW:
            q.popleft()
        out = (
            float(len(q)),
            float(len({h for _, h, _, _ in q})),
            float(len({p for _, _, p, _ in q})),
            float(sum(f for *_, f in q)),
        )
        failed = str(row.get("conn_state", "")) in ("S0", "REJ", "RSTOS0", "RSTR", "SH")
        q.append(
            (ts, str(row.get("id.resp_h", "")), str(row.get("id.resp_p", "")), failed)
        )
        return out


def features(row: dict[str, Any], ctx: Context) -> list[float]:
    proto, service = str(row.get("proto", "")), str(row.get("service", "-"))
    state, hist = str(row.get("conn_state", "")), str(row.get("history", ""))
    port = int(_num(row.get("id.resp_p")))
    ob, rb = _num(row.get("orig_bytes")), _num(row.get("resp_bytes"))
    x = [float(proto == p) for p in PROTOS]
    x += [float(service == s) for s in SERVICES]
    x += [float(state == s) for s in STATES]
    x += [float(hist.count(c)) for c in HIST]
    x += [float(port == p) for p in PORTS]
    x += [float(port < 1024), float(1024 <= port < 49152), float(port >= 49152)]
    x += [math.log1p(_num(row.get(k))) for k in ("duration",)]
    x += [math.log1p(ob), math.log1p(rb)]
    x += [math.log1p(_num(row.get(k))) for k in
          ("missed_bytes", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes")]  # fmt: skip
    x += [float(len(hist)), math.log1p(rb) - math.log1p(ob)]
    x += [math.log1p(v) for v in ctx.update(row)]
    return x


def matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    """Feature matrix in time order (the rolling context needs it)."""
    ctx = Context()
    order = sorted(range(len(rows)), key=lambda i: _num(rows[i].get("ts")))
    X = np.zeros((len(rows), len(FEATURES)), dtype=np.float32)
    for i in order:
        X[i] = features(rows[i], ctx)
    return X


_LOADED: dict[str, Any] = {}


def load() -> Any | None:
    if "m" not in _LOADED:
        if not Path(MODEL).is_file():
            return None
        import joblib

        _LOADED["m"] = joblib.load(MODEL)
    return _LOADED["m"]


_CTX = Context()


def score(row: dict[str, Any]) -> float | None:
    """Live scoring of one connection (probability it is malicious)."""
    bundle = load()
    if bundle is None:
        return None
    x = np.asarray([features(row, _CTX)], dtype=np.float32)
    return float(bundle["model"].predict_proba(x)[0, 1])


def technique(row: dict[str, Any], ctx_ports: float) -> str:
    """ATT&CK label for a flagged connection, from its shape."""
    if ctx_ports >= 10:
        return "T1046"  # many ports from one source: scanning
    if str(row.get("conn_state")) == "S0" and str(row.get("proto")) in ("tcp", "udp"):
        return "T1498"  # one-way floods
    return "T1071"  # repeated application-layer traffic: command and control
