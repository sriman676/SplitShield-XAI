"""Flow sources for the assistant: replay a CSV/pcap 'as if live', or sniff live.

Two generators, both yielding ``(meta, feature_row)`` where ``feature_row`` is a
dict of ``feature_name -> float`` (the assistant maps it onto the model's columns):

  * ``replay_flows``  - stream rows from a CSV of flows at a chosen rate. Deterministic,
    no privileges, guaranteed feature parity. Good for demos and the adversarial sets.
  * ``live_flows``    - sniff a real interface via the ``cicflowmeter`` Python port
    (needs Npcap + admin on Windows). Best-effort feature parity with the training
    schema; missing features are filled with 0 by the assistant.

Keeping capture separate from scoring means the assistant does not care where a flow
came from.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from utils.logging_config import get_logger

_log = get_logger(__name__)

Flow = tuple[dict[str, Any], dict[str, float]]


def replay_flows(
    csv_path: str | Path, rate_hz: float = 0.0, limit: int | None = None
) -> Iterator[Flow]:
    """Yield flows from a CSV as if arriving live.

    ``rate_hz`` > 0 sleeps between rows to simulate real time; 0 means as-fast-as-
    possible. ``limit`` caps how many rows are streamed.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"replay source not found: {path}")
    delay = 1.0 / rate_hz if rate_hz > 0 else 0.0
    df = pd.read_csv(path, nrows=limit) if limit else pd.read_csv(path)
    for i, row in df.iterrows():
        meta = {"source": str(path.name), "index": int(i)}
        feats = {k: _to_float(v) for k, v in row.items()}
        yield meta, feats
        if delay:
            time.sleep(delay)


def live_flows(
    interface: str | None = None, limit: int | None = None
) -> Iterator[Flow]:
    """Sniff a live interface and yield CICFlowMeter-style flows.

    Requires the optional ``cicflowmeter`` package + Npcap (Windows) / libpcap (Linux),
    and administrator/root privileges. Raises ``RuntimeError`` with an install hint if
    the dependency is missing so the caller can fall back to replay.
    """
    try:
        from cicflowmeter.sniffer import create_sniffer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "live capture needs the 'cicflowmeter' package and Npcap/libpcap. "
            "Install: pip install cicflowmeter ; then run as admin. "
            "Or use replay mode instead."
        ) from exc

    # cicflowmeter writes finished flows to a callback/CSV; we bridge it to a queue.
    import queue

    q: queue.Queue = queue.Queue()

    class _Sink:
        def write(self, flow: dict) -> None:  # noqa: D401 - duck-typed writer
            q.put(flow)

    sniffer = create_sniffer(
        input_interface=interface, output_mode="flow", output=_Sink()
    )
    sniffer.start()
    _log.info("live capture started on interface=%s", interface)
    n = 0
    try:
        while limit is None or n < limit:
            flow = q.get()
            meta = {"source": "live", "interface": interface, "index": n}
            feats = {k: _to_float(v) for k, v in flow.items()}
            yield meta, feats
            n += 1
    finally:
        sniffer.stop()
        _log.info("live capture stopped")


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
