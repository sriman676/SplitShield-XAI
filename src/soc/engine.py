"""SOC engine: detection over flows, log intake, incidents, SOC push, learning.

Storage (``artifacts/soc/``, JSON lines so the dashboard, the monitor process and a
SIEM agent can all read it without a database):
  events.jsonl  - every normalized event (log events + detector alerts)
  alerts.jsonl  - alerts raised for the analyst (detector + log events with ATT&CK)
  pushed.json   - incident ids already sent to the SOC, with the severity sent

The monitor (``python -m src.soc.engine monitor ...``) runs as its own process so the
dashboard stays responsive: it streams flows (CSV replay or live capture), watches the
inbox folder for exported logs, listens for syslog, raises alerts, and pushes new or
escalated incidents to the configured SOC.
"""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.soc import sinks
from src.soc.correlate import CLASS_TECHNIQUES, correlate, technique_info
from src.soc.parsers import Event, parse_line, parse_text
from src.soc.rcl import RCL
from src.soc.rcl import softmax as softmax_t
from utils.logging_config import configure_logging, get_logger
from utils.paths import PROJECT_ROOT

_log = get_logger(__name__)

SOC_DIR = sinks.SOC_DIR
EVENTS = SOC_DIR / "events.jsonl"
ALERTS = SOC_DIR / "alerts.jsonl"
PUSHED = SOC_DIR / "pushed.json"
STATUS = SOC_DIR / "monitor_status.json"
USER_DATA = PROJECT_ROOT / "data" / "user_datasets"

_SRC = ("Src IP", "Source IP", "src_ip", "Src_IP", "IPV4_SRC_ADDR")
_DST = ("Dst IP", "Destination IP", "dst_ip", "Dst_IP", "IPV4_DST_ADDR")
_PORT = ("Destination Port", "Dst Port", "dst_port", "L4_DST_PORT")
_TS = ("Timestamp", "timestamp", "ts", "FLOW_START_MILLISECONDS")
_lock = threading.Lock()


# --- storage -------------------------------------------------------------------


FLOW_SAMPLE = SOC_DIR / "flow_sample.jsonl"
SAMPLE_PER_BATCH = 200


def _sample_flows(X: np.ndarray | None, source: str) -> None:
    """Keep a random sample of every scored batch (ALL flows, not only alerts) for
    the drift monitor (src/soc/drift.py); the autopilot keeps it a rolling window."""
    if X is None or not len(X):
        return
    idx = np.random.default_rng().choice(
        len(X), min(len(X), SAMPLE_PER_BATCH), replace=False
    )
    now = time.time()
    _append(
        FLOW_SAMPLE,
        [{"ts": now, "source": source, "x": X[i].tolist()} for i in idx],
    )


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows as JSON lines. Several processes write the stores at once (live log
    watcher, monitor, dashboard uploads): a lock file serialises them across processes
    and each batch goes out in one write, so no line is ever torn or interleaved
    (2026-10-16 concurrency stress test: 66 torn lines in 60,000 without it)."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, default=str) + "\n" for r in rows).encode("utf-8")
    with _lock, _file_lock(path.with_name(path.name + ".lock")):
        with open(path, "ab") as fh:
            fh.write(data)


def _file_lock(lock_path: Path) -> Any:
    """Exclusive cross-process lock (Windows: msvcrt; elsewhere: fcntl)."""
    import contextlib
    import sys

    @contextlib.contextmanager
    def held() -> Any:
        fh = open(lock_path, "a+b")  # noqa: SIM115 - kept open while locked
        try:
            if sys.platform == "win32":
                import msvcrt

                while True:
                    try:
                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)  # retries ~10 s
                        break
                    except OSError:
                        continue  # another process holds it longer: keep waiting
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            fh.close()

    return held()


# path -> ((mtime_ns, size), rows, bytes parsed, first 256 bytes). The dashboard re-reads
# the stores every 3 s; an unchanged file is not re-parsed and a grown one only from
# where it was (append-only; a cleared or replaced file starts differently -> re-read).
_jsonl_cache: dict[Path, tuple[tuple[int, int], list[dict[str, Any]], int, bytes]] = {}


def read_jsonl(path: Path, last: int | None = None) -> list[dict[str, Any]]:
    try:
        st = path.stat()
    except OSError:
        return []
    key = (st.st_mtime_ns, st.st_size)
    hit = _jsonl_cache.get(path)
    if hit is None or hit[0] != key:
        # append-only: when the same file only grew, parse just the new lines
        with open(path, "rb") as fh:
            head = fh.read(256)
            old = hit if hit is not None and hit[2] <= st.st_size and hit[3] == head else None
            rows = list(old[1]) if old else []
            start = old[2] if old else 0
            fh.seek(start)
            data = fh.read()
        cut = data.rfind(b"\n") + 1
        tail = data[cut:].strip()
        if tail:  # a last line without newline: complete JSON counts, else it waits
            try:
                json.loads(tail)
                cut = len(data)
            except json.JSONDecodeError:
                pass
        for ln in data[:cut].decode("utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        hit = _jsonl_cache[path] = (key, rows, start + cut, head)
    return hit[1][-last:] if last else list(hit[1])


RECENT = 200_000  # rule tests and the place check look at the newest events only


def recent_events(n: int = RECENT) -> list[dict[str, Any]]:
    """The newest ``n`` stored events (bounded memory at any store size)."""
    import sqlite3

    from src.soc import eventindex

    try:
        return eventindex.recent(n)
    except (sqlite3.Error, OSError):
        return read_jsonl(EVENTS, last=n)


def events_between(start: float, end: float) -> list[dict[str, Any]]:
    """Stored events with start <= ts < end, in store order."""
    import sqlite3

    from src.soc import eventindex

    try:
        return eventindex.between(start, end)
    except (sqlite3.Error, OSError):
        return [e for e in read_jsonl(EVENTS) if start <= float(e.get("ts") or 0) < end]


def event_count() -> int:
    import sqlite3

    from src.soc import eventindex

    try:
        return eventindex.count()
    except (sqlite3.Error, OSError):
        return len(read_jsonl(EVENTS))


def clear_store() -> None:
    """Empty the event / alert stores. On Windows a file another part of the SOC
    still has open cannot be deleted (2026-10-16, every-button stress test): it is
    emptied instead - the index then sees a shrunk store and rebuilds itself."""
    from src.soc import eventindex

    eventindex.close()
    for p in (EVENTS, ALERTS, PUSHED):
        try:
            p.unlink(missing_ok=True)
        except PermissionError:
            p.write_text("", "utf-8")
    _jsonl_cache.clear()


# --- detector --------------------------------------------------------------------


def load_detector() -> Any | None:
    """The deployed detector (``deploy.py`` output), or None before training ends."""
    from src.soc.model import DEPLOY, Detector

    if not (DEPLOY / "model.pt").exists():
        return None
    try:
        return Detector()
    except (OSError, KeyError, ValueError, RuntimeError) as exc:  # half-written deploy
        _log.warning("detector artifacts unreadable (%s); flow detection disabled", exc)
        return None


def flow_logits(det: Any, df: pd.DataFrame) -> np.ndarray:
    import torch

    names = det.feature_names
    stats = getattr(det, "input_stats", None)
    missing = [n for n in names if n not in df.columns]
    frame = (
        df.reindex(columns=names)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    # same cleaning as training: missing / NaN / Inf -> the training median
    # (0 on older artifacts without input stats)
    fill = dict(zip(names, stats["median"], strict=True)) if stats else 0
    X = frame.fillna(fill).to_numpy(np.float32)
    det.last_X = X  # raw, cleaned inputs (the drift sample is drawn from these)
    if stats and len(X):
        from src.data.robust import input_check

        chk = input_check(X, stats, missing=len(missing), n_features=len(names))
        det.last_input_check = chk
        for w in chk["warnings"]:
            _log.warning("live input check: %s", w)
    Xs = det.scaler.transform(X).astype(np.float32)
    det.last_Xs = Xs  # scaled inputs, for per-flow "why" attributions
    det.model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(Xs)
        mlp = det.model(xt).numpy()
        ens = getattr(det, "ensemble", None)
        # ensemble -> log of the averaged probabilities; the MLP's own logits are
        # kept for the D3 energy score (constant on log-probabilities)
        det.last_mlp_logits = mlp
        return ens(xt).numpy() if ens is not None else mlp


def is_flow_table(det: Any, df: pd.DataFrame) -> bool:
    """A CICFlowMeter table in any naming style (2017 / 2018 / v4, src/soc/columns)."""
    from src.soc.columns import is_flowmeter, match

    return det is not None and (
        is_flowmeter(list(df.columns)) or match(list(df.columns), det.feature_names)["share"] >= 0.5
    )


def _first(row: pd.Series, names: tuple[str, ...]) -> Any:
    for n in names:
        if n in row and pd.notna(row[n]) and str(row[n]).strip():
            return row[n]
    return None


def detect_flows(
    det: Any, df: pd.DataFrame, source: str, rcl: RCL | None = None
) -> list[Event]:
    """Score a flow table; return alert events for flows that pass the RCL gate."""
    from src.soc.columns import align

    rcl = rcl or RCL()
    df, _ = align(df, det.feature_names)  # other CICFlowMeter naming styles
    logits = flow_logits(det, df)
    _sample_flows(getattr(det, "last_X", None), source)
    # analyst-feedback calibration once it exists, else the deploy-time fit
    probs = (
        rcl.calibrated(logits)
        if rcl.history
        else softmax_t(logits, getattr(det, "temperature", 1.0))
    )
    now = time.time()
    out: list[Event] = []
    from src.defense.detector import energy_from_logits
    from src.soc.parsers import _ts

    try:  # per-site calibration sample (src/soc/sitecal.py); never stops detection
        from src.soc import sitecal

        b = list(det.label_names).index("BENIGN")
        sitecal.observe(1.0 - probs[:, b])
    except Exception:  # noqa: BLE001
        pass
    mlp_logits = getattr(det, "last_mlp_logits", None)
    en = energy_from_logits(
        mlp_logits
        if mlp_logits is not None and len(mlp_logits) == len(logits)
        else logits
    )
    thr = getattr(det, "energy_threshold", None)
    names = list(det.label_names)
    benign = names.index("BENIGN") if "BENIGN" in names else None
    rows_idx: list[int] = []  # flow row of each alert, for per-flow explanations
    for i, (lg, p) in enumerate(zip(logits, probs, strict=True)):
        # Detect on the TOTAL attack probability, name the likeliest attack class:
        # BENIGN 0.2 / DoS 0.4 / DDoS 0.4 is 80 % attack, but no single class
        # reached the threshold, so deciding on the top class alone missed it.
        attack = p.copy()
        if benign is not None:
            attack[benign] = -1.0
        idx = int(attack.argmax())
        cls, class_conf = names[idx], float(p[idx])
        qhat = getattr(det, "conformal_qhat", None)
        pset = [
            names[c] for c in range(len(p)) if qhat is not None and 1 - p[c] <= qhat
        ]
        unsure = qhat is not None and len(pset) != 1
        conf = 1.0 - float(p[benign]) if benign is not None else class_conf
        alert = rcl.should_alert(cls, conf)
        # D3 runtime detector: input unlike any clean training traffic, even if
        # the classifier calls it benign -> possible evasion / manipulated flow.
        suspicious = thr is not None and bool(en[i] > thr)
        if not alert and not suspicious:
            continue
        row = df.iloc[i]
        ts_raw = _first(row, _TS)
        # Two-tier alerting (owner 2026-10-13): above the deploy-time threshold for
        # ~1% false alarms -> a normal alert whose ATT&CK techniques drive incident
        # severity and SIEM pushes; between 0.5 and that -> recorded as a LOW
        # severity "possible" alert, techniques kept only as candidates.
        high = getattr(det, "alert_threshold_high", None)
        by_class = getattr(det, "alert_threshold_by_class", None) or {}
        high = by_class.get(cls, high)  # per-class threshold when deployed with one
        confirmed = alert and (high is None or conf >= high)
        mapped = CLASS_TECHNIQUES.get(cls, []) if alert else []
        techs = mapped if confirmed else []
        port = _first(row, _PORT)
        out.append(
            {
                "id": uuid.uuid4().hex[:12],
                "ts": _ts(ts_raw) if ts_raw is not None else now + i * 1e-3,
                "src_ip": None if (v := _first(row, _SRC)) is None else str(v),
                "dst_ip": None if (v := _first(row, _DST)) is None else str(v),
                "dst_port": None if port is None else int(_num(port)),
                "source": "detector",
                "kind": (
                    ("flow_detection" if confirmed else "flow_detection_low")
                    if alert
                    else "suspicious_input"
                ),
                "tier": "confirmed" if confirmed else ("low" if alert else "d3"),
                "label": cls if alert else "SuspiciousInput",
                "confidence": conf,
                "threshold": rcl.threshold(cls),
                "energy": float(en[i]),
                "title": (
                    (
                        f"{cls} detected"
                        if confirmed
                        else f"Possible {cls} (low confidence)"
                    )
                    + (
                        f" - model unsure: {' / '.join(pset)}"
                        if unsure and pset
                        else ""
                    )
                    if alert
                    else "Possible evasion / manipulated flow"
                ),
                "detail": f"{cls} (attack probability {conf:.2f}, {cls} "
                f"{class_conf:.2f}) from {source}",
                "class_confidence": class_conf,
                # conformal set ~95 % sure to hold the true class; 2+ -> unsure
                "possible_classes": pset,
                "uncertain": unsure,
                "reason": (
                    det.reason(cls)
                    if alert
                    else f"Energy {en[i]:.2f} exceeds the clean-traffic threshold "
                    f"{thr:.2f} (D3): unlike anything seen in training; the model's "
                    f"guess was {names[int(p.argmax())]}."
                ),
                "techniques": techs,
                "candidate_techniques": mapped,
                "severity": (_sev(techs) if confirmed else 2) if alert else 5,
                "logits": [float(x) for x in lg],
                "features": {n: _num(row.get(n, 0)) for n in det.feature_names},
                "status": "new",
            }
        )
        rows_idx.append(i)
    _why(det, out, rows_idx)
    return out


def _why(det: Any, alerts: list[Event], rows_idx: list[int], k: int = 3) -> None:
    """Per-flow reason: the features that pushed THIS flow toward its class
    (gradient x input on the MLP; the class-level SHAP indicators stay in
    ``reason``). Adds ``why`` (list) and appends it to ``reason``."""
    Xs = getattr(det, "last_Xs", None)
    if not alerts or Xs is None or len(Xs) <= max(rows_idx):
        return
    if getattr(det, "ensemble", None) and _why_ensemble(det, alerts, rows_idx, Xs, k):
        return
    import torch

    names = list(det.label_names)
    x = torch.as_tensor(Xs[rows_idx]).clone().requires_grad_(True)
    det.model.eval()
    logits = det.model(x)
    cls = torch.as_tensor(
        [
            names.index(a["label"]) if a["label"] in names else int(logits[j].argmax())
            for j, a in enumerate(alerts)
        ]
    )
    logits.gather(1, cls[:, None]).sum().backward()
    grad = x.grad
    if grad is None:
        return
    contrib = (grad * x).detach().numpy()
    feats = det.feature_names
    for j, a in enumerate(alerts):
        top = [t for t in np.argsort(-contrib[j])[:k] if contrib[j][t] > 0]
        why = [f"{feats[t]} {'high' if Xs[rows_idx[j]][t] > 0 else 'low'}" for t in top]
        a["why"] = why
        if why:
            a["reason"] = f"{a['reason']}  [this flow: {', '.join(why)}]"


def _why_ensemble(
    det: Any, alerts: list[Event], rows_idx: list[int], Xs: Any, k: int
) -> bool:
    """Explain the score the analyst sees (MLP gradient x input + LightGBM TreeSHAP,
    weighted like the ensemble; src/explain/ensemble_why.py). False -> fall back to
    the MLP-only explanation."""
    try:
        from src.explain.ensemble_why import explain_flows

        labels = [a["label"] for a in alerts]
        res = explain_flows(det, Xs, rows_idx, k=4 * k, labels=labels)
    except Exception:  # noqa: BLE001 - an explanation must never stop detection
        _log.debug("ensemble explanation failed; using the MLP", exc_info=True)
        return False
    whys = [
        [
            f"{t['feature']} {'high' if t['value'] > (t['typical'] or 0.0) else 'low'}"
            for t in r["top"]
            if t["contribution"] > 0  # pushed toward the alerted class
        ][:k]
        for r in res
    ]
    if not all(whys):  # nothing pushes toward that class: use the MLP's view
        return False
    for a, r, why in zip(alerts, res, whys, strict=True):
        a["why"] = why
        a["why_detail"] = "\n".join(r["text"].splitlines()[:k])
        if why:
            a["reason"] = f"{a['reason']}  [this flow: {', '.join(why)}]"
    return True


def _num(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if np.isfinite(f) else 0.0


def _sev(techs: list[str]) -> int:
    from src.soc.correlate import TACTIC_WEIGHT, technique_info

    return max((TACTIC_WEIGHT.get(technique_info(t)[1], 3) for t in techs), default=3)


# --- intake ----------------------------------------------------------------------


def _as_alert(ev: Event) -> dict[str, Any]:
    names = ", ".join(technique_info(t)[0] for t in ev["techniques"])
    return ev | {
        "id": ev.get("id") or uuid.uuid4().hex[:12],
        "title": ev.get("title") or f"{names or ev['kind']} ({ev['source']})",
        "label": ev.get("label") or ev["kind"],
        "confidence": ev.get("confidence", 1.0),
        "severity": ev.get("severity") or _sev(ev["techniques"]),
        "reason": ev.get("reason") or ev["detail"],
        "status": ev.get("status", "new"),
    }


def record(
    events: list[Event], send: bool = True, block: bool = False
) -> list[dict[str, Any]]:
    """Store events; raise + forward alerts for the ones that carry ATT&CK evidence.
    Cross-model checks (src/soc/crosscheck.py) add their findings to the same batch."""
    from src.soc import allowlist, crosscheck, custom

    events = list(events) + crosscheck.run(events)
    # the users' changes to built-in rules (on / off, severity, technique, exceptions:
    # src/soc/rulectl.py), then their own rules and thresholds (src/soc/custom.py)
    from src.soc import rulectl

    rulectl.apply(events)
    custom.apply(events)
    sens = custom._sens()  # read once per batch, not once per event
    from src.soc import owasp

    for e in events:
        custom.demote(e, sens=sens)
        if e.get("techniques"):
            owasp.label(e)  # web alerts: their OWASP Top 10 category (e.g. A05:2025 Injection)
    _append(EVENTS, events)
    alerts = [
        _as_alert(e) for e in events if e.get("techniques") or e["source"] == "detector"
    ]
    # allow-list: muted alerts are stored and shown, but not forwarded unless critical
    fwd: dict[str, bool] = {}
    for a in alerts:
        ok, mark = allowlist.forward(a)
        fwd[a["id"]] = ok
        if mark:
            a["allowlisted"] = mark
    _append(ALERTS, alerts)
    if send and alerts:
        settings = sinks.load_settings()
        for a in alerts:
            if not fwd[a["id"]]:
                continue
            sinks.send(
                {k: v for k, v in a.items() if k not in ("logits", "features")},
                settings,
            )
            if (
                (block or settings.get("auto_block"))
                and a.get("src_ip")
                and a["severity"] >= 6
                and not a.get("allowlisted")
            ):
                from src.soc.model import firewall_block

                firewall_block(a["src_ip"])
    return alerts


def ingest_text(
    text: str, name: str, det: Any = None, send: bool = True, observer: str | None = None
) -> dict[str, Any]:
    """Analyse one uploaded/exported file: flow CSV -> detector, else log parsers.
    ``observer``: the machine that wrote the log (an agent's host); stored on every
    event as ``observer`` so the risk board can follow an attack onto its victim."""
    import io

    if name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(io.StringIO(text), low_memory=False)
        except (pd.errors.ParserError, ValueError):
            df = None
        if df is not None and is_flow_table(
            det, df.rename(columns=lambda c: str(c).strip())
        ):
            alerts = record(detect_flows(det, df, name), send=send)
            return {"kind": "flows", "rows": len(df), "alerts": len(alerts)}
    if (sharded := _sharded_ingest(text, name, send, observer)) is not None:
        return sharded
    events = parse_text(text, name)
    if observer:
        for e in events:
            e.setdefault("observer", observer)
    alerts = record(events, send=send)
    return {"kind": "logs", "rows": len(events), "alerts": len(alerts)}


def ingest_page(data: bytes, name: str, send: bool = True, url: str = "") -> dict[str, Any]:
    """An uploaded web page (.html / .htm) or e-mail (.eml: its HTML body and HTML
    attachments) -> the phishing page model (owner 2026-10-24: pages it is given, never
    fetched). A phishing verdict becomes an alert event (source "phishing")."""
    from src.mlareas import phishing_page

    pages = (phishing_page.html_from_email(data) if name.lower().endswith(".eml")
             else [(name, data.decode("utf-8", "replace"))])  # fmt: skip
    results, events = [], []
    for part, html in pages:
        r = phishing_page.check_page(html, url)
        results.append({"part": part, **r})
        if r["verdict"] == "phishing":
            events.append({"ts": time.time(), "src_ip": None, "dst_ip": None, "dst_port": None,
                           "source": "phishing", "kind": "phishing_page", "url": url or None,
                           "file": f"{name}: {part}" if part != name else name,
                           "detail": f"phishing page {name} ({part}): p={r['probability']:.2f}; "
                           + "; ".join(r["reasons"][:4]),
                           "techniques": ["T1566.001" if name.lower().endswith(".eml") else "T1566.002"]})  # fmt: skip
    alerts = record(events, send=send) if events else []
    return {"kind": "page", "pages": len(pages), "phishing": len(events), "alerts": len(alerts), "results": results}


def _sharded_ingest(text: str, name: str, send: bool, observer: str | None) -> dict[str, Any] | None:
    """Big line-by-line logs: parse AND store in the worker processes (src/soc/parallel.py
    parse_and_record). Only where the result is the same as one process: a line-based
    format the parsers know (not XML / Zeek / W3C / JSON documents, not an unknown
    format), and no own rule that counts events across the file (N times in M minutes).
    One difference: cross-check findings (a URL / file seen many times) are de-duplicated
    per 10,000-line slice instead of per file."""
    from src.soc import custom, parallel, perf
    from src.soc.parsers import parse_line

    lines = text.splitlines()
    if len(lines) < parallel.MIN_LINES:
        return None
    workers = int(perf.plan()["workers"])
    head = text[:4000]
    if workers <= 1 or "<EventID" in head or "id.orig_h" in head or "#Fields:" in head or "#path\t" in head:
        return None
    if text.lstrip()[:1] == "[" or '"Records"' in head:  # JSON documents, not JSON lines
        return None
    try:
        rules, _ = custom.rules()
    except Exception:  # noqa: BLE001 - unreadable rules: the normal path reports them
        return None
    if any(getattr(r, "count", 1) > 1 for r in rules):
        return None
    sample = [e for ln in lines[:2000] if (e := parse_line(ln))]
    if len(sample) < 5 or sum(1 for e in sample if e["kind"] == "log" and not e["techniques"]) / len(sample) >= 0.9:
        return None  # mostly unrecognised: the unknown-format path decides
    res = parallel.parse_and_record(lines, workers, send, observer)
    return {"kind": "logs", "rows": res["rows"], "alerts": res["alerts"], "workers": res["workers"]}


def ingest_evtx(data: bytes, name: str, send: bool = True) -> dict[str, Any]:
    """Analyse an uploaded Windows event log export (.evtx, binary)."""
    import tempfile

    from src.soc.parsers import read_evtx

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "upload.evtx"
        p.write_bytes(data)
        events = read_evtx(p)
    for e in events:
        e["detail"] = f"{name}: {e['detail']}"
    alerts = record(events, send=send)
    return {"kind": "evtx", "rows": len(events), "alerts": len(alerts)}


ZIP_MAX_BYTES = 4 * 2**30  # uncompressed guard against zip bombs
ZIP_MAX_FILES = 500


def ingest_zip(
    data: bytes, name: str, det: Any = None, send: bool = True
) -> dict[str, Any]:
    """Analyse every log / flow file inside an uploaded .zip (e.g. OTRF datasets)."""
    import io
    import zipfile

    out: dict[str, Any] = {
        "kind": "zip",
        "files": 0,
        "rows": 0,
        "alerts": 0,
        "skipped": [],
    }
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        if (
            len(infos) > ZIP_MAX_FILES
            or sum(i.file_size for i in infos) > ZIP_MAX_BYTES
        ):
            raise ValueError(f"{name}: archive too large to analyse (unpack it first)")
        for i in infos:
            member = Path(i.filename).name
            if member.startswith("._") or "__MACOSX" in i.filename:
                continue
            raw = z.read(i)
            if member.lower().endswith(".evtx"):
                r = ingest_evtx(raw, member, send)
            elif member.lower().endswith((".json", ".log", ".txt", ".csv", ".jsonl")):
                r = ingest_text(raw.decode("utf-8", "replace"), member, det, send)
            else:
                out["skipped"].append(member)
                continue
            out["files"] += 1
            out["rows"] += r["rows"]
            out["alerts"] += r["alerts"]
    return out


_inc_cache: dict[float, tuple[tuple[int, int], list[dict[str, Any]]]] = {}


def incidents(window_s: float | None = None) -> list[dict[str, Any]]:
    """Incidents over the whole store. Through the event index (src/soc/eventindex.py:
    only entities with new events are recomputed - milliseconds at millions of
    events); the whole-list path is the fallback if the index cannot be used.
    ``window_s`` None: the correlation gap from Customise -> Engine settings."""
    import sqlite3

    from src.soc import allowlist, eventindex, rulectl

    if window_s is None:
        window_s = float(rulectl.engine_settings()["correlation_gap_s"])

    try:
        incs = eventindex.incidents(window_s)
    except (sqlite3.Error, OSError) as exc:
        _log.warning("event index unavailable (%s) - correlating the whole list", exc)
        incs = _incidents_from_list(window_s)
    # re-checked every call: entries can be added or expire (read once per call)
    active = [e for e in allowlist.entries() if e["active"]]
    for inc in incs:
        inc["allowlisted"] = allowlist.is_muted(inc, active=active)[1]
    return incs


def _incidents_from_list(window_s: float) -> list[dict[str, Any]]:
    try:
        st = EVENTS.stat()
    except OSError:
        return []
    key = (st.st_mtime_ns, st.st_size)
    hit = _inc_cache.get(window_s)
    if hit is None or hit[0] != key:  # events unchanged -> same incidents
        from src.soc import places, riskboard

        evs = read_jsonl(EVENTS)
        board = riskboard.board_for(evs, scale=places.board_scale())
        hit = _inc_cache[window_s] = (key, correlate(evs, window_s, board=board))
    return [dict(i) for i in hit[1]]


def _retry_delivery() -> None:
    """Retry alerts queued while the SIEM was unreachable (src/soc/delivery.py)."""
    from src.soc import delivery

    try:
        delivery.flush()
    except Exception as exc:  # noqa: BLE001 - delivery problems never stop monitoring
        _log.warning("delivery retry failed: %s", exc)


def push_incidents(min_severity: int | None = None) -> int:
    """Send new or escalated incidents to the SOC. Returns how many were sent.

    Default: the place profile decides (src/soc/places.py - minimum severity and a
    daily budget; critical incidents always go). An explicit ``min_severity`` keeps
    the plain threshold without a budget."""
    from src.soc import allowlist, places

    sent = json.loads(PUSHED.read_text("utf-8")) if PUSHED.is_file() else {}
    settings, n = sinks.load_settings(), 0
    for inc in incidents():
        if sent.get(inc["id"], -1) >= inc["severity"]:
            continue
        ok, mark = allowlist.forward(inc)  # muted unless critical; not marked sent
        if not ok:
            continue
        if min_severity is not None:
            if inc["severity"] < min_severity:
                continue
        elif not places.may_push(inc["severity"]):
            continue
        sinks.send(
            {
                "ts": inc["end"],
                "kind": "incident",
                "title": f"Correlated incident ({inc['level']})",
                "src_ip": inc["src_ip"],
                "severity": inc["severity"],
                "techniques": inc["techniques"],
                "reason": " ".join(inc["findings"] + inc.get("linked_findings", []))
                or ", ".join(t["name"] for t in inc["techniques"]),
                "sources": inc["sources"],
                **({"allowlisted": mark} if mark else {}),
            },
            settings,
        )
        sent[inc["id"]], n = inc["severity"], n + 1
    PUSHED.parent.mkdir(parents=True, exist_ok=True)
    PUSHED.write_text(json.dumps(sent), "utf-8")
    return n


# --- learning ----------------------------------------------------------------------


def _scheme(label_names: list[str]) -> str:
    from src.data.labels import BINARY_ORDER, CIOT_ORDER

    if tuple(label_names) == CIOT_ORDER:
        return "ciot"
    return "binary" if tuple(label_names) == BINARY_ORDER else "family"


def _finetune(
    det: Any, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> dict[str, Any]:
    """Continual fine-tune the deployed model; keep it only if validation does not drop."""
    from types import SimpleNamespace

    import torch
    from sklearn.metrics import f1_score

    from config.base import build_config, load_experiment
    from src.selfimprove.extensions import continual_update
    from src.soc.model import DEPLOY

    def f1(model: Any) -> float:
        model.eval()  # continual_update leaves it in train mode (dropout on)
        with torch.no_grad():
            pred = (
                model(torch.as_tensor(det.scaler.transform(X_val).astype(np.float32)))
                .argmax(1)
                .numpy()
            )
        return float(f1_score(y_val, pred, average="macro", zero_division=0))

    before = f1(det.model)
    state = {k: v.clone() for k, v in det.model.state_dict().items()}
    cfg = build_config(load_experiment("smoke.yaml"), seed=0)
    data = SimpleNamespace(
        X_train=det.scaler.transform(X).astype(np.float32),
        y_train=y,
        class_weights=None,
    )
    continual_update(det.model, data, cfg, "cpu")  # type: ignore[arg-type]  # duck-typed
    after = f1(det.model)
    kept = after >= before - 0.02  # guard: never deploy a model that got worse
    if kept:
        torch.save(state, DEPLOY / "model.prev.pt")
        torch.save(det.model.state_dict(), DEPLOY / "model.pt")
    else:
        det.model.load_state_dict(state)
    return {
        "macro_f1_before": before,
        "macro_f1_after": after,
        "kept": kept,
        "n_train": int(len(y)),
    }


def retrain_from_feedback(det: Any, rcl: RCL | None = None) -> dict[str, Any]:
    """Fold analyst-reviewed alerts (with their true label) into the deployed model."""
    rows = [
        r
        for r in (rcl or RCL()).rows()
        if r.get("features") and r["true_label"] in det.label_names
    ]
    if len(rows) < 10:
        return {
            "error": f"need at least 10 reviewed alerts with features (have {len(rows)})"
        }
    from sklearn.model_selection import train_test_split

    X = np.array(
        [[r["features"].get(n, 0.0) for n in det.feature_names] for r in rows],
        np.float32,
    )
    y = np.array([det.label_names.index(r["true_label"]) for r in rows])
    # Hold reviewed alerts out of the fine-tune: judging the update on its own
    # training rows meant the keep/rollback guard could never reject anything.
    strat = y if np.bincount(y).min() >= 2 else None
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=strat
    )
    return _finetune(det, X_tr, y_tr, X_va, y_va)


def add_dataset(
    det: Any, df: pd.DataFrame, label_col: str, name: str
) -> dict[str, Any]:
    """Validate a labelled flow CSV, save it, and fine-tune the detector on it.

    Raises ``ValueError`` with a plain-language reason when the file cannot be used.
    """
    from sklearn.model_selection import train_test_split

    from src.soc.columns import align, explain

    label_col = label_col.strip()
    df, cols = align(df, det.feature_names)
    if label_col not in df.columns:
        raise ValueError(f"label column {label_col!r} not found")
    if cols["share"] < 0.8:
        raise ValueError(explain(cols))
    labels, unknown = map_labels(df[label_col], det.label_names)
    keep = labels.notna().to_numpy()
    df, labels = df[keep], labels[keep]
    if len(df) < 20:
        raise ValueError(
            "need at least 20 rows whose labels match the detector's classes "
            f"({', '.join(det.label_names)}); not recognised: {', '.join(unknown[:8])}"
        )
    stats = getattr(det, "input_stats", None)
    fill = dict(zip(det.feature_names, stats["median"], strict=True)) if stats else {}
    X = (
        df.reindex(columns=det.feature_names)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    X[cols["missing"]] = X[cols["missing"]].fillna(
        {n: fill.get(n, 0.0) for n in cols["missing"]}
    )  # absent columns -> typical training value
    ok = X.notna().all(axis=1)
    X, labels = X[ok].to_numpy(np.float32), labels[ok]
    y = np.array([det.label_names.index(v) for v in labels])
    strat = y if np.bincount(y).min() >= 2 else None
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=strat
    )

    safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_") or "dataset"
    USER_DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(USER_DATA / f"{safe}.csv", index=False)
    return _finetune(det, X_tr, y_tr, X_va, y_va) | {
        "saved_to": str(USER_DATA / f"{safe}.csv"),
        "class_counts": {
            det.label_names[i]: int(c)
            for i, c in enumerate(np.bincount(y, minlength=len(det.label_names)))
            if c
        },
        "columns": {k: cols[k] for k in ("format", "matched", "needed", "missing")},
        "ignored_labels": unknown,
    }


def map_labels(raw: pd.Series, label_names: list[str]) -> tuple[pd.Series, list[str]]:
    """Raw labels -> the detector's classes; unknown labels become NaN (dropped and
    reported) instead of failing the whole file."""
    from src.data.labels import resolve_labels

    scheme = _scheme(label_names)
    out: dict[Any, Any] = {}
    unknown: list[str] = []
    for v in raw.unique():
        try:
            cls = resolve_labels(pd.Series([v]), scheme).iloc[0]
        except (KeyError, ValueError):
            cls = None
        if cls in label_names:
            out[v] = cls
        else:
            out[v] = None
            unknown.append(str(v))
    return raw.map(out), sorted(unknown)


# --- monitor process -------------------------------------------------------------


def _read_status() -> dict[str, Any]:
    """The monitor rewrites this file every few seconds; a read that lands mid-write
    sees a partial file, which must not crash the dashboard."""
    try:
        return dict(json.loads(STATUS.read_text("utf-8")))
    except (OSError, ValueError):
        return {}


def _status(**kw: Any) -> None:
    cur = _read_status()
    cur.update(kw, heartbeat=time.time())
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(cur), "utf-8")


def monitor_running() -> dict[str, Any] | None:
    s = _read_status()
    return s if s.get("running") and time.time() - s.get("heartbeat", 0) < 15 else None


def _watch_inbox(folder: Path, det: Any, stop: threading.Event) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / ".processed"
    seen = set(done.read_text("utf-8").splitlines()) if done.is_file() else set()
    while not stop.is_set():
        for p in sorted(folder.iterdir()):
            if p.is_file() and not p.name.startswith(".") and p.name not in seen:
                if time.time() - p.stat().st_mtime < 3:
                    continue  # still being copied in; a partial read would be final
                try:
                    r = ingest_text(p.read_text("utf-8", errors="replace"), p.name, det)
                    _log.info("inbox %s -> %s", p.name, r)
                except OSError as exc:
                    _log.warning("inbox %s unreadable: %s", p.name, exc)
                    continue
                seen.add(p.name)
                with open(done, "a", encoding="utf-8") as fh:
                    fh.write(p.name + "\n")
        stop.wait(3)


def _syslog_listener(port: int, stop: threading.Event) -> None:
    class H(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            data = self.request[0].decode("utf-8", errors="replace")
            evs = [e for ln in data.splitlines() if (e := parse_line(ln))]
            for e in evs:
                e["src_ip"] = e["src_ip"] or self.client_address[0]
            record(evs)

    with socketserver.UDPServer(("0.0.0.0", port), H) as srv:  # noqa: S104 - SOC listener
        srv.timeout = 1
        _log.info("syslog listener on udp/%d", port)
        while not stop.is_set():
            srv.handle_request()


FLOW_BATCH = 256  # live flows scored together
FLOW_BATCH_S = 0.5  # ...or after this long, whichever comes first (was 1 s)


def monitor(args: argparse.Namespace) -> None:
    configure_logging()
    det = load_detector()
    settings = sinks.load_settings()
    stop = threading.Event()
    threads = []
    if settings["inbox"]["enabled"]:
        threads.append(
            threading.Thread(
                target=_watch_inbox,
                args=(Path(settings["inbox"]["path"]), det, stop),
                daemon=True,
            )
        )
    if settings["listener"]["enabled"]:
        threads.append(
            threading.Thread(
                target=_syslog_listener,
                args=(int(settings["listener"]["port"]), stop),
                daemon=True,
            )
        )
    for t in threads:
        t.start()
    _status(
        running=True,
        pid=os.getpid(),
        started=time.time(),
        source=args.replay or ("live" if args.live else "logs only"),
        flows=0,
        alerts=0,
    )

    flows = None
    if det is not None and (args.replay or args.live):
        from src.live.capture import live_flows, replay_flows

        flows = (
            live_flows(args.iface)
            if args.live
            else replay_flows(args.replay, rate_hz=args.rate)
        )
    n = n_alerts = 0
    last_push = 0.0
    push_s = sinks.live_timing("push_s")  # incidents to the SIEM (was every 5 s)
    # flows are scored in batches (2026-10-17: one table per flow ran ~58 flows/s,
    # batches ~2,300/s with identical alerts); a flow waits at most FLOW_BATCH_S
    batch: list[Any] = []
    first = 0.0
    source_name = "live" if args.live else Path(args.replay).name if args.replay else "live"

    def flush() -> int:
        if not batch:
            return 0
        out = len(record(detect_flows(det, pd.DataFrame(batch), source_name), block=args.block))
        batch.clear()
        return out

    try:
        for _meta, feats in flows or iter(()):
            n += 1
            if not batch:
                first = time.time()
            batch.append(feats)
            if len(batch) >= FLOW_BATCH or time.time() - first >= FLOW_BATCH_S:
                n_alerts += flush()
            if time.time() - last_push > push_s:
                push_incidents()
                _retry_delivery()
                _status(flows=n, alerts=n_alerts)
                last_push = time.time()
        n_alerts += flush()  # the last partial batch
        while (
            flows is None or args.keep_running
        ):  # logs-only mode, or keep watching after a replay
            push_incidents()
            _retry_delivery()
            _status(flows=n, alerts=n_alerts)
            time.sleep(push_s)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        push_incidents()
        _status(running=False, flows=n, alerts=n_alerts)


def main() -> None:
    ap = argparse.ArgumentParser(description="SplitShield SOC engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("monitor", help="run the live monitor")
    m.add_argument("--replay", default=None, help="CSV of flows to stream as if live")
    m.add_argument("--rate", type=float, default=20.0, help="replay flows/second")
    m.add_argument(
        "--live", action="store_true", help="sniff a live interface (Npcap + admin)"
    )
    m.add_argument("--iface", default=None)
    m.add_argument(
        "--block", action="store_true", help="firewall-block severe sources (admin)"
    )
    m.add_argument(
        "--keep-running",
        action="store_true",
        help="keep watching logs after a replay ends",
    )
    a = sub.add_parser("analyze", help="analyse log/flow files once")
    a.add_argument("files", nargs="+")
    args = ap.parse_args()
    if args.cmd == "monitor":
        monitor(args)
    else:
        det = load_detector()
        for f in args.files:
            print(
                f,
                ingest_text(
                    Path(f).read_text("utf-8", errors="replace"), Path(f).name, det
                ),
            )
        print(f"incidents: {len(incidents())}, pushed: {push_incidents()}")


if __name__ == "__main__":
    main()
