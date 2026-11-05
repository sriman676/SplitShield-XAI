"""SOC self-test: the deployed detector, end to end, on held-out labelled flows.

Replays the deployed dataset's TEST split (raw features + true labels, never seen in
training) through the real alerting engine in an isolated temporary store, then
measures what an analyst and a SIEM would experience:

  * alerting   - recall / precision / false-positive rate per tier (low, confirmed),
                 per true class; missed attacks by class
  * mapping    - share of confirmed alerts whose ATT&CK techniques match the TRUE
                 class's techniques (wrong family -> wrong mapping)
  * correlation- incidents built from the alerts + the demo multi-sensor logs
  * delivery   - incidents pushed to local Splunk-HEC / syslog receivers + outbox
  * assistant  - fixed questions answered with the expected intent

Writes results/soc_selftest.json (rendered in FINAL_REPORT). Real artifacts/soc is
never touched. Run after deploy.py:

    python -m scripts.soc_selftest [--max-flows 20000]
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = [
    ("what attacks are happening today?", "incidents"),
    ("who is attacking us?", "top"),
    ("explain port scan", "explain"),
    ("how do I stop brute force?", "defend"),
    ("which incidents used T1110?", "mitre"),
    ("how many alerts in the last hour?", "count"),
]


def _isolate(tmp: Path) -> None:
    from src.soc import engine, rcl, sinks

    for mod, names in (
        (engine, ("EVENTS", "ALERTS", "PUSHED", "STATUS", "FLOW_SAMPLE")),
        (rcl, ("STATE", "FEEDBACK")),
    ):
        for n in names:
            setattr(mod, n, tmp / Path(getattr(mod, n)).name)
    sinks.SETTINGS, sinks.OUTBOX = tmp / "settings.json", tmp / "outbox.jsonl"


def _receivers(tmp: Path) -> tuple[HTTPServer, socket.socket, list[Any]]:
    from src.soc import sinks

    seen: list[Any] = []

    class HEC(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append((self.headers.get("Authorization"), json.loads(body)))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a: object) -> None:
            pass

    hec = HTTPServer(("127.0.0.1", 0), HEC)
    threading.Thread(target=hec.serve_forever, daemon=True).start()
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    udp.settimeout(0.5)
    s = sinks.apply_preset(sinks.load_settings(), "Splunk")
    s["jsonl"]["path"] = str(sinks.OUTBOX)
    s["webhook"] |= {
        "url": f"http://127.0.0.1:{hec.server_port}/services/collector/event",
        "token": "selftest-token",
    }
    s["syslog"] |= {"enabled": True, "host": "127.0.0.1", "port": udp.getsockname()[1]}
    sinks.save_settings(s)
    return hec, udp, seen


def run(max_flows: int = 20_000) -> dict[str, Any]:
    from config.base import build_config, experiment_for_dataset
    from src.data.preprocess import preprocess_frame
    from src.data.sources import get_spec, load_raw
    from src.soc import engine, sinks
    from src.soc.correlate import CLASS_TECHNIQUES
    from src.soc.model import DEPLOY, Detector
    from src.soc.samples import make_samples

    cols = json.loads((DEPLOY / "columns.json").read_text("utf-8"))
    dataset = cols["dataset"]
    tmp = Path(tempfile.mkdtemp(prefix="soc_selftest_"))
    _isolate(tmp)
    det = Detector()

    # the deployed dataset's held-out test rows, raw (unscaled) + true labels
    spec = get_spec(dataset)
    cfg = build_config(experiment_for_dataset(dataset), seed=0)
    data = preprocess_frame(load_raw(spec), spec, cfg, seed=0)
    rng = np.random.default_rng(0)
    idx = np.asarray(data.idx_test)
    if len(idx) > max_flows:
        idx = rng.choice(idx, max_flows, replace=False)
    frame = data.frame.iloc[idx]
    truth = frame["_family"].astype(str).tolist()  # true class per flow
    flows = frame[det.feature_names].copy()
    # row id rides in a column the engine copies into each alert (dst_ip); cleared
    # again before correlation so it cannot shape incidents
    flows["dst_ip"] = [str(i) for i in range(len(flows))]
    alerts = engine.detect_flows(det, flows, "selftest")
    alert_rows = {int(a["dst_ip"]): a for a in alerts}
    for a in alerts:
        a["dst_ip"] = None

    res: dict[str, Any] = {"dataset": dataset, "flows": len(flows)}
    is_attack = np.array([t != "BENIGN" for t in truth])
    for tier in ("any", "confirmed"):
        flagged = np.zeros(len(flows), dtype=bool)
        for r, a in alert_rows.items():
            if a.get("tier") in (
                ("low", "confirmed") if tier == "any" else ("confirmed",)
            ):
                flagged[r] = True
        tp, fp = int((flagged & is_attack).sum()), int((flagged & ~is_attack).sum())
        fn = int((~flagged & is_attack).sum())
        res[f"alerting_{tier}"] = {
            "recall": tp / max(1, tp + fn),
            "precision": tp / max(1, tp + fp),
            "false_positive_rate": fp / max(1, int((~is_attack).sum())),
            "missed_by_class": _count([t for i, t in enumerate(truth) if is_attack[i] and not flagged[i]]),
        }  # fmt: skip
    conf = [
        (r, a)
        for r, a in alert_rows.items()
        if a.get("tier") == "confirmed" and is_attack[r]
    ]
    right = sum(
        set(a["techniques"]) == set(CLASS_TECHNIQUES.get(truth[r], [])) for r, a in conf
    )
    res["mapping_correct_share"] = right / max(1, len(conf))

    # correlation + delivery: alerts plus the demo multi-sensor logs
    hec, udp, seen = _receivers(tmp)
    engine.record([a for a in alerts if a.get("tier") == "confirmed"][:2000])
    for f in make_samples(tmp / "samples"):
        if f.suffix != ".csv":
            engine.ingest_text(f.read_text("utf-8", errors="replace"), f.name, det)
    incs = engine.incidents(window_s=3600)
    pushed = engine.push_incidents()
    syslog = 0
    try:
        while True:
            udp.recv(65535)
            syslog += 1
    except TimeoutError:
        pass
    hec.shutdown()
    res["correlation"] = {"incidents": len(incs), "pushed": pushed}
    res["delivery"] = {
        "hec_received": len(seen),
        "hec_auth_ok": all(a == "Splunk selftest-token" for a, _ in seen),
        "syslog_received": syslog,
        "outbox_lines": len(sinks.OUTBOX.read_text("utf-8").splitlines()) if sinks.OUTBOX.is_file() else 0,
    }  # fmt: skip

    from src.nlp.assistant import ShieldAssistant

    bot = ShieldAssistant()
    got = [(q, want, bot.intent(q)) for q, want in QUESTIONS]
    res["assistant"] = {
        "intent_correct": sum(w == g for _, w, g in got),
        "questions": len(got),
        "wrong": [{"q": q, "want": w, "got": g} for q, w, g in got if w != g],
        "narrator": bot.lm_name,
    }
    out = ROOT / "results" / "soc_selftest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), "utf-8")
    return res


def _count(xs: list[str]) -> dict[str, int]:
    vals, n = (
        np.unique(np.array(xs, dtype=object), return_counts=True) if xs else ([], [])
    )
    return {str(v): int(c) for v, c in zip(vals, n, strict=True)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-flows", type=int, default=20_000)
    args = ap.parse_args()
    res = run(args.max_flows)
    a, c = res["alerting_any"], res["alerting_confirmed"]
    print(
        f"[soc-selftest] {res['flows']} test flows | any-tier recall {a['recall']:.3f} "
        f"FPR {a['false_positive_rate']:.3%} | confirmed recall {c['recall']:.3f} "
        f"FPR {c['false_positive_rate']:.3%} precision {c['precision']:.3f} | "
        f"ATT&CK mapping correct {res['mapping_correct_share']:.1%} | incidents "
        f"{res['correlation']['incidents']}, pushed {res['correlation']['pushed']} | "
        f"HEC {res['delivery']['hec_received']} syslog {res['delivery']['syslog_received']} | "
        f"assistant {res['assistant']['intent_correct']}/{res['assistant']['questions']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
