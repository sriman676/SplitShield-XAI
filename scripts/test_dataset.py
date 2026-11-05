"""Test ANY dataset (owner 2026-10-15): drop in a file or folder, get a report card.

Auto-detects what each file is and runs the matching detector:

  * flow tables (CICFlowMeter, any naming style)   -> deployed flow detector
  * NetFlow v9 tables (IN_BYTES, L4_DST_PORT, ...)  -> NetFlow detector
  * a list / column of URLs                          -> phishing model
  * EMBER-format JSON lines (PE file features)       -> malware model
  * anything else (logs: Windows, Linux, web, cloud, Zeek, unknown formats)
                                                     -> parsers + rules + Sigma

If the data has a label column (label / Label / attack / class / is_malicious ...),
the card also shows how many attacks were caught, missed and how many normal rows
raised an alert. Nothing is trained or tuned on the data being tested.

    python -m scripts.test_dataset <file-or-folder> [--label COLUMN] [--max-rows N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_LABELS = ("label", "attack", "class", "is_malicious", "malicious", "target", "y")


def _label_col(df: pd.DataFrame, given: str | None) -> str | None:
    if given and given in df.columns:
        return given
    low = {str(c).strip().lower(): c for c in df.columns}
    return next((low[k] for k in _LABELS if k in low), None)


def _is_attack(v: Any) -> bool:
    s = str(v).strip().lower()
    return s not in (
        "0",
        "0.0",
        "benign",
        "normal",
        "false",
        "legitimate",
        "-",
        "",
        "nan",
    )


def _score(y: list[bool], alert: list[bool]) -> dict[str, Any]:
    ya, al = np.asarray(y, bool), np.asarray(alert, bool)
    tp, fn = int((ya & al).sum()), int((ya & ~al).sum())
    fp, tn = int((~ya & al).sum()), int((~ya & ~al).sum())
    return {
        "attacks": int(ya.sum()), "caught": tp, "missed": fn,
        "normal": int((~ya).sum()), "false_alarms": fp,
        "detection_rate": tp / max(1, tp + fn), "false_alarm_rate": fp / max(1, fp + tn),
        "precision": tp / max(1, tp + fp),
    }  # fmt: skip


def card_table(path: Path, df: pd.DataFrame, label: str | None) -> dict[str, Any]:
    from src.soc import engine
    from src.soc.columns import match

    cols = list(df.columns)
    out: dict[str, Any] = {"file": path.name, "rows": len(df), "label_column": label}
    det = engine.load_detector()
    if det is not None and match(cols, det.feature_names)["share"] >= 0.5:
        out["type"] = "CICFlowMeter flows"
        alerts = engine.detect_flows(det, df, path.name)
        flagged = {int(a.get("row", -1)) for a in alerts}
        out["alerts"] = len(alerts)
        out["by_class"] = Counter(a.get("label") for a in alerts).most_common(8)
        if label:
            out["score"] = _score(
                [_is_attack(v) for v in df[label]],
                [i in flagged for i in range(len(df))],
            )
        return out
    upper = {str(c).strip().upper() for c in cols}
    if {"IN_BYTES", "OUT_BYTES", "L4_DST_PORT"} <= upper:
        from src.soc.netflow_model import load_netflow, score_netflow

        out["type"] = "NetFlow v9 flows"
        if load_netflow() is None:
            out["error"] = "NetFlow detector not deployed"
            return out
        _, nf_alerts = score_netflow(df)
        out["alerts"] = int(np.asarray(nf_alerts).sum())
        if label:
            out["score"] = _score(
                [_is_attack(v) for v in df[label]], list(np.asarray(nf_alerts, bool))
            )
        return out
    url_col = next(
        (
            c
            for c in cols
            if str(c).strip().lower() in ("url", "urls", "domain", "link")
        ),
        None,
    )
    if url_col is not None:
        from src.mlareas.phishing import check_urls

        out["type"] = "URLs"
        res = check_urls([str(u) for u in df[url_col]])
        alert = [r.get("verdict") == "phishing" for r in res]
        out["alerts"] = int(sum(alert))
        if label:
            out["score"] = _score([_is_attack(v) for v in df[label]], alert)
        return out
    out["type"] = "unrecognised table: scanned as log records"
    return card_logs(path, df.to_csv(index=False))


def card_logs(path: Path, text: str) -> dict[str, Any]:
    from src.soc.correlate import correlate
    from src.soc.parsers import parse_text

    evs = parse_text(text, path.name)
    alerts = [e for e in evs if e.get("techniques")]
    incs = correlate(evs)
    return {
        "file": path.name, "type": "logs", "events": len(evs), "alerts": len(alerts),
        "alert_rate": len(alerts) / max(1, len(evs)),
        "sources": Counter(e["source"] for e in evs).most_common(5),
        "top_techniques": Counter(t for e in alerts for t in e["techniques"]).most_common(10),
        "incidents": len(incs),
        "top_incidents": [
            {"entity": i["src_ip"], "level": i["level"], "severity": i["severity"],
             "path": [t["name"] for t in i["techniques"]][:6]}
            for i in incs[:5]
        ],
    }  # fmt: skip


def card_ember(path: Path, lines: list[str]) -> dict[str, Any]:
    from src.mlareas.malware import check_file_features

    y, alert = [], []
    for ln in lines:
        rec = json.loads(ln)
        r = check_file_features(rec)
        alert.append(r.get("verdict") != "benign")
        y.append(rec.get("label", -1))
    out: dict[str, Any] = {"file": path.name, "type": "EMBER file features", "rows": len(lines),
                           "alerts": int(sum(alert))}  # fmt: skip
    if any(v in (0, 1) for v in y):
        keep = [i for i, v in enumerate(y) if v in (0, 1)]
        out["score"] = _score([y[i] == 1 for i in keep], [alert[i] for i in keep])
    return out


def test_path(
    path: Path, label: str | None = None, max_rows: int = 200_000
) -> list[dict[str, Any]]:
    files = (
        [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    )
    cards = []
    for f in files:
        try:
            if f.suffix.lower() in (".csv", ".parquet"):
                df = (
                    pd.read_parquet(f)
                    if f.suffix.lower() == ".parquet"
                    else pd.read_csv(f, nrows=max_rows, low_memory=False)
                )
                cards.append(
                    card_table(
                        f,
                        df.rename(columns=lambda c: str(c).strip()),
                        _label_col(df, label),
                    )
                )
            elif f.suffix.lower() in (".jsonl", ".json"):
                head = f.open(encoding="utf-8", errors="replace").readline()
                if '"byteentropy"' in head and '"histogram"' in head:
                    with f.open(encoding="utf-8", errors="replace") as fh:
                        cards.append(
                            card_ember(
                                f,
                                [
                                    ln
                                    for _, ln in zip(range(max_rows), fh, strict=False)
                                ],
                            )
                        )
                else:
                    cards.append(card_logs(f, f.read_text("utf-8", errors="replace")))
            elif f.suffix.lower() == ".evtx":
                from src.soc.parsers import read_evtx

                evs = read_evtx(f)
                cards.append({"file": f.name, "type": "Windows .evtx", "events": len(evs),
                              "alerts": sum(1 for e in evs if e["techniques"])})  # fmt: skip
            else:
                cards.append(card_logs(f, f.read_text("utf-8", errors="replace")))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the card
            cards.append(
                {"file": f.name, "error": f"{type(exc).__name__}: {exc}"[:300]}
            )
    return cards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-rows", type=int, default=200_000)
    a = ap.parse_args()
    for c in test_path(Path(a.path), a.label, a.max_rows):
        print(json.dumps(c, indent=1, default=str))


if __name__ == "__main__":
    main()
