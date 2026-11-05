"""Speed of searches, chat answers and incidents at large event stores (owner 2026-10-16).

    python -m scripts.speed_benchmark --sizes 200000 1000000 --work D:/speedbench
    python -m scripts.speed_benchmark --sizes 10000000 --profiles 4

Real log lines (loghub SSH / Apache / Linux) go through the SOC's own parsers once;
the parsed events are then repeated with shifted times and varied source IPs and
users until each size is reached, written to a store in ``--work`` (never the SOC's
own store), and indexed. Per performance setting (``auto`` or a memory budget in
GB, src/soc/perf.py) it measures (median of repeats, ms):

  * search      plain-English searches (narrow and broad)
  * chat        whole assistant answers (search, "is X happening", incidents, explain)
  * incidents   first view (cold) and after 1,000 new events (incremental)
  * status      "is brute force happening?" over the whole store

Results: results/extra/speed_benchmark.json (shown on the dashboard's Customise > Speed).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

from utils.paths import PROJECT_ROOT, RAW_DIR, RESULTS_DIR

OUT = RESULTS_DIR / "extra" / "speed_benchmark.json"
SEARCHES = [
    '"break-in attempt"',  # narrow phrase
    "failed logins for root",  # kind + user, broad
    "events from 183.62.140.253",  # one IP
    "invalid user admin",  # text, medium
    "show T1110 alerts",  # technique
    "failed logins last 2 hours",  # time window
]
CHAT = [
    "failed logins for root",
    "is brute force happening?",
    "show critical incidents",
    "what is T1110",
    "summarise today",
]


def base_events() -> list[dict[str, Any]]:
    from src.soc.parsers import parse_text

    evs: list[dict[str, Any]] = []
    for rel in (
        "loghub/SSH/SSH.log",
        "loghub/Apache/Apache.log",
        "loghub/Linux/Linux.log",
    ):
        p = RAW_DIR / rel
        if p.is_file():
            text = p.read_text("utf-8", errors="replace")
            for i in range(0, len(text), 8_000_000):  # parse in slices (memory)
                chunk = text[i : i + 8_000_000]
                evs += parse_text(chunk[: chunk.rfind("\n") + 1], p.name)
    if not evs:
        raise SystemExit("no loghub logs under data/raw - nothing to benchmark")
    return evs


def write_store(path: Path, base: list[dict[str, Any]], n: int, now: float) -> None:
    """``n`` events: the base set repeated over the last 30 days, IPs and users varied."""
    rng = random.Random(0)
    span = 30 * 86400
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n):
            e = dict(base[i % len(base)])
            rep = i // len(base)
            e["ts"] = now - span + span * i / n + rng.random()
            if rep and e.get("src_ip"):
                e["src_ip"] = (
                    f"10.{rep % 250}.{rng.randrange(256)}.{rng.randrange(256)}"
                )
            if rep and e.get("user") and rng.random() < 0.5:
                e["user"] = f"{e['user']}{rep % 97}"
            fh.write(json.dumps(e) + "\n")


def _ms(fn: Any, repeat: int = 5) -> float:
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(times), 1)


def measure(store: Path, profile: str, base: list[dict[str, Any]]) -> dict[str, Any]:
    from src.nlp.assistant import ShieldAssistant
    from src.nlp.status import check_store
    from src.soc import engine, eventindex, search

    engine.EVENTS = store
    eventindex.close()
    from src.soc import perf

    if profile == "auto":
        perf.save_settings("auto")
    else:
        perf.save_settings("memory", memory_gb=float(profile))
    eventindex.close()
    row: dict[str, Any] = {"profile": profile}
    t = time.perf_counter()
    n = eventindex.count()  # catches up with the store: the first build
    row["events"] = n
    row["index_build_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    eventindex.incidents(900.0, scale=1.0)
    row["incidents_first_s"] = round(time.perf_counter() - t, 1)
    row["search_ms"] = {q: _ms(lambda q=q: search.run_store(q)) for q in SEARCHES}
    bot = ShieldAssistant()
    row["chat_ms"] = {q: _ms(lambda q=q: bot.answer(q), repeat=3) for q in CHAT}
    row["status_ms"] = _ms(lambda: check_store(["T1110"], None))
    row["incidents_cached_ms"] = _ms(lambda: eventindex.incidents(900.0, scale=1.0))
    # 1,000 new events arrive: only their entities are recomputed
    with open(store, "a", encoding="utf-8") as fh:
        for i in range(1000):
            e = dict(base[i % len(base)])
            e["ts"] = time.time() + i
            fh.write(json.dumps(e) + "\n")
    row["incidents_after_1000_new_ms"] = _ms(
        lambda: eventindex.incidents(900.0, scale=1.0), repeat=1
    )
    row["db_mb"] = round(eventindex.db_path().stat().st_size / 2**20)
    row["store_mb"] = round(store.stat().st_size / 2**20)
    eventindex.close()
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[200_000, 1_000_000])
    ap.add_argument(
        "--profiles",
        nargs="+",
        default=["auto", "1", "4"],
        help="auto and/or memory budgets in GB",
    )
    ap.add_argument(
        "--work", type=Path, default=PROJECT_ROOT / "artifacts" / "speedbench"
    )
    ap.add_argument("--note", default="", help="e.g. 'measured while training ran'")
    args = ap.parse_args()
    import os

    os.environ["SPLITSHIELD_IN_TRIAL"] = "1"  # never start nightly work from here
    base = base_events()
    print(f"{len(base):,} parsed base events")
    doc = json.loads(OUT.read_text("utf-8")) if OUT.is_file() else {"runs": []}
    for n in args.sizes:
        work = args.work / f"n{n}"
        shutil.rmtree(work, ignore_errors=True)
        store = work / "events.jsonl"
        write_store(store, base, n, time.time())
        for prof in args.profiles:
            # same store, fresh index per profile (the build time is measured too)
            for f in work.glob("events.db*"):
                f.unlink()
            row = measure(store, prof, base)
            row["size"] = n
            row["when"] = time.strftime("%Y-%m-%d %H:%M")
            row["note"] = args.note
            print(json.dumps(row), flush=True)
            doc["runs"] = [
                r for r in doc["runs"] if (r["size"], r["profile"]) != (n, prof)
            ] + [row]
        shutil.rmtree(work, ignore_errors=True)
    doc["rows"] = [
        {
            "events": f"{r['size']:,}",
            "profile": r["profile"],
            "search (median)": statistics.median(r["search_ms"].values()),
            "search (slowest)": max(r["search_ms"].values()),
            "chat (median)": statistics.median(r["chat_ms"].values()),
            "chat (slowest)": max(r["chat_ms"].values()),
            "incidents (cached)": r["incidents_cached_ms"],
            "incidents (+1,000 new)": r["incidents_after_1000_new_ms"],
            "first index build (s)": r["index_build_s"],
            "note": r.get("note", ""),
        }
        for r in sorted(doc["runs"], key=lambda r: (r["size"], r["profile"]))
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2), "utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
