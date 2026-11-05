"""Full re-run in an isolated copy + reproducibility report (owner 2026-10-16).

    python -m scripts.full_rerun --original "C:/.../ML - XAL Project"

Run from inside the COPY (it refuses to run in the original). Steps:
  1. ai_engineer.py --fresh --parallel N  (checks, gate, every experiment and seed,
     E3 matrix, self-improve, deploys, NLP, corrected follow-ups, final report)
  2. the area / SOC pipelines (phishing, malware, ICS, insider, NetFlow, logs, extra
     datasets, H2 follow-up, robustness, hold-outs, PE scan, risk board), then the
     promotions behind every deployed model (PROMOTION_STEPS). Blind sets are NOT
     re-scored: they are scored once (owner rule). ``--areas-only`` skips step 1.
  3. REPRODUCIBILITY.md: every number in the copy's results/ against the registered
     results/ of the original, plus run time per step
  4. RERUN_DONE.json + a desktop pop-up

Safe to start again after a crash or power cut: finished steps are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "artifacts" / "full_rerun_state.json"
LOG = ROOT / "logs" / "full_rerun.log"
DONE = ROOT / "RERUN_DONE.json"
REPORT = ROOT / "REPRODUCIBILITY.md"
MALWARE_FREE_GB = 11.0  # the EMBER model peaks near 10 GB: runs with nothing beside it

# (key, module args). Run two at a time after the research pipeline.
AREA_STEPS: list[tuple[str, list[str]]] = [
    ("train_phishing", ["scripts.train_phishing"]),
    ("phishing_threshold", ["scripts.phishing_threshold", "--no-save"]),
    ("train_ics", ["scripts.train_ics"]),
    ("train_insider", ["scripts.train_insider"]),
    ("deploy_netflow", ["scripts.deploy_netflow"]),
    ("extra_datasets", ["scripts.extra_datasets"]),
    # one step per log area (2026-10-23: the bare module needs --area and exited at once);
    # Splunk attack_data (--blind) is the blind set: not here
    ("extra_logs_cloud", ["scripts.extra_logs", "--area", "cloud"]),
    ("extra_logs_zeek", ["scripts.extra_logs", "--area", "zeek"]),
    ("extra_logs_ait_dev", ["scripts.extra_logs", "--area", "ait-dev"]),
    ("extra_logs_ait_heldout", ["scripts.extra_logs", "--area", "ait-heldout"]),
    ("followup_h2", ["scripts.followup_h2"]),
    ("robust_deployed", ["scripts.robust_deployed"]),
    ("build_holdouts", ["scripts.build_holdouts"]),
    ("validate_pescan", ["scripts.validate_pescan"]),
    ("validate_riskboard", ["scripts.validate_riskboard"]),
    ("attackdata_followup", ["scripts.attackdata_followup", "score"]),
]
# the promotions behind every DEPLOYED model, in order, each with its own rule (2026-10-23):
# the copy then reproduces what is deployed, not only the first training runs
PROMOTION_STEPS: list[tuple[str, list[str]]] = [
    # phishing in its original order (2026-10-23: v4 was compared with an untuned champion
    # when the threshold step only reported): threshold saved -> host2 (v4) -> band
    ("phishing_threshold_saved", ["scripts.phishing_threshold"]),
    ("improve_phishing_host2_chain", ["scripts.improve_phishing", "--mode", "host2"]),
    ("phishing_band", ["scripts.phishing_band"]),
    ("improve_insider", ["scripts.improve_insider"]),
    ("improve_intent", ["scripts.improve_intent"]),
    ("improve_flows_cascade", ["scripts.improve_flows", "cascade"]),
    ("deploy_ddos_specialist", ["scripts.deploy_ddos_specialist", "--apply"]),  # owner exception
    ("ics_v3", ["scripts.train_ics_v3", "--no-save"]),
    ("ics_v3_deploy", ["scripts.train_ics_v3", "--deploy", "A", "--seeds", "0"]),  # owner exception
]
MALWARE_PROMOTION = ("improve_malware_reg", ["scripts.improve_malware", "--regularised"])
_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(f"[full-rerun] {msg}", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text("utf-8"))
    except (OSError, ValueError):
        return {"done": {}, "failed": {}}


def save_state(st: dict) -> None:
    with _lock:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2), "utf-8")
        tmp.replace(STATE)


def env(threads: int) -> dict[str, str]:
    e = dict(os.environ)
    e["PYTHONPATH"] = str(ROOT)
    e["PYTHONUTF8"] = "1"
    e["PYTHONIOENCODING"] = "utf-8"
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        e[var] = str(threads)
    e["LOKY_MAX_CPU_COUNT"] = str(threads)
    return e


def step(st: dict, key: str, cmd: list[str], threads: int) -> bool:
    if key in st["done"]:
        log(f"skip (done): {key}")
        return True
    log(f"START {key}")
    t0 = time.time()
    out = ROOT / "logs" / "full_rerun" / f"{key}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        rc = subprocess.run(
            cmd, cwd=ROOT, env=env(threads), stdout=fh, stderr=subprocess.STDOUT
        ).returncode
    mins = round((time.time() - t0) / 60, 1)
    if rc == 0:
        st["done"][key] = mins
        st["failed"].pop(key, None)
        log(f"OK    {key} in {mins} min")
    else:
        st["failed"][key] = f"exit {rc} after {mins} min (logs/full_rerun/{key}.log)"
        log(f"FAIL  {key} (exit {rc}) after {mins} min")
    save_state(st)
    return rc == 0


def free_gb() -> float:
    sys.path.insert(0, str(ROOT))
    from ai_engineer import free_ram_gb

    return free_ram_gb()


def area_steps(st: dict, workers: int) -> None:
    py = sys.executable
    threads = max(1, (os.cpu_count() or 4) // workers)
    last = [0.0]
    gate = threading.Lock()

    def one(key: str, mod: list[str]) -> None:
        while True:  # same RAM guard as the research runner
            with gate:
                if time.time() - last[0] >= 120 and free_gb() >= 6.0:
                    last[0] = time.time()
                    break
            time.sleep(30)
        step(st, key, [py, "-m", *mod], threads)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in [pool.submit(one, k, m) for k, m in AREA_STEPS]:
            f.result()
    # the malware model alone: it needs most of the RAM
    while free_gb() < MALWARE_FREE_GB and "train_malware" not in st["done"]:
        time.sleep(60)
    step(st, "train_malware", [py, "-m", "scripts.train_malware"], os.cpu_count() or 4)
    while free_gb() < MALWARE_FREE_GB and MALWARE_PROMOTION[0] not in st["done"]:
        time.sleep(60)
    step(st, MALWARE_PROMOTION[0], [py, "-m", *MALWARE_PROMOTION[1]], os.cpu_count() or 4)
    for key, mod in PROMOTION_STEPS:  # one at a time: they replace deployed models in order
        while free_gb() < 6.0 and key not in st["done"]:
            time.sleep(30)
        step(st, key, [py, "-m", *mod], os.cpu_count() or 4)


def _numbers(obj: object, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_numbers(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list) and len(obj) <= 200:
        for i, v in enumerate(obj):
            out.update(_numbers(v, f"{prefix}[{i}]"))
    elif isinstance(obj, int | float) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


_SKIP = ("time", "_s", "sec", "speed", "per_s", "rows_s", "duration", "ts", "date")


def compare(original: Path) -> dict:
    """Every numeric result in the original results/ vs the copy's, per file."""
    files, rows = 0, []
    same = close = differ = 0
    for f in sorted((original / "results").rglob("*.json")):
        rel = f.relative_to(original)
        g = ROOT / rel
        if not g.is_file():
            rows.append((str(rel), "missing in re-run", "", ""))
            continue
        try:
            a = _numbers(json.loads(f.read_text("utf-8")))
            b = _numbers(json.loads(g.read_text("utf-8")))
        except (OSError, ValueError):
            continue
        files += 1
        for k, va in a.items():
            if k not in b or any(s in k.lower() for s in _SKIP):
                continue
            vb, d = b[k], abs(b[k] - va)
            if d <= 1e-9:
                same += 1
            elif d <= 0.01 or d <= 0.02 * abs(va):
                close += 1
            else:
                differ += 1
                rows.append((str(rel), k, f"{va:.4g}", f"{vb:.4g}"))
    return {
        "files": files,
        "same": same,
        "close": close,
        "differ": differ,
        "rows": rows,
    }


def write_report(original: Path, st: dict, t0: float, ai_rc: int) -> None:
    cmp = compare(original)
    total = cmp["same"] + cmp["close"] + cmp["differ"]
    hours = (time.time() - t0) / 3600
    lines = [
        "# Full re-run: reproducibility report",
        "",
        f"Copy: `{ROOT}`  ·  original: `{original}` (its runs/ and results/ untouched)",
        f"Finished {time.strftime('%Y-%m-%d %H:%M')}; wall-clock **{hours:.1f} h**.",
        f"Research pipeline (ai_engineer.py --fresh --parallel): exit {ai_rc}"
        " (details: logs/ai_engineer.log).",
        "",
        "## Numbers compared",
        "",
        f"{cmp['files']} result files, {total} numbers: **{cmp['same']} identical**, "
        f"{cmp['close']} within 0.01 / 2 %, **{cmp['differ']} different**.",
        "Timing/speed fields are excluded (they depend on machine load).",
        "",
        "| file | value | registered | re-run |",
        "|---|---|---|---|",
        *[f"| {a} | {b} | {c} | {d} |" for a, b, c, d in cmp["rows"][:300]],
        "",
        "## Area / SOC steps",
        "",
        "| step | result |",
        "|---|---|",
        *[f"| {k} | OK, {v} min |" for k, v in st["done"].items()],
        *[f"| {k} | FAILED: {v} |" for k, v in st["failed"].items()],
    ]
    if len(cmp["rows"]) > 300:
        lines.insert(
            lines.index("## Area / SOC steps") - 1,
            f"... {len(cmp['rows']) - 300} more rows",
        )
    REPORT.write_text("\n".join(lines) + "\n", "utf-8")


def notify(text: str) -> None:
    """Desktop pop-up (notification only): Windows msg, macOS osascript, Linux
    notify-send; silently skipped when none is available."""
    msg = f"SplitShield-XAI: {text[:240]}"
    if sys.platform == "win32":
        cmd = ["msg", "*", "/TIME:86400", msg]
    elif sys.platform == "darwin":
        cmd = ["osascript", "-e", f"display notification {json.dumps(msg)} with title \"SplitShield-XAI\""]
    else:
        cmd = ["notify-send", "SplitShield-XAI", msg]
    try:
        subprocess.Popen(cmd)
    except OSError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--research-only", action="store_true",
                    help="only the research pipeline (2026-10-17: research runs start before "
                         "the SOC code freeze); writes RESEARCH_DONE.json")  # fmt: skip
    ap.add_argument("--areas-only", action="store_true",
                    help="only the area / SOC steps and the promotions (2026-10-23: the research "
                         "pipeline already ran twice in its own copies)")  # fmt: skip
    args = ap.parse_args()
    if ROOT.resolve() == args.original.resolve():
        raise SystemExit("refusing to run in the original project: use the copy")
    st = load_state()
    t0 = st.setdefault("started", time.time())
    save_state(st)
    log(f"begin; copy {ROOT}; original {args.original}")
    py = sys.executable
    ai_rc = 0
    for name, check in (
        ("lint", [py, "-m", "ruff", "check", "."]),
        ("types", [py, "-m", "mypy"]),
        ("tests", [py, "-m", "pytest", "-q", "-p", "no:warnings"]),
    ):  # broken code: stop here, before any training
        if subprocess.run(check, cwd=ROOT, env=env(4)).returncode != 0:
            log(f"{name} check failed - nothing started")
            notify(f"full re-run NOT started: {name} check failed")
            raise SystemExit(1)
    if args.areas_only:
        st["done"].setdefault("research", "skipped (--areas-only)")
        # reads this copy's research runs/ (none here): run it on the research copies
        # instead (python -m scripts.followup_h2 --root <research copy>)
        st["done"].setdefault("followup_h2", "skipped (--areas-only: needs the research runs)")
    if "research" not in st["done"]:
        log("START research (ai_engineer.py)")
        fresh = [] if st.get("research_started") else ["--fresh"]
        st["research_started"] = True
        save_state(st)
        ai = [
            py,
            "ai_engineer.py",
            *fresh,
            "--skip-intake",
            "--parallel",
            str(args.parallel),
        ]
        ai_rc = subprocess.run(ai, cwd=ROOT, env=env(os.cpu_count() or 4)).returncode
        if ai_rc == 0:
            st["done"]["research"] = round((time.time() - t0) / 60, 1)
        else:
            st["failed"]["research"] = f"exit {ai_rc} (logs/ai_engineer.log)"
        save_state(st)
        log(f"research finished, exit {ai_rc}")
    if args.research_only:
        write_report(args.original, st, t0, ai_rc)
        (ROOT / "RESEARCH_DONE.json").write_text(json.dumps({
            "finished": time.strftime("%Y-%m-%d %H:%M"), "hours": round((time.time() - t0) / 3600, 1),
            "failed": st["failed"], "research_exit": ai_rc}, indent=2), "utf-8")  # fmt: skip
        notify(f"research run finished in {ROOT.name} (exit {ai_rc})")
        return
    area_steps(st, workers=2)
    write_report(args.original, st, t0, ai_rc)
    DONE.write_text(
        json.dumps(
            {
                "finished": time.strftime("%Y-%m-%d %H:%M"),
                "hours": round((time.time() - t0) / 3600, 1),
                "failed": st["failed"],
            },
            indent=2,
        ),
        "utf-8",
    )
    n = len(st["failed"])
    notify(f"full re-run finished ({n} failed step(s)). See {REPORT}")
    log("done")


if __name__ == "__main__":
    main()
