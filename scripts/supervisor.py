"""Unattended run supervisor: keeps ai_engineer.py going until everything is done.

    .venv\\Scripts\\python.exe scripts\\supervisor.py          # start / adopt / resume
    .venv\\Scripts\\python.exe scripts\\supervisor.py --status # print state and exit

Rules (owner-approved, 2026-10-13):
  R1 One supervisor at a time (lock file). A run already in progress, started by
     hand, is adopted: the watchdog waits for it instead of starting a second one.
  R2 Resume, never redo: every launch is plain ``ai_engineer.py`` (no --fresh), so
     finished phases are skipped.
  R3 Crash / power cut / reboot: relaunch. A scheduled task starts this script at
     logon, so an interrupted run continues after a restart.
  R4 A failed phase is retried, at most 3 attempts per phase; then it is left
     failed and the rest of the run continues.
  R5 Out of memory in a phase: before the retry, that dataset's per-file row sample
     is halved (1.0 -> 0.5 -> 0.25, never lower), written to
     config/memory_overrides.json and recorded in results/MEMORY_DEVIATIONS.md.
  R6 Stop and wait for the owner (state "halted") when the problem is not
     retryable: lint/type/test checks fail, the data gate fails, or 3 launches in
     a row finish no new phase.
  R7 No git commits during the run (``.no_autocommit`` marker).
  R8 Keep the PC awake while supervising; done -> remove the logon task and exit.
  R9 Queued work (artifacts/queued_phases.json, owner-approved additions such as the
     2026-10-14 corrected follow-up): a run that "finished" while a queued phase is
     not yet done is relaunched, so phases added after the engine started still run.

Everything is logged to logs/watchdog.log; the current state is in
artifacts/watchdog_state.json and results/WATCHDOG_STATUS.md.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
ENGINE_LOG = ROOT / "logs" / "ai_engineer.log"
RUN_LOG = ROOT / "logs" / "full_run.log"
LOG = ROOT / "logs" / "watchdog.log"
STATE = ROOT / "artifacts" / "watchdog_state.json"
LOCK = ROOT / "artifacts" / "watchdog.lock"
STATUS_MD = ROOT / "results" / "WATCHDOG_STATUS.md"
MEM_JSON = ROOT / "config" / "memory_overrides.json"
MEM_MD = ROOT / "results" / "MEMORY_DEVIATIONS.md"
TASK_NAME = "SplitShield-XAI watchdog"

MAX_ATTEMPTS = 3  # R4
MIN_SCALE = 0.25  # R5
MAX_IDLE_LAUNCHES = 3  # R6
OOM = re.compile(
    r"MemoryError|Unable to allocate|not enough memory|bad allocation|"
    r"out of memory|DefaultCPUAllocator|std::bad_alloc|"
    r"exit (3221225477|-1073741819|3221226505|-1073740791|3221225725|-1073741571)\)",
    re.IGNORECASE,
)
FATAL = re.compile(r"check failed - not starting training|aborting: gate failed")


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(f"[watchdog] {msg}", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state() -> dict[str, Any]:
    try:
        return json.loads(STATE.read_text("utf-8"))
    except (OSError, ValueError):
        return {"status": "running", "attempts": {}, "idle_launches": 0, "launches": 0}


def save_state(st: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2), "utf-8")
    tmp.replace(STATE)
    write_status(st)


def write_status(st: dict[str, Any]) -> None:
    done = sorted(_done_phases())
    lines = [
        "# Unattended run status",
        "",
        f"- updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- watchdog state: **{st['status']}**"
        + (f" - {st['reason']}" if st.get("reason") else ""),
        f"- launches by watchdog: {st.get('launches', 0)}",
        f"- phases finished ({len(done)}): {', '.join(done) or '-'}",
        f"- retry attempts: {json.dumps(st.get('attempts', {}))}",
        f"- memory overrides: {json.dumps(_mem().get('datasets', {}))}",
        "",
        "Latest automatic fit checks:",
        *(_fit_lines() or ["- none yet"]),
        "",
        "Logs: logs/full_run.log (everything), logs/ai_engineer.log (phases), "
        "logs/watchdog.log (decisions). After fixing a halt, delete "
        "artifacts/watchdog_state.json and start scripts/supervisor.py again.",
    ]
    STATUS_MD.parent.mkdir(parents=True, exist_ok=True)
    STATUS_MD.write_text("\n".join(lines) + "\n", "utf-8")


def _fit_lines(n: int = 8) -> list[str]:
    """Newest fit verdicts the pipeline logged (src/eval/fitcheck.py)."""
    try:
        lines = ENGINE_LOG.read_text("utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return ["- " + ln.split(" FIT ", 1)[1] for ln in lines if " FIT " in ln][-n:]


QUEUE = ROOT / "artifacts" / "queued_phases.json"


def queued_pending() -> list[str]:
    """R9: queued phase keys not finished yet ([] when nothing is queued)."""
    try:
        keys = json.loads(QUEUE.read_text("utf-8")).get("phases", [])
    except (OSError, ValueError):
        return []
    done = _done_phases()
    return [k for k in keys if k not in done]


def _done_phases() -> set[str]:
    try:
        st = json.loads(
            (ROOT / "artifacts" / "ai_engineer_state.json").read_text("utf-8")
        )
        return set(st.get("full", []))
    except (OSError, ValueError):
        return set()


def _mem() -> dict[str, Any]:
    try:
        return json.loads(MEM_JSON.read_text("utf-8"))
    except (OSError, ValueError):
        return {"datasets": {}}


def _pid_alive(pid: int) -> bool:
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
    ).stdout
    return str(pid) in out


def engine_running() -> bool:
    """True if any ai_engineer.py process is alive (R1: adopt a manual run)."""
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -match 'ai_engineer\\.py' } | "
        "Measure-Object | Select-Object -ExpandProperty Count"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True
    ).stdout.strip()
    return out.isdigit() and int(out) > 0


def keep_awake() -> None:
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED (R8)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000041)


def last_run_segment() -> tuple[list[str], str]:
    """Engine-log lines since the last 'begin (full)', and the matching full-run text."""
    lines = ENGINE_LOG.read_text("utf-8", errors="replace").splitlines()
    starts = [i for i, ln in enumerate(lines) if "begin (full)" in ln]
    seg = lines[starts[-1] :] if starts else []
    try:
        text = RUN_LOG.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
    except OSError:
        text = ""
    k = text.rfind("begin (full)")
    return seg, text[k:] if k >= 0 else text


def phase_output(text: str, key: str) -> str:
    """Console output of the LAST attempt of one phase."""
    a = text.rfind(f"START {key}")
    if a < 0:
        return ""
    b = text.find("[ai-engineer] START ", a + 10)
    return text[a : b if b > 0 else len(text)]


def dataset_of(key: str) -> str | None:
    kind, _, rest = key.partition(":")
    if kind in ("improve", "deploy"):
        return rest
    if kind == "e3_matrix":
        exp = "E3_alpha_skew"
    elif kind == "train":
        exp = rest.split(":")[0]
    elif kind == "followup":  # R9 corrected follow-up configs live in a subfolder
        exp = "followup/" + rest.split(":")[0]
    else:
        return None
    try:
        raw = yaml.safe_load(
            (ROOT / "config" / "experiments" / f"{exp}.yaml").read_text("utf-8")
        )
        return str(raw.get("dataset"))
    except (OSError, yaml.YAMLError, AttributeError):
        return None


def shrink(dataset: str, key: str) -> bool:
    """R5: halve the dataset's row scale; False once already at the floor."""
    doc = _mem()
    cur = float(doc.setdefault("datasets", {}).get(dataset, 1.0))
    if cur <= MIN_SCALE:
        return False
    new = max(MIN_SCALE, cur / 2)
    doc["datasets"][dataset] = new
    MEM_JSON.write_text(json.dumps(doc, indent=2), "utf-8")
    first = not MEM_MD.exists()
    with open(MEM_MD, "a", encoding="utf-8") as fh:
        if first:
            fh.write(
                "# Memory deviations (number-affecting)\n\n"
                "Applied automatically by scripts/supervisor.py (rule R5) after a phase "
                "ran out of memory on this 16 GB machine. Each source file of the "
                "dataset is randomly sub-sampled (seed 0) to the given fraction before "
                "preprocessing; class caps then apply as usual. Results for these "
                "datasets use less raw data than the design specifies.\n\n"
                "| time | phase | dataset | row fraction |\n|---|---|---|---|\n"
            )
        fh.write(
            f"| {time.strftime('%Y-%m-%d %H:%M')} | {key} | {dataset} | {cur:g} -> {new:g} |\n"
        )
    log(f"R5 out of memory in {key}: {dataset} row fraction {cur:g} -> {new:g}")
    return True


def notify(text: str) -> None:
    """Desktop pop-up for the logged-in user (msg.exe; best effort, never fatal)."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return  # tests exercise halt/done; no real pop-ups
    try:
        subprocess.Popen(
            ["msg", "*", "/TIME:86400", f"SplitShield-XAI: {text[:240]}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def halt(st: dict[str, Any], reason: str) -> None:
    st["status"], st["reason"] = "halted", reason
    save_state(st)
    log(f"HALTED (needs the owner): {reason}")
    notify(f"training STOPPED and needs you: {reason}. See results/WATCHDOG_STATUS.md")


def judge(st: dict[str, Any]) -> bool:
    """Inspect the run that just ended. Returns True if another launch is warranted."""
    seg, text = last_run_segment()
    if FATAL.search(text):
        m = FATAL.search(text)
        halt(st, f"not retryable: '{m.group(0) if m else ''}' (see logs/full_run.log)")
        return False
    finished = any("finished in" in ln for ln in seg)
    failed: list[str] = []
    for ln in seg:
        if "FAILED phases" in ln:
            tail = ln.split("only these):", 1)[-1]  # keys contain ":" themselves
            failed = [s.strip() for s in tail.split(",") if s.strip()]
    if finished and not failed and (pending := queued_pending()):
        log(
            f"R9 queued work not done yet ({len(pending)} phase(s), e.g. {pending[0]}) -> relaunch"
        )
        return True
    if finished and not failed:
        st["status"], st["reason"] = "done", "every phase finished"
        save_state(st)
        log("DONE: every phase finished; results/FINAL_REPORT.md is final")
        notify("training finished. Report: results/FINAL_REPORT.md")
        return False
    if not finished:
        log(
            "previous run ended before finishing (crash / power cut / closed) -> resume"
        )
        return True
    retry = False
    for key in failed:
        n = st["attempts"].get(key, 1) + 1
        if n > MAX_ATTEMPTS:
            log(f"R4 {key}: {MAX_ATTEMPTS} attempts used; left failed")
            continue
        out = phase_output(text, key)
        ds = dataset_of(key)
        if OOM.search(out) and ds:
            if not shrink(ds, key):
                log(f"R5 {key}: out of memory at the minimum row fraction; left failed")
                st["attempts"][key] = MAX_ATTEMPTS
                continue
        st["attempts"][key] = n
        log(f"R4 retry {key} (attempt {n}/{MAX_ATTEMPTS})")
        retry = True
    if not retry:
        st["status"], st["reason"] = "finished_with_failures", ", ".join(failed)
        save_state(st)
        log(f"finished; still failed after retries: {', '.join(failed)}")
        notify(f"training finished with failures: {', '.join(failed)}")
    return retry


def launch() -> None:
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "LOKY_MAX_CPU_COUNT": str(os.cpu_count() or 4),
    }
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "ab") as fh:
        subprocess.run(
            [str(PY), "ai_engineer.py"],
            cwd=ROOT,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )


def remove_logon_task() -> None:
    subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"], capture_output=True)


def main() -> int:
    st = load_state()
    if "--status" in sys.argv:
        write_status(st)
        print(STATUS_MD.read_text("utf-8"))
        return 0
    if st["status"] in ("halted", "done"):
        log(f"state is '{st['status']}' ({st.get('reason', '')}); nothing to do")
        return 0

    # R1 single instance
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip())
            if pid != os.getpid() and _pid_alive(pid):
                print(f"[watchdog] already running (pid {pid})")
                return 0
        except ValueError:
            pass
    LOCK.write_text(str(os.getpid()))
    (ROOT / ".no_autocommit").touch()  # R7

    try:
        keep_awake()
        log("watchdog started")
        # one judgement per finished run: wait (adopt) -> judge -> launch -> repeat
        while True:
            if engine_running():  # R1 adopt a run started by hand
                save_state(st)
                log("a run is already in progress; supervising it")
                while engine_running():
                    keep_awake()
                    time.sleep(60)
                log("that run ended")
            if not judge(st):
                break
            before = len(_done_phases())
            st["launches"] = st.get("launches", 0) + 1
            st["status"], st["reason"] = "running", ""
            save_state(st)
            log(f"launch #{st['launches']}: ai_engineer.py (resume)")
            launch()
            gained = len(_done_phases()) - before
            st["idle_launches"] = 0 if gained else st.get("idle_launches", 0) + 1
            log(f"run ended; {gained} new phase(s) finished")
            if st["idle_launches"] >= MAX_IDLE_LAUNCHES:  # R6 crash loop
                halt(st, f"{MAX_IDLE_LAUNCHES} launches in a row finished nothing")
                break
    finally:
        LOCK.unlink(missing_ok=True)
    if st["status"] in ("done", "finished_with_failures"):
        remove_logon_task()  # R8: nothing left to resume
    return 0


if __name__ == "__main__":
    sys.exit(main())
