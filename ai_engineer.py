"""AI-engineer pipeline: data intake -> training -> testing -> validation -> report.

    python ai_engineer.py              # FULL real run (hours; GPU recommended)
    python ai_engineer.py --fast       # whole chain on the tiny fixture (minutes)
    python ai_engineer.py --fresh      # ignore saved progress and redo every phase

Phases (each an isolated subprocess, so memory is released between them):
  0 INTAKE    unzip / identify / place every dataset found (nested zips too)
  1 GATE      blocking data-integrity gates (rare-class, leakage, shuffled-label)
  2 TRAIN+TEST every experiment with data, every seed: baselines (RF, GBM, MLP),
              4 training schemes, SHAP/LIME + attack explanations, attacks, cut-depth
              (H1), drift (H2), defences (H3), transfer
  3 E3        training scheme x label skew x seeds (H4), resumable
  4 IMPROVE   self-improvement / fine-tuning (+USL, continual, HPO) per dataset
  5 DEPLOY    freeze the detector (+ D3 energy threshold) for the dashboard
  6 NLP       intent model + ShieldLM (own language model) + RLCD alignment
  7 VALIDATE  aggregate + results/FINAL_REPORT.md (H1-H5 verdicts vs controls)

Robustness: progress is saved after every phase (artifacts/ai_engineer_state.json),
so re-running after a crash or power cut continues where it stopped; a failed phase
is retried once and then recorded without stopping the others; missing datasets are
skipped; the exit code is non-zero if anything failed, with the list printed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "config" / "experiments"
RAW_DIR = ROOT / "data" / "raw"
STATE = ROOT / "artifacts" / "ai_engineer_state.json"
LOG = ROOT / "logs" / "ai_engineer.log"
RAW_DEPLOY = "cicids2017-original"  # raw-unit CIC-IDS2017 for the live detector
PRIMARY = "E4_cut_depth"  # carries H1-H3 (improved CIC-IDS2017)
MATRIX = "E3_alpha_skew"  # H4, run by run_matrix.py
NON_PIPELINE = {"smoke", MATRIX}

failures: list[str] = []
_STATE_LOCK = threading.Lock()  # parallel phases mark progress from threads
PARALLEL = 2  # follow-up phases run side by side (owner 2026-10-14)
MIN_FREE_GB = 6.0  # a second parallel phase starts only with this much free RAM


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(f"[ai-engineer] {msg}", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


class Progress:
    """Phase keys already completed (persisted after each success)."""

    def __init__(self, fresh: bool, tag: str) -> None:
        self.tag = tag
        self.done: set[str] = set()
        if fresh and STATE.is_file():
            # 2026-10-17: _mark() merges what is on disk, so with the old file left in
            # place the first finished phase brought back every saved phase and --fresh
            # skipped them all (a copied project's state). Set it aside, never deleted.
            STATE.replace(STATE.with_name(f"{STATE.stem}.before-fresh-{int(time.time())}.json"))
        if STATE.is_file() and not fresh:
            try:
                self.done = set(json.loads(STATE.read_text("utf-8")).get(tag, []))
            except (OSError, ValueError):
                self.done = set()

    def mark(self, key: str) -> None:
        with _STATE_LOCK:
            self._mark(key)

    def _mark(self, key: str) -> None:
        self.done.add(key)
        STATE.parent.mkdir(parents=True, exist_ok=True)
        try:
            cur = json.loads(STATE.read_text("utf-8")) if STATE.is_file() else {}
        except (OSError, ValueError):
            cur = {}
        # union with what is on disk: never drop a phase another writer recorded
        self.done |= set(cur.get(self.tag, []))
        cur[self.tag] = sorted(self.done)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cur, indent=2), "utf-8")
        tmp.replace(STATE)  # atomic: a power cut never leaves a half-written file


CHILD_ENV: dict[str, str] | None = None  # --parallel: per-job CPU thread caps


def phase(
    prog: Progress,
    key: str,
    cmd: list[str],
    *,
    critical: bool = False,
    retries: int = 1,
) -> bool:
    """Run one phase unless already done; retry, then record the failure."""
    if key in prog.done:
        _log(f"skip (already done): {key}")
        return True
    for attempt in range(retries + 1):
        _log(f"START {key}" + (f"  (retry {attempt})" if attempt else ""))
        t0 = time.time()
        try:
            rc = subprocess.run(cmd, cwd=ROOT, env=CHILD_ENV).returncode
        except OSError as exc:
            rc = -1
            _log(f"could not start: {exc}")
        dt = (time.time() - t0) / 60
        if rc == 0:
            _log(f"OK    {key} in {dt:.1f} min")
            prog.mark(key)
            return True
        _log(f"FAIL  {key} (exit {rc}) after {dt:.1f} min")
    if critical:
        raise SystemExit(f"[ai-engineer] aborting: {key} failed.")
    failures.append(key)
    return False


def _has_data(dataset: str) -> bool:
    d = RAW_DIR / dataset
    return d.is_dir() and any(
        p.suffix.lower() in (".csv", ".parquet") and "__MACOSX" not in p.parts
        for p in d.rglob("*")
    )


ATTACK_STAGES = (4, 6, 7)  # need a reviewed attacker mutability map
STAGE_LIMIT: dict[str, set[int]] = {}  # experiment -> its YAML ``stages`` (if any)
NO_IMPROVE: set[str] = set()  # datasets whose YAML says ``improve: false``


def _mapped(dataset: str) -> bool:
    """True if every feature of ``dataset`` has a reviewed mutability class.

    Read from the file header only. Without a map the attack stages cannot run
    (owner decision 2026-10-13: skip them as not applicable rather than fail and
    retry; NetFlow v2 has a draft map pending review, config/constraints_netflow_draft.py).
    """
    import pandas as pd

    from src.attack.constraints import resolve_classes
    from src.data.sources import get_spec

    spec = get_spec(dataset)
    files = sorted(
        p
        for p in (RAW_DIR / dataset).rglob("*")
        if p.suffix.lower() in (".csv", ".parquet") and "__MACOSX" not in p.parts
    )
    if not files:
        return True  # no data: the experiment is skipped elsewhere anyway
    if files[0].suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        cols = list(pq.read_schema(files[0]).names)
    else:
        cols = list(spec.column_names) or list(pd.read_csv(files[0], nrows=0).columns)
    drop = {spec.label_col, *spec.id_columns}
    feats = [c.strip() for c in dict.fromkeys(cols) if c.strip() not in drop]
    try:
        resolve_classes(feats)
        return True
    except ValueError:
        return False


def _log_fit(exp: str, seed: int) -> None:
    """Log the automatic fit check of the newest run of (exp, seed): one line in
    logs/ai_engineer.log as soon as the phase ends (src/eval/fitcheck.py)."""
    runs = sorted((ROOT / "runs").glob(f"{exp}__seed{seed}__*"), key=lambda d: d.name)
    for d in reversed(runs):
        f = d / "metrics" / "stage1_baseline.json"
        if f.is_file():
            try:
                fit = json.loads(f.read_text("utf-8"))["metrics"]["mlp_fit"]["value"]
            except (OSError, ValueError, KeyError):
                return
            _log(
                f"FIT {exp} seed{seed}: {fit['verdict']} (train {fit['train_macro_f1']:.3f}"
                f" / val {fit['best_val_macro_f1']:.3f}, best epoch {fit['best_epoch']}"
                f"/{fit['epochs_run']})"
                + (f" - {'; '.join(fit['findings'])}" if fit["findings"] else "")
            )
            return


def _experiments(only: str | None) -> list[tuple[str, str, list[int]]]:
    """(experiment id, dataset, seeds) for every pipeline experiment with data."""
    out = []
    for f in sorted(EXP_DIR.glob("*.yaml")):
        try:
            raw = yaml.safe_load(f.read_text("utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            _log(f"skip {f.name}: unreadable config ({exc})")
            continue
        exp = raw.get("id", f.stem)
        if exp in NON_PIPELINE or (only and exp != only):
            continue
        ds = raw.get("dataset", "cicids2017-improved")
        if not _has_data(ds):
            _log(f"skip {exp}: no data for {ds} (put its zip next to the project)")
            continue
        out.append((exp, ds, list(raw.get("seeds") or [0])))
        if raw.get("improve") is False:
            NO_IMPROVE.add(ds)
        if raw.get("stages"):
            STAGE_LIMIT[exp] = {int(x) for x in raw["stages"]}
    return out


def free_ram_gb() -> float:
    """Available physical memory in decimal GB on Windows, Linux and macOS
    (utils/sysinfo.py; scheduling only, never changes what is computed)."""
    from utils.sysinfo import free_ram_gb as _f

    return _f() * (2**30 / 1e9)


Chain = list[tuple[str, list[str]]]  # phases that must run one after another
SETTLE_S = 180  # --parallel: a new job waits this long after the previous start,
# so the RAM check sees the dataset the previous job is still loading


def run_parallel(
    prog: Progress, jobs: list[tuple[str, list[str]]], workers: int
) -> None:
    """Run independent phases ``workers`` at a time. A slot only starts a phase
    when another one is running if at least MIN_FREE_GB RAM is free (each run
    re-reads its dataset; an out-of-memory crash would cost more than waiting)."""
    run_chains(prog, [[job] for job in jobs], workers, settle=0)


def run_chains(
    prog: Progress, chains: list[Chain], workers: int, settle: float = SETTLE_S
) -> None:
    """Run independent chains ``workers`` at a time (phases inside a chain run in
    order). Same RAM guard as run_parallel, plus ``settle`` seconds between starts."""
    from concurrent.futures import ThreadPoolExecutor

    active = [0]
    last_start = [0.0]
    lock = threading.Lock()

    def one(chain: Chain) -> None:
        if all(key in prog.done for key, _ in chain):
            for key, _ in chain:
                _log(f"skip (already done): {key}")
            return
        while True:
            with lock:
                ready = time.time() - last_start[0] >= settle
                if active[0] == 0 or (ready and free_ram_gb() >= MIN_FREE_GB):
                    active[0] += 1
                    last_start[0] = time.time()
                    break
            time.sleep(30)
        try:
            for key, cmd in chain:
                ok = phase(prog, key, cmd, retries=0)
                if ok and key.startswith("train:"):
                    _, exp, seed = key.split(":")
                    _log_fit(exp, int(seed.removeprefix("seed")))
                if not ok:
                    break  # later phases of the chain depend on this one
        finally:
            with lock:
                active[0] -= 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, c) for c in chains]
        for f in futs:
            f.result()


def past_minutes() -> dict[str, float]:
    """Duration of each phase in earlier full runs (logs/ai_engineer.log), used to
    start the longest jobs first so the run does not end on one long job alone."""
    out: dict[str, float] = {}
    if not LOG.is_file():
        return out
    for line in LOG.read_text("utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[2] == "OK" and parts[4] == "in":
            try:
                out[parts[3]] = float(parts[5])
            except ValueError:
                continue
    return out


def thread_env(workers: int) -> dict[str, str]:
    """Child environment that splits the CPU cores between ``workers`` jobs."""
    import os

    n = str(max(1, (os.cpu_count() or 4) // workers))
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = n
    env["LOKY_MAX_CPU_COUNT"] = n
    return env


def _followups() -> list[tuple[str, str, list[int]]]:
    """(experiment id, dataset, seeds) of the corrected follow-ups with data."""
    out = []
    for f in sorted((EXP_DIR / "followup").glob("C_*.yaml")):
        raw = yaml.safe_load(f.read_text("utf-8"))
        ds = raw.get("dataset", "cicids2017-improved")
        if _has_data(ds) and _mapped(ds):
            out.append((f.stem, ds, list(raw.get("seeds") or [0])))
    return out


def _train_cmd(py: str, exp: str, ds: str, seed: int, first: bool) -> list[str]:
    """run_pipeline.py command of one (experiment, seed) training phase."""
    transfer = exp == "E8_netflow"
    attacks = _mapped(ds)
    stages = (
        [1, 2, 3]
        + (list(ATTACK_STAGES) if attacks else [])
        + ([0, 5] if first else [])
        + ([8] if transfer else [])
    )
    if exp in STAGE_LIMIT:
        stages = [s for s in stages if s in STAGE_LIMIT[exp] | {0}]
    stage_arg = ",".join(str(s) for s in sorted(stages))
    cmd = [py, "run_pipeline.py", "--experiment", exp, "--seed", str(seed)]
    return [*cmd, "--resume", "--stage", stage_arg]


def full_parallel(
    prog: Progress,
    plan: list[tuple[str, str, list[int]]],
    py: str,
    args: argparse.Namespace,
) -> None:
    """The full run with independent work side by side (owner 2026-10-16: re-run in
    under 24 h). Same phases and commands as the sequential path; only the order
    differs. Self-improvement writes shared files (leaderboard, auto.yaml), so it
    stays one sequential chain, followed by the deploys that read its result."""
    chains: list[Chain] = []
    for exp, ds, seeds in plan:
        for i, seed in enumerate(seeds):
            key = f"train:{exp}:seed{seed}"
            chains.append([(key, _train_cmd(py, exp, ds, seed, i == 0))])
    if _has_data("cicids2017-improved"):
        chains.append(
            [("e3_matrix", [py, "run_matrix.py", "--experiment", MATRIX, "--resume"])]
        )
    chains.append([("nlp", [py, "-m", "src.nlp.train"])])
    datasets = sorted({ds for _, ds, _ in plan} - NO_IMPROVE)
    target = "cicids2017-improved" if "cicids2017-improved" in datasets else datasets[0]
    deploy: Chain = []
    if not args.skip_improve:
        for ds in datasets:
            cmd = [py, "improve.py", "--dataset", ds, "--trials", str(args.trials)]
            deploy.append(
                (f"improve:{ds}", [*cmd, "--usl", "--continual", "--hpo", "--report"])
            )
    deploy += [
        (f"deploy:{target}", [py, "deploy.py", "--dataset", target]),
        (f"soc_selftest:{target}", [py, "-m", "scripts.soc_selftest"]),
    ]
    if _has_data(RAW_DEPLOY):
        deploy += [
            ("deploy_backup", [py, "-m", "scripts.backup_deploy"]),
            (f"deploy:{RAW_DEPLOY}", [py, "deploy.py", "--dataset", RAW_DEPLOY]),
            (f"soc_selftest:{RAW_DEPLOY}", [py, "-m", "scripts.soc_selftest"]),
        ]
    chains.append(deploy)
    if not args.experiment:
        phase(prog, "data_quality:v2", [py, "-m", "scripts.data_quality"])
        for exp, _ds, seeds in _followups():
            for seed in seeds:
                cmd = [py, "run_pipeline.py", "--experiment", f"followup/{exp}"]
                cmd += ["--seed", str(seed), "--stage", "4,6,7", "--resume"]
                chains.append([(f"followup:{exp}:seed{seed}", cmd)])
    past = past_minutes()
    chains.sort(key=lambda c: -sum(past.get(k, 30.0) for k, _ in c))
    _log(f"parallel: {len(chains)} jobs, {args.parallel} at a time, longest first")
    run_chains(prog, chains, args.parallel)
    # stage-1 re-run of runs made before the GBM class-weight fix (needs every run)
    phase(prog, "baselines_fix", [py, "-m", "scripts.rerun_baselines"])


def main() -> None:
    ap = argparse.ArgumentParser(description="AI-engineer: train, test, validate")
    ap.add_argument("--fast", action="store_true", help="whole chain on the fixture")
    ap.add_argument("--experiment", default=None, help="run only this experiment")
    ap.add_argument("--trials", type=int, default=6, help="self-improve RL episodes")
    ap.add_argument("--fresh", action="store_true", help="ignore saved progress")
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--skip-improve", action="store_true")
    ap.add_argument(
        "--skip-intake", action="store_true", help="datasets already placed (copies)"
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="full run: this many independent phases side by side (CPU split evenly)",
    )
    args = ap.parse_args()
    global CHILD_ENV
    if args.parallel > 1 and not args.fast:
        CHILD_ENV = thread_env(args.parallel)

    py = sys.executable
    t0 = time.time()
    prog = Progress(args.fresh, "fast" if args.fast else "full")
    free = shutil.disk_usage(ROOT).free / 1e9
    mode = "fast" if args.fast else "full"
    _log(f"begin ({mode}); {free:.0f} GB free; python {sys.version.split()[0]}")
    if free < 10:
        _log("WARNING: under 10 GB free disk - large datasets may fail to unpack/train")

    # 0. CHECKS - every run, fast or full, never cached as "done": the code may
    # have changed since the last run. Broken code stops here, before hours of training.
    for name, check in (
        ("lint", [py, "-m", "ruff", "check", "."]),
        ("types", [py, "-m", "mypy"]),
        ("tests", [py, "-m", "pytest", "-q", "-p", "no:warnings"]),
    ):
        _log(f"check: {name}")
        if subprocess.run(check, cwd=ROOT).returncode != 0:
            raise SystemExit(
                f"[ai-engineer] {name} check failed - not starting training."
            )

    # INTAKE - every run (cheap when nothing changed). One bad archive is reported,
    # not fatal: the experiments whose data did land still run.
    if (
        not args.fast
        and not args.skip_intake
        and subprocess.run([py, "-m", "src.data.intake"], cwd=ROOT).returncode
    ):
        _log("WARNING: intake reported a failure - see data/raw/.intake.json")

    if args.fast:
        plan = [("smoke", "fixture", [0])]
        fast, primary, matrix_ds = ["--fast"], "smoke", ["--dataset", "fixture"]
    else:
        plan = _experiments(args.experiment)
        fast, primary, matrix_ds = [], PRIMARY, []
        if not plan:
            raise SystemExit(
                "[ai-engineer] no experiment has data. Put the dataset zips in the "
                "project folder (or a 'datasets' folder) and run again."
            )

    # 1. GATE
    if args.fast:
        phase(
            prog,
            "gate",
            [py, "-m", "scripts.gate_check", "--fast"],
            critical=True,
            retries=0,
        )
    elif not args.skip_gate and _has_data("cicids2017-improved"):
        phase(prog, "gate", [py, "-m", "scripts.gate_check"], critical=True, retries=0)

    if args.parallel > 1 and not args.fast:
        full_parallel(prog, plan, py, args)
        _validate(py, primary, t0)
        return

    # 2. TRAIN + TEST. Stage 5 loops its own seeds -> first seed only. Stage 8 is the
    # fixed NetFlow transfer pair (independent of the experiment's dataset) -> only in
    # the transfer experiment (or the fast smoke), not reloaded for every run.
    # retries=0: a deterministic failure would just burn hours again; re-running the
    # script redoes only the phases that did not finish.
    for exp, ds, seeds in plan:
        transfer = args.fast or exp == "E8_netflow"
        attacks = args.fast or _mapped(ds)
        if not attacks:
            _log(
                f"{exp}: no reviewed mutability map for {ds}; attack stages "
                f"{list(ATTACK_STAGES)} not applicable (skipped)"
            )
        for i, seed in enumerate(seeds):
            stages = (
                [1, 2, 3]
                + (list(ATTACK_STAGES) if attacks else [])
                + ([0, 5] if i == 0 else [])
                + ([8] if transfer else [])
            )
            if exp in STAGE_LIMIT:
                stages = [s for s in stages if s in STAGE_LIMIT[exp] | {0}]
            # --resume: after a power cut / crash, continue the unfinished run's
            # remaining stages instead of starting the whole phase again
            cmd = [py, "run_pipeline.py", "--experiment", exp, "--seed", str(seed)]
            cmd += [] if args.fast else ["--resume"]
            key = f"train:{exp}:seed{seed}"
            stage_arg = ",".join(str(s) for s in sorted(stages))
            if phase(prog, key, [*cmd, "--stage", stage_arg, *fast], retries=0):
                _log_fit(exp, seed)

    # 2b. Re-run stage 1 of runs made before the GBM class-weight fix (no-op after).
    if not args.fast:
        phase(prog, "baselines_fix", [py, "-m", "scripts.rerun_baselines"])

    # 3. E3 matrix (H4).
    if args.fast or (not args.experiment and _has_data("cicids2017-improved")):
        cmd = [py, "run_matrix.py", "--experiment", MATRIX, "--resume"]
        phase(prog, "e3_matrix", [*cmd, *fast, *matrix_ds])

    # 4-5. IMPROVE + DEPLOY
    datasets = sorted({ds for _, ds, _ in plan} - NO_IMPROVE)
    if not args.skip_improve:
        for ds in datasets:
            cmd = [py, "improve.py", "--dataset", ds, "--trials", str(args.trials)]
            phase(
                prog,
                f"improve:{ds}",
                [*cmd, "--usl", "--continual", "--hpo", "--report", *fast],
            )
    if args.fast:
        target = "fixture"
    else:
        target = (
            "cicids2017-improved" if "cicids2017-improved" in datasets else datasets[0]
        )
    phase(prog, f"deploy:{target}", [py, "deploy.py", "--dataset", target, *fast])
    # SOC self-test: the deployed detector end to end on held-out labelled flows
    # (alerting per tier, ATT&CK mapping, incidents, SIEM delivery, assistant)
    st = ["--max-flows", "2000"] if args.fast else []
    phase(prog, f"soc_selftest:{target}", [py, "-m", "scripts.soc_selftest", *st])

    # 6. NLP: intent model + ShieldLM + RLCD.
    phase(prog, "nlp", [py, "-m", "src.nlp.train", *fast])

    # 6b. CORRECTED FOLLOW-UP (owner decision 2026-10-14). Registered runs above stay
    # untouched; these re-run the attack stages with the verified DERIVED formulas,
    # the pre-normalized-data guard and a converged attribution probe, under their
    # own ids (C_*), reported side by side. C_H5 (raw units, 5 seeds) is the
    # raw-unit cross-check for the normalized primary dataset.
    if not args.fast and not args.experiment:
        # v2: text columns no longer counted as NaN (2026-10-14 fix)
        phase(prog, "data_quality:v2", [py, "-m", "scripts.data_quality"])
        jobs = []
        for exp, _ds, seeds in _followups():
            for seed in seeds:
                cmd = [py, "run_pipeline.py", "--experiment", f"followup/{exp}"]
                jobs.append(
                    (
                        f"followup:{exp}:seed{seed}",
                        [*cmd, "--seed", str(seed), "--stage", "4,6,7", "--resume"],
                    )
                )
        run_parallel(prog, jobs, PARALLEL)
        # Raw-unit deployment: the primary CSV is min-max normalized, real sensor
        # flows are not. Keep the normalized detector as a backup, redeploy on raw
        # CIC-IDS2017 and self-test it end to end.
        if _has_data(RAW_DEPLOY):
            phase(prog, "deploy_backup", [py, "-m", "scripts.backup_deploy"])
            phase(
                prog, f"deploy:{RAW_DEPLOY}", [py, "deploy.py", "--dataset", RAW_DEPLOY]
            )
            phase(
                prog, f"soc_selftest:{RAW_DEPLOY}", [py, "-m", "scripts.soc_selftest"]
            )

    _validate(py, primary, t0)


def _validate(py: str, primary: str, t0: float) -> None:
    """7. VALIDATE: always re-run so the report reflects everything above."""
    # retried phases / stage-1 re-runs: newest attempt per stage counts, once
    subprocess.run([py, "-m", "scripts.dedupe_runs"], cwd=ROOT)
    subprocess.run([py, "run_pipeline.py", "--aggregate"], cwd=ROOT)
    rep = [py, "-m", "src.eval.final_report", "--primary", primary]
    if subprocess.run(rep, cwd=ROOT).returncode != 0:
        failures.append("final_report")

    _log(
        f"finished in {(time.time() - t0) / 60:.1f} min. Report: results/FINAL_REPORT.md"
    )
    if failures:
        _log("FAILED phases (re-run to retry only these): " + ", ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
