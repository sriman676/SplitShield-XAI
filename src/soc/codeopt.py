"""Nightly speed profiling + code speed-up PROPOSALS (owner 2026-10-15: local only,
the owner approves every code change).

  1. ``profile()``: runs fixed workloads (log-rule engine on the evolve benchmark,
     every model's speed probe) under cProfile; saves throughput and the slowest
     project functions to ``artifacts/controller/profile.json``; a workload more than
     ``SLOWDOWN`` slower than the previous profile raises a notification.
  2. ``propose()``: scans the slowest project functions for patterns from a built-in
     catalogue of safe speed-ups and writes each as a unified diff. Every diff is
     TRIED in a sandbox copy of the project (code copied, data folders linked):
     workload benchmark before/after in fresh interpreters + the full test suite.
     Only proposals that pass every test AND are measurably faster are offered.
  3. ``apply(id)`` (the owner's click on the dashboard): applies the diff to the real
     file, keeps a backup (``revert(id)``), and re-runs the tests; failure -> reverted.

Nothing is changed without the owner. Catalogue (each needs no judgement about
behaviour, so the tests are a meaningful check):
  * regex literal compiled on every call  -> module-level compiled pattern;
  * ``DataFrame.iterrows()`` in a hot loop -> advisory only (needs a human rewrite).

    python -m src.soc.codeopt --profile | --propose | --bench rules
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import cProfile
import difflib
import hashlib
import json
import os
import pstats
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from utils.logging_config import get_logger
from utils.paths import PROJECT_ROOT

_log = get_logger(__name__)
HOME = PROJECT_ROOT / "artifacts" / "controller"
PROFILE = HOME / "profile.json"
PROPOSALS = HOME / "code_proposals.json"
BACKUPS = HOME / "code_backups"
SLOWDOWN = 0.20  # notify when a workload is >20 % slower than last night
MIN_GAIN = 0.03  # a proposal must be >= 3 % faster to be offered
RE_FUNCS = (
    "search",
    "match",
    "fullmatch",
    "findall",
    "finditer",
    "sub",
    "subn",
    "split",
)
CODE_DIRS = ("src", "utils", "config", "scripts", "tests")
LINK_DIRS = ("artifacts", "third_party", "data", "results")


# --- workloads -------------------------------------------------------------------------


def _bench_lines(n: int = 4000) -> list[str]:
    from src.soc import evolve

    rows = evolve._bench()
    return [r["line"] for r in rows[:n]]


def workload_rules() -> float:
    """Log lines per second through the full parser + rules + Sigma (no line cache)."""
    from src.soc import parsers

    lines = _bench_lines()
    for ln in lines[:200]:  # warm-up: Sigma rules compile once per process
        parsers._parse_line(ln)
    t0 = time.perf_counter()
    for ln in lines:
        parsers._parse_line(ln)
    return len(lines) / max(time.perf_counter() - t0, 1e-9)


PROBE_EVENTS = 50_000  # size of the practice store the search / chat probes use
SEARCH_PROBES = ('"failed password"', "failed logins for root", "invalid user admin",
                 "show T1110 alerts", "failed logins last 2 hours")  # fmt: skip
CHAT_PROBES = ("failed logins for root", "is brute force happening?",
               "show critical incidents", "what is T1110")  # fmt: skip


def _probe_store() -> Any:
    """A temporary SOC store of PROBE_EVENTS parsed benchmark events (the real store
    is never touched); the engine points at it while the probe runs."""
    import contextlib

    from src.soc import engine, eventindex

    @contextlib.contextmanager
    def ctx() -> Any:
        tmp = Path(tempfile.mkdtemp(prefix="probe_"))
        old = engine.EVENTS
        try:
            from src.soc.parsers import parse_text

            evs = parse_text("\n".join(_bench_lines()), "bench.log")
            if not evs:
                raise FileNotFoundError("the evolve benchmark (artifacts/evolve/benchmark.jsonl) is missing or empty")
            now = time.time()
            with open(tmp / "events.jsonl", "w", encoding="utf-8") as fh:
                for i in range(PROBE_EVENTS):
                    e = dict(evs[i % len(evs)])
                    e["ts"] = now - 7 * 86400 * (1 - i / PROBE_EVENTS)
                    fh.write(json.dumps(e, default=str) + "\n")
            eventindex.close()
            engine.EVENTS = tmp / "events.jsonl"
            eventindex.count()  # build the index outside the timed part
            eventindex.incidents(900.0, scale=1.0)
            yield
        finally:
            eventindex.close()
            engine.EVENTS = old
            shutil.rmtree(tmp, ignore_errors=True)

    return ctx()


def workload_search() -> float:
    """Plain-English searches per second on the practice store (through the index)."""
    from src.soc import search

    with _probe_store():
        for q in SEARCH_PROBES:
            search.run_store(q)  # warm
        t0 = time.perf_counter()
        for _ in range(5):
            for q in SEARCH_PROBES:
                search.run_store(q)
        return 5 * len(SEARCH_PROBES) / max(time.perf_counter() - t0, 1e-9)


def workload_chat() -> float:
    """Whole assistant answers per second on the practice store."""
    from src.nlp.assistant import ShieldAssistant

    with _probe_store():
        bot = ShieldAssistant()
        for q in CHAT_PROBES:
            bot.answer(q)  # warm (narratives are written once, then reused)
        t0 = time.perf_counter()
        for _ in range(3):
            for q in CHAT_PROBES:
                bot.answer(q)
        return 3 * len(CHAT_PROBES) / max(time.perf_counter() - t0, 1e-9)


def workloads() -> dict[str, Any]:
    from src.soc import controller

    w: dict[str, Any] = {"rules": workload_rules, "search": workload_search,
                         "chat": workload_chat}  # fmt: skip
    for name, m in controller.registry().items():
        w[name] = m.speed
    return w


# --- profiling -------------------------------------------------------------------------


def _project_file(fn: str) -> str | None:
    try:
        rel = Path(fn).resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    return str(rel) if rel.parts and rel.parts[0] in ("src", "utils") else None


def profile(top: int = 15) -> dict[str, Any]:
    """Throughput of every workload + the slowest project functions (rules engine)."""
    out: dict[str, Any] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "throughput": {}}
    for name, fn in workloads().items():
        try:
            out["throughput"][name] = round(float(fn()), 1)
        except Exception as exc:  # noqa: BLE001 - one broken probe must not stop the rest
            out["throughput"][name] = None
            out.setdefault("errors", {})[name] = f"{type(exc).__name__}: {exc}"[:200]
    prof = cProfile.Profile()
    prof.enable()
    workload_rules()
    prof.disable()
    stats = pstats.Stats(prof)
    rows = []
    for (fn, line, func), (_cc, nc, tt, ct, _callers) in stats.stats.items():  # type: ignore[attr-defined]
        rel = _project_file(fn)
        if rel:
            rows.append({"file": rel, "line": line, "function": func, "calls": nc,
                         "self_s": round(tt, 4), "total_s": round(ct, 4)})  # fmt: skip
    out["hotspots"] = sorted(rows, key=lambda r: -r["self_s"])[:top]
    prev = _read(PROFILE)
    out["slower"] = {}
    for k, v in out["throughput"].items():
        old = (prev.get("throughput") or {}).get(k)
        if v and old and v < old * (1 - SLOWDOWN):
            out["slower"][k] = {"before": old, "now": v}
    if out["slower"]:
        from src.soc.controller import notify

        notify("slowdown: " + ", ".join(f"{k} {d['before']:.0f}->{d['now']:.0f}/s"
                                         for k, d in out["slower"].items()))  # fmt: skip
    if prev:
        out["previous"] = {"ts": prev.get("ts"), "throughput": prev.get("throughput")}
    HOME.mkdir(parents=True, exist_ok=True)
    PROFILE.write_text(json.dumps(out, indent=1), "utf-8")
    return out


# --- catalogue: find candidates --------------------------------------------------------


def _regex_calls(tree: ast.AST, func_names: set[str]) -> list[tuple[ast.Call, str]]:
    """``re.<f>("literal", ...)`` calls inside the named (hot) functions."""
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name in func_names
        ):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                        and isinstance(sub.func.value, ast.Name) and sub.func.value.id == "re"
                        and sub.func.attr in RE_FUNCS and sub.args
                        and isinstance(sub.args[0], ast.Constant)
                        and isinstance(sub.args[0].value, str)):  # fmt: skip
                    found.append((sub, node.name))
    return found


def _regex_patch(path: Path, func_names: set[str]) -> tuple[str, str] | None:
    """Rewrite literal regex calls in hot functions to module-level compiled patterns.
    Returns (new_source, description) or None. Flags passed as ``flags=`` or as the
    last positional argument of search/match/fullmatch/findall/finditer are moved into
    ``re.compile``; anything less clear-cut is left alone."""
    src = path.read_text("utf-8")
    tree = ast.parse(src)
    calls = _regex_calls(tree, func_names)
    if not calls:
        return None
    lines = src.splitlines(keepends=True)
    consts: dict[str, str] = {}
    edits: list[tuple[int, int, int, int, str]] = []
    for call, _fn in calls:
        assert isinstance(call.func, ast.Attribute)  # guaranteed by _regex_calls
        attr = call.func.attr
        pat = ast.get_source_segment(src, call.args[0])
        if pat is None or call.end_lineno != call.lineno:
            continue  # multi-line calls: skip (keep the patch trivially reviewable)
        flags = None
        rest = list(call.args[1:])
        kws = [k for k in call.keywords if k.arg != "flags"]
        fk = [k for k in call.keywords if k.arg == "flags"]
        if fk:
            flags = ast.get_source_segment(src, fk[0].value)
        elif (
            attr in ("search", "match", "fullmatch", "findall", "finditer")
            and len(rest) == 2
        ):
            flags = ast.get_source_segment(src, rest.pop())
        elif attr in ("sub", "subn", "split") and len(rest) > 2:
            continue  # count/maxsplit + flags positional: not worth the risk
        if flags and not all(t.strip().startswith("re.") for t in flags.split("|")):
            continue  # flags from a variable cannot move to module level
        compiled = f"re.compile({pat}{', ' + flags if flags else ''})"
        name = "_RX_" + hashlib.sha1(compiled.encode()).hexdigest()[:8].upper()
        consts[name] = compiled
        args = [ast.get_source_segment(src, a) or "" for a in rest]
        args += [f"{k.arg}={ast.get_source_segment(src, k.value)}" for k in kws]
        new = f"{name}.{attr}({', '.join(args)})"
        edits.append((call.lineno, call.col_offset, call.end_lineno or call.lineno,
                       call.end_col_offset or 0, new))  # fmt: skip
    if not edits:
        return None
    for ln, c0, _ln2, c1, new in sorted(edits, reverse=True):
        s = lines[ln - 1]
        b = s.encode()  # ast offsets are UTF-8 byte offsets
        lines[ln - 1] = (b[:c0] + new.encode() + b[c1:]).decode()
    # constants after the last top-level import
    last_import = max((n.end_lineno or 0 for n in tree.body
                       if isinstance(n, ast.Import | ast.ImportFrom)), default=0)  # fmt: skip
    block = "".join(f"{k} = {v}  # codeopt: compiled once\n" for k, v in consts.items())
    lines.insert(last_import, "\n" + block)
    new_src = "".join(lines)
    if "import re" not in new_src.split("\n_RX_")[0]:
        return None  # module does not import re at top level: leave it
    desc = f"compile {len(consts)} regex pattern(s) once instead of on every call"
    return new_src, desc


def _iterrows_advice(path: Path, func_names: set[str]) -> list[str]:
    tree = ast.parse(path.read_text("utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in func_names:
            if any(
                isinstance(s, ast.Attribute) and s.attr == "iterrows"
                for s in ast.walk(node)
            ):
                out.append(node.name)
    return out


# --- sandbox trial ---------------------------------------------------------------------


@contextlib.contextmanager
def _sandbox() -> Any:
    from src.soc.updater import _link_dir, _unlink_dir

    root = Path(tempfile.mkdtemp(prefix="co_"))
    links: list[Path] = []
    try:
        for d in CODE_DIRS:
            if (PROJECT_ROOT / d).is_dir():
                shutil.copytree(PROJECT_ROOT / d, root / d,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))  # fmt: skip
        for f in ("pyproject.toml", "dashboard.py", "deploy.py", "run_pipeline.py",
                  "run_matrix.py", "improve.py", "ai_engineer.py"):  # fmt: skip
            if (PROJECT_ROOT / f).is_file():
                shutil.copy2(PROJECT_ROOT / f, root / f)
        for d in LINK_DIRS:
            if (PROJECT_ROOT / d).is_dir() and _link_dir(root / d, PROJECT_ROOT / d):
                links.append(root / d)
        yield root
    finally:
        for link in links:  # never let rmtree walk into the real data
            _unlink_dir(link)
        shutil.rmtree(root, ignore_errors=True)


def _bench_in(root: Path, workload: str = "rules", repeats: int = 3) -> float:
    vals = []
    for _ in range(repeats):
        r = subprocess.run([sys.executable, "-m", "src.soc.codeopt", "--bench", workload],
                           cwd=root, capture_output=True, text=True, timeout=900,
                           env=_CHILD_ENV())  # fmt: skip
        try:
            vals.append(float(r.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            raise RuntimeError(f"benchmark failed: {r.stderr[-300:]}") from None
    vals.sort()
    return vals[len(vals) // 2]


def _tests_in(root: Path) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "-x"],
                       cwd=root, capture_output=True, text=True, timeout=3600,
                       env=_CHILD_ENV())  # fmt: skip
    tail = (r.stdout or r.stderr).strip().splitlines()[-1:] or [""]
    return r.returncode == 0, tail[0]


GUARD = (
    "search",
    "chat",
)  # a speed-up elsewhere must not slow these (owner 2026-10-16)
GUARD_LOSS = 0.05  # ... by more than 5 %


def trial(rel: str, new_src: str) -> dict[str, Any]:
    """Benchmark + full tests of one patched file in a sandbox copy. Besides the rules
    gain, search and chat speed are measured before and after: a patch that makes
    either >5 % slower is not offered."""
    with _sandbox() as root:
        before = _bench_in(root)
        guard_before = {w: _bench_in(root, w, repeats=1) for w in GUARD}
        (root / rel).write_text(new_src, "utf-8")
        after = _bench_in(root)
        guard_after = {w: _bench_in(root, w, repeats=1) for w in GUARD}
        ok, summary = _tests_in(root)
    guard_ok = all(guard_after[w] >= guard_before[w] * (1 - GUARD_LOSS) for w in GUARD)
    return {"before": round(before, 1), "after": round(after, 1),
            "gain": round(after / before - 1, 4), "tests_passed": ok, "tests": summary,
            "guard": {w: [round(guard_before[w], 1), round(guard_after[w], 1)] for w in GUARD},
            "guard_ok": guard_ok}  # fmt: skip


# --- proposals ---------------------------------------------------------------------------


def _CHILD_ENV() -> dict[str, str]:  # noqa: N802 - reads like the constant it replaces
    """Environment for benchmark / test subprocesses. SPLITSHIELD_IN_TRIAL stops any
    code inside them from starting trials of its own (2026-10-15 regression: a test
    reached propose(), whose trial re-ran the test suite, recursively)."""
    return dict(os.environ) | {
        "SPLITSHIELD_NO_NOTIFY": "1",
        "SPLITSHIELD_IN_TRIAL": "1",
    }


def propose(run_trials: bool = True) -> list[dict[str, Any]]:
    """New proposals from tonight's hotspots (kept with earlier undecided ones)."""
    if os.environ.get("SPLITSHIELD_IN_TRIAL"):
        run_trials = False  # inside a trial or a test: never start nested trials
    prof = _read(PROFILE) or profile()
    hot: dict[str, set[str]] = {}
    for h in prof.get("hotspots", []):
        hot.setdefault(h["file"], set()).add(h["function"])
    props = {p["id"]: p for p in _read_list(PROPOSALS)}
    for rel, funcs in hot.items():
        path = PROJECT_ROOT / rel
        if not path.is_file():
            continue
        old = path.read_text("utf-8")
        patch = _regex_patch(path, funcs)
        if patch:
            new_src, desc = patch
            pid = hashlib.sha1((rel + new_src).encode()).hexdigest()[:10]
            if pid not in props:
                diff = "".join(difflib.unified_diff(old.splitlines(keepends=True),
                                                    new_src.splitlines(keepends=True),
                                                    f"a/{rel}", f"b/{rel}"))  # fmt: skip
                p: dict[str, Any] = {"id": pid, "file": rel, "kind": "precompile-regex",
                                     "description": desc, "diff": diff,
                                     "base_sha": hashlib.sha1(old.encode()).hexdigest(),
                                     "status": "untested", "created": time.strftime("%Y-%m-%dT%H:%M:%S")}  # fmt: skip
                if run_trials:
                    try:
                        p["trial"] = trial(rel, new_src)
                        good = (
                            p["trial"]["tests_passed"]
                            and p["trial"]["gain"] >= MIN_GAIN
                            and p["trial"].get("guard_ok", True)
                        )
                        p["status"] = (
                            "ready for approval"
                            if good
                            else "not offered (no gain, tests failed or search/chat slower)"
                        )
                    except Exception as exc:  # noqa: BLE001
                        p["status"] = f"trial failed: {exc}"[:200]
                p["_new_src"] = new_src
                props[pid] = p
        for fn in _iterrows_advice(path, funcs):
            pid = hashlib.sha1(f"iterrows:{rel}:{fn}".encode()).hexdigest()[:10]
            props.setdefault(pid, {"id": pid, "file": rel, "kind": "advice",
                                   "description": f"{fn}() loops over DataFrame.iterrows() in a hot path: "
                                                  "a vectorised rewrite is usually 10-100x faster (needs a human)",
                                   "status": "advice", "created": time.strftime("%Y-%m-%dT%H:%M:%S")})  # fmt: skip
    out = list(props.values())
    _write_list(PROPOSALS, out)
    ready = [p for p in out if p["status"] == "ready for approval"]
    if ready:
        from src.soc.controller import notify

        notify(
            f"{len(ready)} code speed-up proposal(s) wait for your approval (dashboard)"
        )
    return out


def apply(pid: str) -> dict[str, Any]:
    """Owner-approved: apply to the real file (backup kept), re-test, revert on failure."""
    props = {p["id"]: p for p in _read_list(PROPOSALS)}
    p = props.get(pid)
    if p is None or "_new_src" not in p:
        return {"error": "unknown proposal or nothing to apply"}
    path = PROJECT_ROOT / p["file"]
    cur = path.read_text("utf-8")
    if hashlib.sha1(cur.encode()).hexdigest() != p["base_sha"]:
        p["status"] = "stale (file changed since the proposal)"
        _write_list(PROPOSALS, list(props.values()))
        return {"error": p["status"]}
    BACKUPS.mkdir(parents=True, exist_ok=True)
    backup = BACKUPS / f"{pid}_{path.name}"
    shutil.copy2(path, backup)
    path.write_text(p["_new_src"], "utf-8")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "-x"],
                       cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=3600,
                       env=_CHILD_ENV())  # fmt: skip
    if r.returncode != 0:
        shutil.copy2(backup, path)
        p["status"] = "reverted (tests failed after applying)"
    else:
        p["status"] = "applied"
        p["applied"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        p["backup"] = str(backup)
    _write_list(PROPOSALS, list(props.values()))
    from src.soc.controller import record

    record(
        {
            "model": "code",
            "why": f"owner approved {pid}",
            "decision": p["status"],
            "file": p["file"],
        }
    )
    return {"status": p["status"]}


def revert(pid: str) -> dict[str, Any]:
    props = {p["id"]: p for p in _read_list(PROPOSALS)}
    p = props.get(pid)
    if not p or p.get("status") != "applied" or not Path(p.get("backup", "")).is_file():
        return {"error": "nothing to revert"}
    shutil.copy2(p["backup"], PROJECT_ROOT / p["file"])
    p["status"] = "reverted by owner"
    _write_list(PROPOSALS, list(props.values()))
    return {"status": p["status"]}


def reject(pid: str) -> None:
    props = {p["id"]: p for p in _read_list(PROPOSALS)}
    if pid in props:
        props[pid]["status"] = "rejected by owner"
        _write_list(PROPOSALS, list(props.values()))


def proposals() -> list[dict[str, Any]]:
    return [
        {k: v for k, v in p.items() if k != "_new_src"} for p in _read_list(PROPOSALS)
    ]


def _read(path: Path) -> dict[str, Any]:
    try:
        out: dict[str, Any] = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return out


def _read_list(path: Path) -> list[dict[str, Any]]:
    try:
        out: list[dict[str, Any]] = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return []
    return out


def _write_list(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1), "utf-8")


def main() -> int:
    from utils.logging_config import configure_logging

    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--propose", action="store_true")
    ap.add_argument("--bench", default=None)
    a = ap.parse_args()
    if a.bench:
        print(workloads()[a.bench]())  # last line = items/s (read by _bench_in)
    elif a.propose:
        print(
            json.dumps(proposals() if not propose() else proposals(), indent=1)[:4000]
        )
    else:
        print(json.dumps(profile(), indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
