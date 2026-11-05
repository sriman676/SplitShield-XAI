"""One self-improvement controller around all 9 models (owner 2026-10-15).

The SOC has nine specialist models (one per kind of input). Each already has, or now
gets, a champion/challenger update that is judged on fixed held-out data. This module
wraps every one of them in the same guard:

  1. snapshot the model's files;
  2. measure its speed on fixed probe data (median of 3 runs);
  3. run the model's own update (new labelled data from its inbox folder, analyst
     reviews, or assistant ratings);
  4. if the update was promoted, measure speed again: slower by more than
     ``SPEED_TOL`` (timing noise) -> restore the snapshot ("rolled back: slower");
  5. append the decision to ``artifacts/controller/history.jsonl`` and notify.

So a change is kept only when held-out results are not worse (the model's own gate)
AND it is not slower (this guard). Nothing here edits code: code speed-ups are only
PROPOSED by ``src/soc/codeopt.py`` and wait for the owner.

Inbox (drop labelled files, the next cycle learns from them, then moves them to
``done/``):

  artifacts/inbox/flow/*.csv       CICFlowMeter flows with a Label column
  artifacts/inbox/netflow/*.csv    NetFlow v9 records with a Label column
  artifacts/inbox/zeek/*.log       labelled Zeek conn.log (IoT-23 style label field)
  artifacts/inbox/phishing/*.csv   url,label (1 = phishing)
  artifacts/inbox/malware/*.jsonl  EMBER feature records with label 0/1
  artifacts/inbox/ics/*.csv        plant sensor CSV (normal rows are learned)
  artifacts/inbox/insider/*.csv    user-day features + label

    python -m src.soc.controller            # one cycle (also run hourly by the autopilot)
    python -m src.soc.controller --health   # health of every model and component
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from utils.logging_config import get_logger
from utils.paths import PROJECT_ROOT

_log = get_logger(__name__)
ART = PROJECT_ROOT / "artifacts"
HOME = ART / "controller"
HISTORY = HOME / "history.jsonl"
STATE = HOME / "state.json"
INBOX = ART / "inbox"
SPEED_TOL = 0.10  # a promoted model may be at most 10 % slower (timing noise)


# --- notifications -----------------------------------------------------------------


def notify(text: str) -> None:
    """Desktop pop-up for the signed-in user on Windows, macOS or Linux (best effort,
    never fatal, not in tests) - src/soc/osops.py."""
    from src.soc import osops

    osops.notify(text)


# --- model registry ----------------------------------------------------------------


@dataclass
class Model:
    name: str
    title: str
    files: list[Path]  # what a snapshot copies (files or folders)
    speed: Callable[[], float]  # items per second on fixed probe data
    inbox: str | None = None  # file pattern in artifacts/inbox/<name>/
    learn: Callable[[Path], dict[str, Any]] | None = None
    unit: str = "items/s"
    extra: dict[str, Any] = field(default_factory=dict)


def _timed(fn: Callable[[], Any], n: int, repeats: int = 3) -> float:
    fn()  # warm-up (lazy loads, caches)
    rates = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        rates.append(n / max(time.perf_counter() - t0, 1e-9))
    return float(statistics.median(rates))


def _speed_flow() -> float:
    import pandas as pd

    from src.soc import engine

    det = engine.load_detector()
    if det is None:
        raise FileNotFoundError("flow detector not deployed")
    p = np.load(ART / "deploy" / "probe.npz", allow_pickle=False)
    df = pd.DataFrame(p["X"], columns=det.feature_names)
    return _timed(lambda: engine.flow_logits(det, df), len(df))


def _speed_netflow() -> float:
    import pandas as pd

    from src.soc import netflow_model as nm

    det = nm.load_netflow()
    if det is None:
        raise FileNotFoundError("NetFlow detector not deployed")
    X = np.load(nm.NETFLOW_DIR / "holdout.npz")["X_val"][:5000]
    df = pd.DataFrame(X, columns=det.feature_names)
    return _timed(lambda: det.score(df), len(df))


def _speed_zeek() -> float:
    from src.soc import zeekml

    b = zeekml.load()
    if b is None:
        raise FileNotFoundError("Zeek model not trained")
    X = np.load(Path(zeekml.MODEL).parent / "holdout.npz")["X_val"][:5000].astype(
        np.float32
    )
    return _timed(lambda: b["model"].predict_proba(X), len(X))


def _speed_phishing() -> float:
    from src.mlareas import phishing

    b = phishing.load(refresh=True)
    sample = list(b["val_hosts"][:2000])
    return _timed(lambda: phishing.check_urls(sample, b), len(sample))


def _speed_malware() -> float:
    from src.mlareas import malware

    booster, _ = malware.load_champion()
    X = malware._load_npz(malware.MODEL_DIR / "holdout.npz")[0][:20000]
    return _timed(lambda: booster.predict(X), len(X))


def _speed_ics() -> float:
    from src.mlareas import ics

    b = ics.load(refresh=True)
    det = ics._detector(b)
    arr = np.asarray(b["val_clean"])[:20000]
    return _timed(lambda: ics.full_scores(det, arr, [len(arr)]), len(arr))


def _speed_insider() -> float:
    from src.mlareas import insider

    b = insider.load_bundle()
    _, va = insider._load_store_data(None)
    X = va[b["features"]].to_numpy(np.float32)[:20000]
    return _timed(lambda: insider.model_scores(b["family"], b["model"], X), len(X))


def _speed_intent() -> float:
    import joblib

    from src.nlp import feedback, intent

    clf = joblib.load(feedback.MODEL)
    _, _, Xte, _ = intent.dataset(0)
    qs = Xte[:500]
    return _timed(lambda: clf.predict(qs), len(qs))


def _speed_shieldlm() -> float:
    import torch

    from src.nlp import lm

    path = ART / "nlp" / "shieldlm_deployed.pt"
    model = lm.load(path)
    ids = torch.randint(0, 200, (8, 64))

    def run() -> None:
        with torch.no_grad():
            model(ids)

    return _timed(run, 8 * 64)  # tokens per second


def _learn_csv(fn: Callable[[Any], dict[str, Any]]) -> Callable[[Path], dict[str, Any]]:
    import pandas as pd

    return lambda p: fn(pd.read_csv(p, low_memory=False))


def _learn_flow(p: Path) -> dict[str, Any]:
    import pandas as pd

    from src.soc import engine

    det = engine.load_detector()
    df = pd.read_csv(p, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return engine.add_dataset(det, df, "Label", p.stem)  # dips: see promoted()


def _learn_netflow(p: Path) -> dict[str, Any]:
    import pandas as pd

    from src.soc.model_updates import netflow_improve

    return netflow_improve(pd.read_csv(p, low_memory=False))


def _learn_zeek(p: Path) -> dict[str, Any]:
    from src.soc.model_updates import zeek_improve, zeek_rows

    return zeek_improve(zeek_rows(p))


def _learn_malware(p: Path) -> dict[str, Any]:
    from src.mlareas import malware

    recs = [json.loads(ln) for ln in p.read_text("utf-8").splitlines() if ln.strip()]
    return malware.improve(recs)


def _area(name: str) -> Callable[[Any], dict[str, Any]]:
    import importlib

    fn: Callable[[Any], dict[str, Any]] = importlib.import_module(
        f"src.mlareas.{name}"
    ).improve
    return fn


REGISTRY = PROJECT_ROOT / "config" / "models.yaml"


def _learn_phishing(p: Path) -> dict[str, Any]:
    return _learn_csv(_area("phishing"))(p)


def _learn_ics(p: Path) -> dict[str, Any]:
    return _learn_csv(_area("ics"))(p)


def _learn_insider(p: Path) -> dict[str, Any]:
    return _learn_csv(_area("insider"))(p)


def _resolve(ref: str) -> Callable[..., Any]:
    """ "package.module:function" -> the function (registry entries)."""
    import importlib

    mod, _, fn = ref.partition(":")
    if not mod or not fn:
        raise ValueError(f"expected 'module:function', got {ref!r}")
    out: Callable[..., Any] = getattr(importlib.import_module(mod), fn)
    return out


def _lazy(ref: str) -> Callable[..., Any]:
    """Resolve on first call, so listing the registry never imports every model."""
    return lambda *a, **kw: _resolve(ref)(*a, **kw)


def registry(path: Path | None = None) -> dict[str, Model]:
    """Every model declared in config/models.yaml (see that file for the fields)."""
    import yaml

    doc = yaml.safe_load((path or REGISTRY).read_text("utf-8")) or {}
    out: dict[str, Model] = {}
    for name, e in doc.items():
        out[name] = Model(
            name=name,
            title=e.get("title", name),
            files=[PROJECT_ROOT / f for f in e.get("files", [])],
            speed=_lazy(e["speed"]),
            inbox=e.get("inbox"),
            learn=_lazy(e["learn"]) if e.get("learn") else None,
            unit=e.get("unit", "items/s"),
            extra={k: e[k] for k in ("family", "entities", "inputs") if k in e},
        )
    return out


def family_of(name: str) -> str:
    """Risk-board family of a registered model ("other" when not declared)."""
    m = registry().get(name)
    return str(m.extra.get("family", "other")) if m else "other"


# --- the guard ---------------------------------------------------------------------


def saved(res: dict[str, Any]) -> bool:
    """Did the model's own update write a new model? (keys differ per model)"""
    if res.get("kept") in (True, "challenger"):
        return True
    if "promoted" in res:
        return bool(res["promoted"])
    status = str(res.get("status", res.get("decision", ""))).lower()
    return status.startswith("promot") or "challenger promoted" in status


def promoted(res: dict[str, Any]) -> bool:
    """Saved AND not worse on held-out data (owner rule). The flow detector's own
    guard tolerates a 0.02 macro-F1 dip; here any dip counts as worse."""
    if not saved(res):
        return False
    before, after = res.get("macro_f1_before"), res.get("macro_f1_after")
    return before is None or after is None or after >= before


def _snapshot(model: Model) -> Path:
    snap = HOME / "snapshots" / model.name
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True)
    for i, f in enumerate(model.files):
        if f.is_dir():
            shutil.copytree(f, snap / str(i))
        elif f.is_file():
            (snap / str(i)).mkdir()
            shutil.copy2(f, snap / str(i) / f.name)
    return snap


def _restore(model: Model, snap: Path) -> None:
    for i, f in enumerate(model.files):
        src = snap / str(i)
        if not src.exists():
            continue
        if f.is_dir() or not f.suffix:
            shutil.rmtree(f, ignore_errors=True)
            shutil.copytree(src, f)
        else:
            shutil.copy2(src / f.name, f)
    _clear_caches()


def _clear_caches() -> None:
    """Loaded models are cached per process; drop them so the next use reloads."""
    try:
        from src.soc import netflow_model, zeekml

        netflow_model._CACHE.clear()
        zeekml._LOADED.clear()
    except Exception:  # noqa: BLE001 - best effort
        pass


def record(entry: dict[str, Any]) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S")} | entry
    with HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def guarded(
    name: str, update: Callable[[], dict[str, Any]], why: str
) -> dict[str, Any]:
    """Run one model update under the snapshot + speed guard (see module doc)."""
    from src.soc import evolution

    if not evolution.allowed(name):  # the user froze it (Settings -> Evolution)
        frozen: dict[str, Any] = {"model": name, "why": why,
                                  "decision": "frozen: not updated (evolution off)"}  # fmt: skip
        record(frozen)
        return frozen
    model = registry()[name]
    snap = _snapshot(model)
    try:
        before = model.speed()
    except Exception as exc:  # noqa: BLE001 - no probe -> no speed gate, still gated on quality
        before = None
        _log.warning("%s: speed probe failed (%s)", name, exc)
    try:
        res = update()
    except Exception as exc:  # noqa: BLE001 - a failed update leaves the champion
        _restore(model, snap)
        err = f"{type(exc).__name__}: {exc}"[:300]
        failed: dict[str, Any] = {
            "model": name,
            "why": why,
            "decision": "failed",
            "error": err,
        }
        record(failed)
        notify(f"{model.title}: update FAILED ({err[:120]}); old model kept")
        return failed
    out: dict[str, Any] = {"model": name, "why": why, "result": res}
    waiting = isinstance(res, dict) and bool(res.get("error")) and not saved(res)
    if saved(res) and not promoted(res):
        _restore(model, snap)
        out["decision"] = "rolled back: held-out score dipped"
    elif waiting:  # nothing was trained (e.g. not enough incidents or feedback yet)
        out["decision"] = f"not trained: {str(res['error'])[:160]}"
    elif not promoted(res):
        out["decision"] = "kept champion (challenger not better on held-out data)"
    else:
        _clear_caches()
        after = model.speed() if before is not None else None
        out["speed_before"], out["speed_after"] = before, after
        if (
            before is not None
            and after is not None
            and after < before * (1 - SPEED_TOL)
        ):
            _restore(model, snap)
            out["decision"] = (f"rolled back: {after:,.0f} vs {before:,.0f} {model.unit} "
                               f"(more than {SPEED_TOL:.0%} slower)")  # fmt: skip
        else:
            out["decision"] = "promoted"
    record(out)
    if not out["decision"].startswith(("kept champion", "not trained")):
        notify(f"{model.title}: {out['decision']}")
    return out


# --- one cycle ---------------------------------------------------------------------


def cycle() -> list[dict[str, Any]]:
    """Learn from every file waiting in an inbox, then from new assistant ratings."""
    from src.soc import evolution

    outs = []
    for model in registry().values():
        if not (model.inbox and model.learn):
            continue
        if not evolution.allowed(model.name):
            continue  # frozen: its inbox files wait, learned once it evolves again
        box = INBOX / model.name
        box.mkdir(parents=True, exist_ok=True)
        for f in sorted(box.glob(model.inbox)):
            out = guarded(model.name, _bind(model.learn, f), f"inbox file {f.name}")
            done = box / "done"
            done.mkdir(exist_ok=True)
            shutil.move(str(f), done / f"{time.strftime('%Y%m%d-%H%M%S')}_{f.name}")
            outs.append(out)
    outs += _intent_step()
    st = _state()
    st["last_cycle"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save_state(st)
    return outs


def _bind(
    learn: Callable[[Path], dict[str, Any]], f: Path
) -> Callable[[], dict[str, Any]]:
    return lambda: learn(f)


def _intent_step() -> list[dict[str, Any]]:
    """Ratings that arrived since the last promotion (normally learned instantly by the
    dashboard; this catches any that a closed dashboard did not finish)."""
    from src.nlp import feedback
    from src.soc import evolution

    if not evolution.allowed("intent"):
        return []  # frozen: the ratings stay counted as new for later
    n = len(feedback.labelled()[0])
    st = _state()
    if n == 0 or n == st.get("intent_rated_seen"):
        return []
    out = guarded("intent", feedback.quick_update, f"{n} rated questions")
    st = _state()
    st["intent_rated_seen"] = n
    _save_state(st)
    return [out]


def _state() -> dict[str, Any]:
    try:
        st: dict[str, Any] = json.loads(STATE.read_text("utf-8"))
    except (OSError, ValueError):
        st = {}
    return st


def _save_state(st: dict[str, Any]) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1, default=str), "utf-8")


def history(last: int = 50) -> list[dict[str, Any]]:
    if not HISTORY.is_file():
        return []
    lines = HISTORY.read_text("utf-8").splitlines()[-last:]
    return [json.loads(ln) for ln in lines if ln.strip()]


# --- health ------------------------------------------------------------------------


def health() -> dict[str, Any]:
    """Every model (loads? speed on probe data) + every automatic component."""
    models: dict[str, Any] = {}
    for m in registry().values():
        try:
            models[m.name] = {
                "title": m.title,
                "ok": True,
                "speed": round(m.speed()),
                "unit": m.unit,
            }
        except Exception as exc:  # noqa: BLE001 - report, never raise
            models[m.name] = {"title": m.title, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}  # fmt: skip
    return {"checked": time.strftime("%Y-%m-%dT%H:%M:%S"), "models": models,
            "components": components()}  # fmt: skip


def _age_h(ts: float | str | None) -> float | None:
    if not ts:
        return None
    if isinstance(ts, str):
        try:
            ts = time.mktime(time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            return None
    return (time.time() - float(ts)) / 3600


def components() -> dict[str, Any]:
    """Is each automatic part alive? (``ok`` False -> shown red + notified)"""
    out: dict[str, Any] = {}
    try:
        ap = json.loads((ART / "soc" / "autopilot_state.json").read_text("utf-8"))
        age = _age_h(ap.get("last_run"))
        out["autopilot"] = {"ok": age is not None and age < 2.5,
                            "detail": f"last run {age:.1f} h ago" if age is not None else "never ran"}  # fmt: skip
    except (OSError, ValueError):
        out["autopilot"] = {"ok": False, "detail": "never ran"}
    age = _age_h(_state().get("last_cycle"))
    out["controller"] = {"ok": age is not None and age < 2.5,
                         "detail": f"last cycle {age:.1f} h ago" if age is not None else "never ran"}  # fmt: skip
    try:
        from src.soc import updater

        u = updater.load_state()
        if not u.get("enabled"):
            out["update_agent"] = {"ok": True, "detail": "switched off (by choice)"}
        else:
            age = _age_h(u.get("last_run"))
            bad = [h for h in u.get("history", [])[-1:] if h.get("status") == "error"]
            out["update_agent"] = {"ok": age is not None and age < 30 and not bad,
                                   "detail": f"last run {age:.1f} h ago" if age is not None else "never ran"}  # fmt: skip
    except Exception as exc:  # noqa: BLE001
        out["update_agent"] = {"ok": False, "detail": str(exc)[:120]}
    try:
        ev = json.loads((ART / "evolve" / "state.json").read_text("utf-8"))
        out["evolve_loop"] = {
            "ok": True,
            "detail": f"iteration {ev.get('iteration', 0)}",
        }
    except (OSError, ValueError):
        out["evolve_loop"] = {"ok": False, "detail": "no state yet"}
    from src.soc import osops

    sensor = osops.SENSOR_SERVICE[osops.PLATFORM]
    out["sysmon"] = {"ok": _service_running(sensor), "detail": f"{sensor} service"}
    prof = ART / "controller" / "profile.json"
    age = _age_h(prof.stat().st_mtime) if prof.is_file() else None
    out["speed_profiling"] = {"ok": age is not None and age < 30,
                              "detail": f"last profile {age:.1f} h ago" if age is not None else "never ran"}  # fmt: skip
    return out


def _service_running(name: str) -> bool:
    from src.soc import osops

    return osops.service_running(name)


def watch() -> dict[str, Any]:
    """Health check that notifies about anything newly broken (hourly, autopilot)."""
    comps = components()
    checked = time.strftime("%Y-%m-%dT%H:%M:%S")
    st = _state()
    was_bad = set(st.get("bad", []))
    bad = {k for k, v in comps.items() if not v["ok"]}
    for k in sorted(bad - was_bad):
        notify(f"{k.replace('_', ' ')} needs attention: {comps[k]['detail']}")
    st["bad"] = sorted(bad)
    st["last_watch"] = checked
    _save_state(st)
    return {"checked": checked, "components": comps}


def main() -> int:
    from utils.logging_config import configure_logging

    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--health", action="store_true")
    a = ap.parse_args()
    print(json.dumps(health() if a.health else cycle(), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
