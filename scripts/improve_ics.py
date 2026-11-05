"""Improve the ICS anomaly detector (owner 2026-10-16): longer look-back / smoothing.

HAI attacks are slow manipulations; the deployed autoencoder sees 30 s of history
and smooths its score over 10 s. Here the window settings are varied with
everything else unchanged (same HAI 22.04 splits, same injected-fault validation,
same threshold rule). Step 1 screens each setting on seed 0 and picks the best
VALIDATION (injected-fault) F1; step 2 runs the pick on all 3 seeds. It replaces
the deployed detector only if mean TEST precision AND recall are both not worse
than the deployed one's (results/extra/ics.json). HAI 21.03 (blind) is never read.

    python -m scripts.improve_ics [--screen-only] [--no-save]
    python -m scripts.improve_ics --per-sensor [--no-save]
    python -m scripts.improve_ics --attack-split [--no-save]   (owner 2026-10-17)
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from src.mlareas import ics
from utils.paths import RESULTS_DIR

OUT = RESULTS_DIR / "extra" / "improve_ics.json"
# (look-back s, frame step s, smoothing s); the first is the deployed setting
CONFIGS = [(30, 3, 10), (60, 6, 30), (120, 12, 60), (120, 12, 120), (300, 30, 120)]


def _use(cfg: tuple[int, int, int]) -> None:
    ics.LOOKBACK, ics.STEP, ics.SMOOTH = cfg


def _run(
    cfg: tuple[int, int, int], seeds: tuple[int, ...], frames: Any, save: bool
) -> dict[str, Any]:
    _use(cfg)
    res = ics.train(seeds=seeds, families=("autoencoder",), save=save, frames=frames)
    res.pop("_plot", None)
    fam = res["families"]["autoencoder"]
    return {"config": {"lookback": cfg[0], "step": cfg[1], "smooth": cfg[2]},
            "val_f1": fam["val"]["f1"]["mean"],
            "test": {k: fam["test"][k]["mean"] for k in ("precision", "recall", "f1")},
            "episodes": fam["test"]["episodes_detected"]["mean"],
            "detail": fam}  # fmt: skip


OUT_PER_SENSOR = RESULTS_DIR / "extra" / "improve_ics_per_sensor.json"
# per-sensor normalisation: (quantile, where it is measured); None = deployed (mean)
PER_SENSOR = [None, (0.99, "train"), (0.999, "train"), (0.9999, "train"),
              (0.99, "val"), (0.999, "val"), (0.9999, "val")]  # fmt: skip


def _per_sensor_eval(det: Any, sp: Any, variant: Any, seed: int) -> tuple[dict[str, Any], Any]:
    import numpy as np

    if variant is not None:
        q, where = variant
        if where == "train":
            arr, idx = sp.train, ics.valid_index(sp.train_lengths)
        else:  # clean validation (normal operation, later in time than training)
            arr, idx = sp.val, ics.valid_index([len(sp.val)])
        det = ics.rescale_per_sensor(det, arr, np.asarray(idx), q, seed)
    res, thr, pct, _ = ics._evaluate(det, sp)
    return res | {"threshold": thr, "percentile": pct}, det


def per_sensor(save: bool) -> dict[str, Any]:
    """ICS step 2 (owner 2026-10-16): per-sensor normalisation + threshold, chosen on
    injected-fault validation F1 (seed 0), confirmed on 3 seeds; replace only if
    mean test precision AND recall are both not worse than the deployed detector."""
    t0 = time.time()
    frames = ics.load_hai()
    sp = ics.make_splits(*frames)
    champ = json.loads((RESULTS_DIR / "extra" / "ics.json").read_text("utf-8"))
    ch = champ["families"]["autoencoder"]["test"]
    champion = {k: ch[k]["mean"] for k in ("precision", "recall", "f1")}
    dets = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for seed in (0, 1, 2):
        torch_threads()
        dets[seed] = ics.fit_detector("autoencoder", sp.train, ics.valid_index(sp.train_lengths), seed,
                                      epochs=5, per_epoch=200_000)  # fmt: skip
    for v in PER_SENSOR:
        name = "deployed (mean)" if v is None else f"q{v[0]} on {v[1]}"
        rows[name] = []
        for seed in (0, 1, 2):
            r, _ = _per_sensor_eval(dets[seed], sp, v, seed)
            rows[name].append(r)
        r0 = rows[name][0]
        print(name, "val F1", round(r0["val"]["f1"], 4), "test P/R",
              round(r0["test"]["precision"], 3), round(r0["test"]["recall"], 3), flush=True)  # fmt: skip
    pick = max(PER_SENSOR, key=lambda v: rows["deployed (mean)" if v is None else f"q{v[0]} on {v[1]}"][0]["val"]["f1"])
    pname = "deployed (mean)" if pick is None else f"q{pick[0]} on {pick[1]}"
    mean = lambda name, split, m: sum(r[split][m] for r in rows[name]) / len(rows[name])  # noqa: E731
    summary = {name: {split: {m: round(mean(name, split, m), 4) for m in ("precision", "recall", "f1")}
                      for split in ("train", "val", "test")} for name in rows}  # fmt: skip
    t = summary[pname]["test"]
    better = (pick is not None and t["precision"] >= champion["precision"]
              and t["recall"] >= champion["recall"])  # fmt: skip
    doc: dict[str, Any] = {
        "what": "ICS autoencoder: per-sensor normalisation (quantile of each sensor's own "
                "normal error) instead of the mean, chosen on injected-fault validation F1",
        "rule": "pick on validation (seed 0); replace only if mean test precision AND "
                "recall (3 seeds) are not worse than the deployed detector",
        "champion_test_mean": champion, "variants_mean_3_seeds": summary,
        "picked_on_validation": pname, "blind_used": False,
    }  # fmt: skip
    if pick is None:
        doc["decision"] = "kept champion: validation picks the deployed normalisation"
    elif not better:
        doc["decision"] = (f"kept champion: test precision {t['precision']:.3f} vs "
                           f"{champion['precision']:.3f}, recall {t['recall']:.3f} vs "
                           f"{champion['recall']:.3f} - one of them would drop")  # fmt: skip
    else:
        doc["decision"] = "promoted"
        if save:
            r0, det0 = _per_sensor_eval(dets[0], sp, pick, 0)
            bundle = ics._bundle(det0, sp, r0["threshold"], r0["percentile"], 0)
            old = ics.load(refresh=True)
            bundle["version"] = int(old.get("version", 1)) + 1
            bundle["per_sensor"] = {"quantile": pick[0], "measured_on": pick[1]}
            ics.common.save_bundle(ics.AREA, bundle)
            ics.load(refresh=True)
            doc["saved_version"] = bundle["version"]
    doc["seconds"] = round(time.time() - t0)
    OUT_PER_SENSOR.write_text(json.dumps(doc, indent=1, default=str), "utf-8")
    print(json.dumps({"decision": doc["decision"], "picked": pname, "test": t}, indent=1))
    return doc


OUT_ATTACK_SPLIT = RESULTS_DIR / "extra" / "improve_ics_attack_split.json"
VAL_FILES = 2  # hai-22.04 test1+test2 (2021-07-10..14) = validation; test3+test4 = test


def _attack_eval(det: Any, sp: Any, n_val_rows: int, val_len: list[int], test_len: list[int],
                 thr: float | None = None) -> dict[str, Any]:  # fmt: skip
    """Threshold on the EARLY attack files (+ the clean-validation alarm budget), test on
    the LATER ones. ``thr`` given -> use it as is (the deployed champion)."""
    import numpy as np

    scores = ics.full_scores(det, sp.test, sp.test_lengths)
    sv, st = scores[:n_val_rows], scores[n_val_rows:]
    yv, yt = sp.test_labels[:n_val_rows], sp.test_labels[n_val_rows:]
    pct = None
    if thr is None:
        clean = ics.full_scores(det, sp.val, [len(sp.val)])
        days = max(len(clean) / 86_400, 1e-9)
        best = (-1.0, float(np.max(clean)) + 1e-9, 100.0)
        for p in ics.PERCENTILES:
            t = float(np.percentile(clean, p))
            if len(ics.alarm_events((clean > t).astype(int))) / days > ics.ALARM_BUDGET:
                continue  # owner rule: <= 5 false alarms/day on normal operation
            if ics.event_metrics((sv > t).astype(int), yv)["false_alarms_per_day"] > ics.ALARM_BUDGET:
                continue  # ...and on the normal stretches of the validation attack files
            f1 = ics.point_metrics((sv > t).astype(int), yv)["f1"]
            if f1 > best[0]:
                best = (f1, t, p)
        thr, pct = best[1], best[2]
    a_v, a_t = (sv > thr).astype(int), (st > thr).astype(int)
    return {"threshold": float(thr), "percentile": pct,
            "val": ics.point_metrics(a_v, yv) | ics.event_metrics(a_v, yv),
            "test": ics.point_metrics(a_t, yt) | ics.event_metrics(a_t, yt)}  # fmt: skip


def attack_split(save: bool) -> dict[str, Any]:
    """ICS step 3 (owner 2026-10-17): TIME-SPLIT ATTACK VALIDATION. The labelled HAI
    22.04 attack files are split in time: test1+test2 = validation (threshold and
    variant choice), test3+test4 = test (reported, decides). The deployed champion is
    re-scored on the same new test part with its own threshold. A variant replaces it
    only if, on the new test part, precision AND recall are not lower - for the seed-0
    model that would be deployed AND for the 3-seed mean. HAI 21.03 (blind) never read."""
    import numpy as np

    t0 = time.time()
    frames = ics.load_hai()
    sp = ics.make_splits(*frames)
    val_len, test_len = sp.test_lengths[:VAL_FILES], sp.test_lengths[VAL_FILES:]
    n_val = int(sum(val_len))
    champ_b = ics.load(refresh=True)
    if list(champ_b["prep"].columns) != list(sp.prep.columns):
        raise RuntimeError("rebuilt HAI split does not match the deployed detector's sensors")
    champ = _attack_eval(ics._detector(champ_b), sp, n_val, val_len, test_len, float(champ_b["threshold"]))
    # the deployed model with a threshold chosen on the new attack validation
    champ_rethr = _attack_eval(ics._detector(champ_b), sp, n_val, val_len, test_len)
    rows: dict[str, list[dict[str, Any]]] = {}
    dets: dict[int, Any] = {}
    for seed in (0, 1, 2):
        torch_threads()
        dets[seed] = ics.fit_detector("autoencoder", sp.train, ics.valid_index(sp.train_lengths), seed,
                                      epochs=5, per_epoch=200_000)  # fmt: skip
    for v in PER_SENSOR:
        name = "mean (deployed normalisation)" if v is None else f"q{v[0]} on {v[1]}"
        rows[name] = []
        for seed in (0, 1, 2):
            det = dets[seed]
            if v is not None:
                arr, idx = ((sp.train, ics.valid_index(sp.train_lengths)) if v[1] == "train"
                            else (sp.val, ics.valid_index([len(sp.val)])))  # fmt: skip
                det = ics.rescale_per_sensor(det, arr, np.asarray(idx), v[0], seed)
            rows[name].append(_attack_eval(det, sp, n_val, val_len, test_len))
        r0 = rows[name][0]
        print(name, "val F1", round(r0["val"]["f1"], 4), "test P/R", round(r0["test"]["precision"], 3),
              round(r0["test"]["recall"], 3), flush=True)  # fmt: skip
    mean = lambda name, split, m: float(np.mean([r[split][m] for r in rows[name]]))  # noqa: E731
    summary = {name: {"seed0": {s: {m: round(rows[name][0][s][m], 4) for m in ("precision", "recall", "f1",
                                                                                   "episodes_detected", "false_alarms_per_day")}
                                for s in ("val", "test")},
                      "mean_3_seeds": {s: {m: round(mean(name, s, m), 4) for m in ("precision", "recall", "f1")}
                                       for s in ("val", "test")}} for name in rows}  # fmt: skip
    pick = max(rows, key=lambda n: float(np.mean([r["val"]["f1"] for r in rows[n]])))
    c = champ["test"]
    s0, sm = summary[pick]["seed0"]["test"], summary[pick]["mean_3_seeds"]["test"]
    passes = all(x["precision"] >= c["precision"] and x["recall"] >= c["recall"] for x in (s0, sm))
    doc: dict[str, Any] = {
        "what": "ICS: time-split attack validation (HAI 22.04 test1+2 = validation, test3+4 = test); "
                "per-sensor normalisation variants",
        "split": {"validation_files": ["test1", "test2"], "test_files": ["test3", "test4"],
                  "validation_attack_episodes": len(ics.episodes(sp.test_labels[:n_val])),
                  "test_attack_episodes": len(ics.episodes(sp.test_labels[n_val:]))},
        "rule": "variant and threshold chosen on validation (mean F1 over 3 seeds; <= 5 false alarms/day); "
                "replace only if new-test precision AND recall are not lower than the deployed champion's, "
                "for the seed-0 model and the 3-seed mean",
        "champion_deployed_threshold": {k: champ[k] for k in ("threshold", "test", "val")},
        "champion_rethresholded_on_new_validation": champ_rethr,
        "variants": summary, "picked_on_validation": pick, "blind_used": False,
    }  # fmt: skip
    if not passes:
        doc["decision"] = (f"kept champion: {pick} (picked on validation) test seed-0 P/R "
                           f"{s0['precision']:.3f}/{s0['recall']:.3f}, 3-seed mean {sm['precision']:.3f}/"
                           f"{sm['recall']:.3f} vs champion {c['precision']:.3f}/{c['recall']:.3f}")  # fmt: skip
    else:
        doc["decision"] = "promoted"
        if save:
            v = next(x for x in PER_SENSOR if ("mean (deployed normalisation)" if x is None
                                               else f"q{x[0]} on {x[1]}") == pick)  # fmt: skip
            det = dets[0]
            if v is not None:
                arr, idx = ((sp.train, ics.valid_index(sp.train_lengths)) if v[1] == "train"
                            else (sp.val, ics.valid_index([len(sp.val)])))  # fmt: skip
                det = ics.rescale_per_sensor(det, arr, np.asarray(idx), v[0], 0)
            r0 = rows[pick][0]
            bundle = ics._bundle(det, sp, r0["threshold"], r0["percentile"] or 100.0, 0)
            bundle["version"] = int(champ_b.get("version", 1)) + 1
            bundle["per_sensor"] = None if v is None else {"quantile": v[0], "measured_on": v[1]}
            bundle["threshold_basis"] = "time-split attack validation (HAI 22.04 test1+2), 2026-10-17"
            ics.common.save_bundle(ics.AREA, bundle)
            ics.load(refresh=True)
            doc["saved_version"] = bundle["version"]
    doc["seconds"] = round(time.time() - t0)
    OUT_ATTACK_SPLIT.write_text(json.dumps(doc, indent=1, default=str), "utf-8")
    print(json.dumps({"decision": doc["decision"], "champion_test": c, "picked": pick}, indent=1, default=str))
    return doc


def torch_threads() -> None:
    import torch

    torch.set_num_threads(4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen-only", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--per-sensor", action="store_true", help="step 2: per-sensor normalisation")
    ap.add_argument("--attack-split", action="store_true",
                    help="step 3: time-split attack validation (test1+2 validation, test3+4 test)")
    a = ap.parse_args()
    if a.attack_split:
        attack_split(save=not a.no_save)
        return
    if a.per_sensor:
        per_sensor(save=not a.no_save)
        return
    t0 = time.time()
    frames = ics.load_hai()
    champ = json.loads((RESULTS_DIR / "extra" / "ics.json").read_text("utf-8"))
    ch = champ["families"]["autoencoder"]["test"]
    champion = {k: ch[k]["mean"] for k in ("precision", "recall", "f1")}
    screen = []
    for cfg in CONFIGS:
        r = _run(cfg, (0,), frames, save=False)
        screen.append(r)
        print(json.dumps({k: r[k] for k in ("config", "val_f1", "test")}), flush=True)
    pick = max(screen, key=lambda r: r["val_f1"])
    doc: dict[str, Any] = {
        "what": "ICS autoencoder: look-back / smoothing chosen on injected-fault validation F1",
        "rule": "replace only if mean test precision AND recall (3 seeds) are not worse",
        "champion_test_mean": champion, "screen_seed0": screen,
        "picked_on_validation": pick["config"], "blind_used": False,
    }  # fmt: skip
    decision = "kept champion: validation picks the deployed setting"
    cfg = (pick["config"]["lookback"], pick["config"]["step"], pick["config"]["smooth"])
    if cfg != CONFIGS[0] and not a.screen_only:
        full = _run(cfg, (0, 1, 2), frames, save=False)
        doc["challenger_3_seeds"] = full
        t = full["test"]
        if (
            t["precision"] >= champion["precision"]
            and t["recall"] >= champion["recall"]
        ):
            decision = "promoted"
            if not a.no_save:
                _run(
                    cfg, (0, 1, 2), frames, save=True
                )  # bundle carries the window settings
        else:
            decision = "kept champion: test precision or recall would drop"
    elif a.screen_only:
        decision = "screen only"
    _use(CONFIGS[0])
    doc["decision"] = decision
    doc["seconds"] = round(time.time() - t0)
    OUT.write_text(json.dumps(doc, indent=1, default=str), "utf-8")
    print(json.dumps({"decision": decision, "picked": pick["config"]}, indent=1))


if __name__ == "__main__":
    main()
