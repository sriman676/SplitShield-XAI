"""Windows recordings WITHOUT Sysmon that the rules miss - DEV half only (owner 2026-10-25).

Lists, for the EVTX-ATTACK-SAMPLES development half (scripts.extra_datasets.is_dev),
every recording with no Sysmon events whose labelled tactic is not found, with its
event ids, so rules for the built-in Windows logs (Security, PowerShell, System, Task
Scheduler) can be written from it. The held-out half is never read here.

    python -m scripts.win_nosysmon_misses > misses.txt
"""

from __future__ import annotations

from collections import Counter

from scripts import extra_datasets as ed


def main() -> None:
    from src.knowledge.attack_details import tactics_of
    from src.soc import parsers

    base = ed.RAW_DIR / "evtx-attack-samples"
    for f in sorted(base.rglob("*.evtx")):
        if not ed.is_dev(f.name):
            continue
        truth = ed.TACTIC_DIRS.get(f.parent.name)
        parsers.reset_sessions()
        try:
            evs = parsers.read_evtx(f)
        except Exception:  # noqa: BLE001
            continue
        if any(e.get("source") == "sysmon" for e in evs):
            continue
        tacs = {x for e in evs for t in e.get("techniques", []) for x in tactics_of(t.split(".")[0])}
        if truth is None or truth in tacs:
            continue
        eids = Counter(str(e.get("detail", "")).split(":")[0] for e in evs)
        print(f"{f.parent.name} | {f.name} | found {sorted(tacs)[:4]} | {dict(eids.most_common(6))}")


if __name__ == "__main__":
    main()
