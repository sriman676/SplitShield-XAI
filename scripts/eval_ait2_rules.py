"""AIT-LDS v2 after the 2026-10-24 rule fixes - NOT BLIND (owner 2026-10-24).

AIT-LDS v2 was scored once in blind round 2 (results/extra/blind_round2.json, kept
as recorded). The owner then allowed it for writing rules: russellmitchell was used to
write them, santos is the check. Same scorer as the blind round (blind_test_v2._ait_server).

    python -m scripts.eval_ait2_rules
Writes results/extra/ait2_rules_after.json.
"""

from __future__ import annotations

import json
import time

from utils.paths import RESULTS_DIR


def main() -> None:
    from scripts.blind_test_v2 import BLIND2, _ait_server

    out: dict = {"label": "NOT BLIND: rules written from russellmitchell after blind round 2; santos is the check"}
    for name, role in (("russellmitchell", "rules written from it"), ("santos", "check (rules never saw it)")):
        r = _ait_server(BLIND2 / "ait-lds-v2" / f"{name}.zip")
        r.pop("files", None)
        out[name] = {"role": role, **r}
        print(name, r["steps_caught"], "/", r["n_steps"], round(r["line_recall"], 4), r["false_alarm_rate"], flush=True)
    out["when"] = time.strftime("%Y-%m-%d %H:%M")
    (RESULTS_DIR / "extra" / "ait2_rules_after.json").write_text(json.dumps(out, indent=1, default=float), "utf-8")


if __name__ == "__main__":
    main()
