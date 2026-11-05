"""MITRE ATT&CK Navigator layer export of the observed incidents (owner 2026-10-14).

The JSON opens in https://mitre-attack.github.io/attack-navigator/ ("Open Existing
Layer" -> "Upload from local"): every technique seen in an incident is coloured by
how many incidents used it, with the incident count, highest severity and source
IPs as the comment. Built locally; nothing is uploaded by this code.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any


def layer(
    incidents: list[dict[str, Any]], name: str = "SplitShield-XAI incidents"
) -> dict[str, Any]:
    count: dict[str, int] = defaultdict(int)
    sev: dict[str, int] = defaultdict(int)
    ips: dict[str, set[str]] = defaultdict(set)
    for inc in incidents:
        for t in inc.get("techniques", []):
            tid = t["id"] if isinstance(t, dict) else str(t)
            count[tid] += 1
            sev[tid] = max(sev[tid], int(inc.get("severity", 0)))
            if inc.get("src_ip"):
                ips[tid].add(str(inc["src_ip"]))
    top = max(count.values(), default=1)
    return {
        "name": name,
        "versions": {"attack": "16", "navigator": "5.1.0", "layer": "4.5"},
        "domain": "enterprise-attack",
        "description": f"Exported {time.strftime('%Y-%m-%d %H:%M')} from "
        f"{len(incidents)} incident(s).",
        "techniques": [
            {
                "techniqueID": tid,
                "score": n,
                "comment": f"{n} incident(s), max severity {sev[tid]}/10; sources: "
                + ", ".join(sorted(ips[tid])[:10]),
                "enabled": True,
            }
            for tid, n in sorted(count.items())
        ],
        "gradient": {"colors": ["#fff7b2", "#ff6666"], "minValue": 0, "maxValue": top},
        "legendItems": [],
        "showTacticRowBackground": True,
        "selectTechniquesAcrossTactics": True,
    }
