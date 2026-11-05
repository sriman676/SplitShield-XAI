"""Real-SIEM check: push incidents to a LOCAL Splunk and confirm they are searchable.

Prepared 2026-10-14, run after the training run once Docker Desktop and the
Splunk container (deploy_siem/docker-compose.splunk.yml) are up. Uses the SOC's
own sink code (src/soc/sinks.py) in an isolated temporary store, tags every test
event with a unique marker, then searches Splunk's REST API for that marker.
Writes results/siem_live_check.json. Credentials come from deploy_siem/.env and
never leave this PC (localhost only).

    python -m scripts.siem_live_check splunk
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "deploy_siem" / ".env"


def _env() -> dict[str, str]:
    out = {}
    for line in ENV.read_text("utf-8").splitlines():
        k, _, v = line.partition("=")
        if k.strip() and not k.startswith("#"):
            out[k.strip()] = v.strip()
    return out


def _search(marker: str, password: str, wait_s: int = 60) -> int:
    """Number of Splunk events containing ``marker`` (REST export search)."""
    auth = base64.b64encode(f"admin:{password}".encode()).decode()
    body = urllib.parse.urlencode(
        {"search": f'search index=* "{marker}" | stats count', "output_mode": "json"}
    ).encode()
    deadline = time.time() + wait_s
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False  # the container's own self-signed management cert,
    ctx.verify_mode = ssl.CERT_NONE  # reached on 127.0.0.1 only
    while time.time() < deadline:
        req = urllib.request.Request(
            "https://127.0.0.1:8089/services/search/jobs/export",
            data=body,
            headers={"Authorization": f"Basic {auth}"},
        )
        with urllib.request.urlopen(req, context=ctx, timeout=30) as r:  # noqa: S310
            for line in r.read().decode().splitlines():
                if line.strip():
                    n = int(json.loads(line).get("result", {}).get("count", 0))
                    if n:
                        return n
        time.sleep(5)  # indexing lag
    return 0


def splunk() -> dict[str, Any]:
    from scripts.soc_selftest import _isolate
    from src.soc import sinks

    env = _env()
    tmp = Path(tempfile.mkdtemp(prefix="siem_check_"))
    _isolate(tmp)
    s = sinks.apply_preset(sinks.load_settings(), "Splunk")
    s["webhook"] |= {
        "url": "http://127.0.0.1:8088/services/collector/event",
        "token": env["SPLUNK_HEC_TOKEN"],
    }
    sinks.save_settings(s)
    marker = f"splitshield-check-{uuid.uuid4().hex[:10]}"
    incidents = [
        {"id": f"{marker}-{i}", "src_ip": f"198.51.100.{i + 1}", "severity": 7,
         "level": "high", "start": time.time(), "end": time.time(),
         "techniques": [{"id": "T1110", "name": "Brute Force", "tactic": "credential-access"}],
         "summary": f"{marker} brute force test {i}"}
        for i in range(3)
    ]  # fmt: skip
    sent = [sinks.send(inc) for inc in incidents]  # the same path live alerts use
    found = _search(marker, env["SPLUNK_PASSWORD"])
    res = {"siem": "splunk", "marker": marker, "sent": 3, "push_result": str(sent),
           "found_in_splunk": found, "ok": found >= 3}  # fmt: skip
    out = ROOT / "results" / "siem_live_check.json"
    out.write_text(json.dumps(res, indent=2), "utf-8")
    return res


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "splunk"
    if target != "splunk":
        print("Only 'splunk' is prepared; Wazuh is set up after Docker is installed.")
        return 2
    res = splunk()
    print(f"[siem-check] {res}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
