"""Linux real-host test (owner 2026-10-24): the Linux agent running on a real Linux
(WSL 2 Ubuntu, systemd) sends to the HTTPS receiver on this Windows PC.

1. Windows: a sandboxed receiver on ``--port`` (0.0.0.0), a new agent key and the pinned
   certificate fingerprint.
2. Ubuntu: the agent (scripts/agents/splitshield_agent.py, Python 3.8+, no packages) with
   a test log of known attack lines, the real /var/log/auth.log and the systemd journal;
   ``--once`` twice (the second pass must send nothing again: bookmarks).
3. Checks: every test line stored once, events from the Linux host name, journal and
   auth.log events arrived, the SSH brute force became an incident, nothing twice.

    python -m scripts.linux_host_test [--distro Ubuntu] [--port 8611]
Results: results/extra/linux_host_test.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from utils.paths import PROJECT_ROOT, RESULTS_DIR

OUT = RESULTS_DIR / "extra" / "linux_host_test.json"


def wsl(distro: str, cmd: str, timeout: float = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["wsl.exe", "-d", distro, "--", "bash", "-lc", cmd], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)  # fmt: skip


def run(distro: str, port: int) -> dict[str, Any]:
    from src.soc import collector, engine, sandbox, sinks

    sandbox.isolate(Path(tempfile.mkdtemp(prefix="splitshield-linux-")))
    s = sinks.load_settings()
    s["collector"] |= {"lan_only": True}
    sinks.save_settings(s)
    collector._detector()  # models loaded before listening, as `collector serve` does
    srv = collector.make_server(port=port, bind="0.0.0.0")  # noqa: S104 - test receiver for the WSL host
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    gw = wsl(distro, "ip route show default | head -1 | cut -d' ' -f3").stdout.strip()  # no $ (wsl.exe)
    linux_host = wsl(distro, "hostname").stdout.strip()
    pyver = wsl(distro, "python3 --version").stdout.strip()
    home = "/tmp/splitshield-agent-test"
    marks = [f"LINUXTEST{i:03d}" for i in range(40)]
    t = time.strftime("%b %d %H:%M:%S").replace(" 0", "  ")
    lines = [f"{t} {linux_host} sshd[{4000 + i}]: Failed password for root from 203.0.113.77 port {50000 + i} ssh2 {m}"
             for i, m in enumerate(marks)]  # fmt: skip
    cfg = {"server": f"https://{gw}:{port}", "key": collector.new_agent("linux-test", "wsl-ubuntu"),
           "fingerprint": collector.fingerprint(), "files": [f"{home}/test_auth.log", "/var/log/auth.log"],
           "journald": True, "macos_unified": False, "windows_logs": [], "backfill": True}  # fmt: skip
    agent_src = (PROJECT_ROOT / "scripts" / "agents" / "splitshield_agent.py").read_text("utf-8")
    wsl(distro, f"rm -rf {home} && mkdir -p {home}", timeout=60)
    for name, text in (("splitshield_agent.py", agent_src), ("agent.json", json.dumps(cfg)),
                       ("test_auth.log", "\n".join(lines) + "\n")):  # fmt: skip
        p = subprocess.run(["wsl.exe", "-d", distro, "--", "bash", "-lc", f"cat > {home}/{name}"], input=text,
                           capture_output=True, text=True, encoding="utf-8", timeout=60)  # fmt: skip
        assert p.returncode == 0, p.stderr
    first = wsl(distro, f"cd {home} && python3 splitshield_agent.py --home {home} --once 2>&1 | tail -5")
    n_after_first = engine.event_count()
    second = wsl(distro, f"cd {home} && python3 splitshield_agent.py --home {home} --once 2>&1 | tail -3")
    time.sleep(2)
    evs = engine.read_jsonl(engine.EVENTS)
    found: dict[str, int] = {}
    hosts, sources = set(), set()
    for e in evs:
        d = str(e.get("detail", ""))
        hosts.add(str(e.get("host") or ""))
        sources.add("journald" if "systemd[" in d or "wsl-" in d else ("auth.log" if "sshd[" in d or "sudo" in d else "other"))
        i = d.find("LINUXTEST")
        if i >= 0:
            mk = d[i:i + 12]
            found[mk] = found.get(mk, 0) + 1
    sample = [{k: (str(v)[:80]) for k, v in e.items()} for e in evs[:2] + evs[-3:]]
    incs = engine.incidents()
    brute = [i for i in incs if i["src_ip"] == "203.0.113.77"]
    srv.shutdown()
    res = {"when": time.strftime("%Y-%m-%d %H:%M"), "distro": distro, "linux_host": linux_host, "python": pyver,
           "receiver": cfg["server"], "agent_first_pass": first.stdout.strip()[-400:],
           "agent_second_pass": second.stdout.strip()[-300:], "events_stored": len(evs),
           "events_after_first_pass": n_after_first, "test_lines": len(marks),
           "test_lines_stored_once": sum(1 for m in marks if found.get(m) == 1),
           "test_lines_twice": sum(1 for m in marks if found.get(m, 0) > 1),
           "sources_seen": sorted(x for x in sources if x)[:20],
           "journal_events": sum(1 for e in evs if "systemd[" in str(e.get("detail", ""))),
           "auth_log_events": sum(1 for e in evs if "LINUXTEST" not in str(e.get("detail", ""))
                                  and ("sshd[" in str(e.get("detail", "")) or "sudo" in str(e.get("detail", "")))),
           "sample_events": sample, "brute_force_incident": bool(brute), "incident_severity": brute[0]["severity"] if brute else None}  # fmt: skip
    res["checks"] = {"every test line stored once": res["test_lines_stored_once"] == len(marks),
                     "nothing stored twice (bookmarks)": res["test_lines_twice"] == 0 and len(evs) == n_after_first,
                     "events carry the Linux host name": linux_host in hosts,
                     "systemd journal read": res["journal_events"] > 0,
                     "SSH brute force became an incident": res["brute_force_incident"]}  # fmt: skip
    res["ok"] = all(res["checks"].values())
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distro", default="Ubuntu")
    ap.add_argument("--port", type=int, default=8611)
    a = ap.parse_args()
    res = run(a.distro, a.port)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), "utf-8")
    print(json.dumps(res, indent=1)[:3000])
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
