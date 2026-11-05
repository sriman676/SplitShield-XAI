# Internet update agent

[← Back to the main README](../README.md)

`src/soc/updater.py` keeps the SOC's detection content current. It is **off by
default**; switch it on from the dashboard or with
`python -m scripts.update_agent --enable`. The setting and the run history are in
`artifacts/soc/update_agent.json`.

## What it may update (nothing else)

| Source | From | Stored in |
|---|---|---|
| Sigma rules | latest SigmaHQ release, `sigma_core+.zip` (tag kept) | `third_party/sigma/` |
| MITRE ATT&CK Enterprise | latest `mitre/cti` release, `enterprise-attack.json` | `src/knowledge/attack_enterprise.json` (rebuilt by `build_attack_kb`, same format) |
| OWASP Top 10 | newest edition from `OWASP/Top10` (2025 now): categories and their CWEs | `src/knowledge/owasp_top10.json`; web alerts get their category (e.g. A05:2025 Injection) |
| OWASP Core Rule Set | latest `coreruleset/coreruleset` release: request rules + their word lists, file by file | `third_party/coreruleset/rules/` (staging and the previous copy inside that folder) |
| URLhaus, Feodo Tracker, ThreatFox, OpenPhish | free public feeds, no sign-up | `artifacts/intel/<feed>.json` (with source + fetch date) |

**How often** (owner 2026-10-23): Sigma, ATT&CK, OWASP Top 10 and CRS are checked **once a
month**; the threat-intel feeds **every day** (bad IPs and URLs go stale within days). A
failed download is retried the next day. The task still runs daily and skips what is not
due.

**New ATT&CK tactics:** the knowledge base keeps ATT&CK's own tactic order, and incidents
follow it. A tactic the severity table does not know gets the neutral weight 5 and you get
a desktop notification to review it (`TACTIC_WEIGHT` in `src/soc/correlate.py`).

**Windows Defender and CRS:** Defender flags CRS files (they contain the attack strings the
rules look for). Exclude `third_party\coreruleset` in Windows Security (docs/INSTALL.md) to
use CRS; if files are removed, the update is rejected with that reason and the current
rules stay.

It never updates Python packages. It only makes plain HTTPS GET requests with a
User-Agent header, and it never sends the owner's e-mail address or any local data.

## Safety gate (every update)

1. Download to `artifacts/updates/staging/` and validate it. The file must parse,
   and the counts must be sane compared with the current version: Sigma usable rules
   must not drop below 80 % of the current count; ATT&CK technique count and
   detection/mitigation coverage must not drop below 90 %; a feed must not be empty
   or shrink below 10 %.
2. False-alarm check on the fixed benign sample `artifacts/intel/benign_sample.jsonl`
   (normal Loghub Apache and Linux lines plus this PC's System and Application
   events, built once). The check compares Sigma alerts, or feed hits on the
   sample's indicators plus well-known benign ones (8.8.8.8, google.com, ...),
   BEFORE and AFTER the update. The update is rejected if the rate rises by more
   than 0.1 percentage points.
3. The project's tests (`pytest -q -p no:warnings`) run on a sandbox copy of the
   project with the staged files in place. If several updates fail together, each
   one is retried alone to find which one breaks the tests.
4. Only after all three checks pass is the update applied. The replaced version is
   kept in `artifacts/updates/previous/<source>/`, and every rejection is recorded
   with its reason.

The Sigma fair-test exclusions (`src/soc/sigma.py`: rules by, or citing, the authors
of the held-out test sets) are applied by the rule loader, so they still hold for new
rules. Validation records how many rules were excluded.

Network failures and timeouts are logged, and the source is retried the next day.
They never crash the agent.

## Commands

```
python -m scripts.update_agent --enable | --disable | --status
python -m scripts.update_agent            # what the daily task runs (no-op when off)
python -m scripts.update_agent --force    # run now
python -m scripts.update_agent --dry-run  # full gate, applied to a temp copy only
python -m scripts.update_agent --rollback sigma   # or attack / urlhaus / feodo / ...
```

Daily task (04:00, skipped when off). Run this once in an Administrator PowerShell:
`powershell -ExecutionPolicy Bypass -File scripts\install_update_task.ps1`.

## Lookups

```python
from src.soc import intel
intel.lookup_ip("50.16.16.211")       # [{"source": "feodo", "threat": ..., "fetched": ...}]
intel.lookup_domain("c2.example.net")  # the domain or any parent domain
intel.lookup_url("http://1.2.3.4/x")   # exact URL, or its host
```

Private IPs never match. A bad URL on a shared host (github.com, drive.google.com,
...) matches only as that exact URL, never as the whole domain.
