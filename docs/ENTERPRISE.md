# 🏢 Enterprise features: running SplitShield-XAI next to a SIEM

SplitShield-XAI is a **detection and response layer that works alongside a SIEM**. The
SIEM stays the system of record. SplitShield adds explainable ML detections,
cross-model correlation and investigation help, then forwards what it finds. This page
covers what was added for teams and companies (2026-10-16/17).

| Need | Where | Module |
|---|---|---|
| One account per person, roles, permissions | Settings → Users & roles | `src/soc/users.py` |
| Who did what, tamper-evident | Settings → Audit log | `src/soc/audit.py` |
| Models keep learning, or frozen | Models & data → Evolve or freeze | `src/soc/evolution.py` |
| Reports for any period, own template, schedules | Overview → Reports | `src/soc/reports.py` |
| Alerts never lost on the way to the SIEM | Settings → Connect SOC | `src/soc/delivery.py`, `sinks.py` |
| SIEM formats CEF, LEEF, ECS, OCSF; syslog over TLS | Settings → Connect SOC | `src/soc/sinks.py` |
| Logs from other machines (agents) | Settings → Agents & API | `src/soc/collector.py`, `scripts/agents/` |
| Read-only REST API for SIEM / SOAR | Settings → Agents & API | `src/soc/collector.py` |
| Local retention (keep N days) | Settings → Places & backup | `src/soc/retention.py` |
| Incident cases (status, assignee, notes) | Monitor → Incidents & review | `src/soc/cases.py` |
| Silent agents, clock skew | Settings → Agents & API | `src/soc/heartbeat.py` |
| Pseudonymised names to the SIEM | Settings → Connect SOC → Privacy | `src/soc/sinks.py` |
| NIST CSF 2.0 / ISO 27001 evidence | Reports → section "Compliance evidence" | `src/soc/compliance.py` |
| Windows, macOS, Linux | `INSTALL.cmd` / `install.sh` | `src/soc/osops.py`, `services.py`, `oslive.py` |
| Change any built-in rule, engine settings, rule packs, proposals | Settings → Customise | `src/soc/rulectl.py` |
| Fit the flow detector / ICS model to your own site | Customise → Proposals; `scripts/ics_site_fit.py` | `src/soc/sitecal.py`, `src/mlareas/ics.py` |

---

## 👥 Users, roles and permissions

- **Sign-in:** username + password. On a new install, the first page asks for the
  admin account. An existing single dashboard password becomes the user `admin`.
- **Roles** are sets of permissions. Defaults: **admin** (everything), **analyst**
  (view, review, investigate, rules) and **viewer** (view). The admin can create roles
  such as `responder` and change what each role may do.

| Permission | Allows |
|---|---|
| `view` | alerts, incidents, search, reports, results |
| `review` | mark alerts Real / False alarm, rate answers (this teaches the models) |
| `investigate` | analyse uploaded logs, check URLs / files, test datasets |
| `rules` | own detection rules, allow-list, sensitivity |
| `models` | add datasets, retrain, evolve / freeze, approve code speed-ups |
| `settings` | SIEM connection, performance, places and backup, updates, retention |
| `collection` | agent keys and API tokens |
| `reports` | edit the global report template and schedules (anyone with `view` makes reports) |
| `users` | users, roles, permissions, audit log (**admin only**) |

- **Only an admin** adds users, assigns roles, and changes usernames, passwords and
  permissions (the admin's own too). The last admin can never be removed, demoted or
  disabled, so the SOC cannot be locked out.
- **Passwords** are stored as salted PBKDF2 hashes. Failed sign-ins back off
  exponentially (2, 4, 8 … 30 s).
- **Forgotten admin password:** on this computer run
  `.venv\Scripts\python.exe -m scripts.set_dashboard_password [username]`. The
  reset is recorded in the audit log as `local_recovery`.
- Pages a role may not use are hidden from its menu, and the review buttons need
  `review`.

## 🧾 Audit log

Every sign-in (successful or not) and every change is appended to `audit.jsonl` next
to the event store, with who, when, what and the details. This covers users and
roles, rules, allow-list, sensitivity, SIEM settings, reviews, retrains, code
proposals, evolution, retention, reports, agents and tokens. Each entry contains the
SHA-256 of the previous one. **Editing or deleting any past entry breaks the chain**,
and the Audit log page names the first broken entry. The log is never trimmed by
retention. With *Connect SOC → Delivery guarantee → Also send the audit log*, every
entry also goes to the SIEM.

## 🛠️ Adapt the rules to your site (2026-10-24)

Everything that decides an alert can be changed by a user with the *customise*
permission, on **Settings → Customise**. Every change is tested on the stored logs
first where that makes sense, applies to new logs, can be undone, and is written to
the tamper-evident audit log (who, when, before / after, why).

| Tab | What a user can do |
|---|---|
| My rules | Own rules: a simple form (field / contains / regex / equals, optional "N times in M minutes") or Sigma YAML |
| Built-in rules | Search all 3,612 built-in rules (3,161 Sigma, 192 OWASP CRS, 259 LOLBAS / GTFOBins) and our own rules as they fire; switch one off, override its severity (1-10) or ATT&CK technique, add exceptions ("not when host equals backup01") |
| Engine settings | Correlation gap, risk-board weight / cap / hops / attacker-to-victim follow, OWASP CRS paranoia level and anomaly threshold - each next to its validated value |
| Rule packs | Export own rules + changes + settings as one JSON file; import a pack (every rule compiled and tested on the stored logs, every value checked) before switching it on |
| Proposals & history | Changes the system suggests from the analysts' Real / False-alarm reviews (a rule whose reviewed alerts were all false alarms; false alarms concentrated on one host or user -> an exception), the risk board's per-site weight / hops, and the per-site flow threshold. Nothing changes until a user accepts. The history lists every change |
| Sensitivity | Make a model stricter or looser than validated (looser shows the extra alerts it would add on the stored logs first; never below 0.5) |

**Per-site calibration.** The flow detector keeps a sample of the attack scores of the
site's own traffic; after 50,000 flows and 3 days it proposes the threshold at which at
most 1 % of the site's flows alert (never looser than validated). The ICS model can be
fitted to one plant from about a week of its normal operation
(`python -m scripts.ics_site_fit fit --normal DIR --name PLANT --activate`).

## 🧬 Evolve or freeze

- **Evolving (default):** models keep learning from new labelled data and analyst
  reviews. A new version is kept **only if held-out precision and recall do not
  drop** and it is not slower.
- **Frozen:** no model changes by itself, but detection, alerting and review
  collection go on, so the models can learn later.
- Single models can be frozen while the rest evolves. The switch is audited.

## 📄 Reports

- **Periods:** today, yesterday, this week (since Monday), last 7 days, this month,
  last 30 days, any **date range** (first and last day included), any **time range**
  (from day + time to day + time), or one incident.
- **Global template:** the title, intro text, which of the 14 sections in which order,
  how many top entities and incident write-ups, and the file formats (**Word .docx,
  Markdown, HTML**). Sections: summary, incidents, findings by model family, top
  entities, ATT&CK coverage, what to do, alerts over time, data collected, analyst
  reviews, models, system health, user activity (audit), allow-listed, compliance
  evidence.
- **One report only:** "What goes in (this report only)" changes the template for
  that report and leaves the global one unchanged.
- **After making it:** the report can be edited (Markdown) and saved again; the
  downloads then hold the edited version.
- **Schedules:** name, period, weekdays, time, optionally their own sections, formats
  and folder. Defaults: *Daily* (yesterday, 07:00), *Weekly* (last 7 days, Monday
  07:00) and *Weekly compliance* (Monday 08:00). The autopilot makes each one once a
  day, on its first cycle after the set time.
- **The assistant** understands requests such as "make a report from 2026-10-01 to
  2026-10-07", "report for 2026-10-03", "report for this month".
- HTML reports are self-contained and escape all log content.

## 📤 SIEM delivery: guaranteed, in standard formats

**Formats** (per output):

| Output | Formats |
|---|---|
| Syslog (UDP / TCP / **TLS**) | CEF (ArcSight, Sentinel, Wazuh), **LEEF 2.0** (QRadar), **ECS JSON** (Elastic), **OCSF 1.3 Detection Finding** (Security Lake, SecOps) |
| Webhook / HTTP | SplitShield JSON, Splunk HEC envelope, ECS, OCSF |
| JSON-lines file (agent tails it) | SplitShield JSON, ECS, OCSF |

Presets: Wazuh, Splunk, Elastic, QRadar, Sentinel, Security Lake (OCSF), Generic.

**Syslog over TLS** (RFC 5425, port 6514):
- The receiver's certificate is checked against the Windows / system store, or
  against your CA file.
- An optional client certificate gives mutual TLS.
- One connection is kept open; it reconnects automatically if the receiver closed it.

**Guaranteed delivery:**
- An alert that cannot be delivered goes into a durable queue (`delivery.db`) and is
  retried **in order**: after 30 s, doubling up to 1 h.
- While the SIEM is down, new alerts are queued at once (a circuit breaker), so an
  outage never slows log analysis.
- After *Keep retrying for* (default 7 days), or beyond *Largest queue* (default
  1,000,000), alerts move to `dead_letter.jsonl`. They are never deleted, and
  **Put dead-lettered alerts back in the queue** resends them.
- The Connect SOC page shows per output: delivered, waiting, retried, dead-lettered,
  oldest waiting and last error.

## 🛰️ Central log collection and the REST API

- **The receiver:** `python -m src.soc.collector serve` (or switch it on in Agents &
  API and install it with the background jobs). It runs an HTTPS server on the LAN,
  port **8600**. The dashboard itself stays on `localhost`.
- **Security:**
  - Self-signed certificate made on first start; agents **pin its SHA-256
    fingerprint**, so an impostor server is refused.
  - One **agent key** per machine and one **API token** per tool, 256-bit random,
    shown once; only their SHA-256 is stored. Each can be disabled or removed, and
    every change is audited.
  - **Local network only:** private / loopback / link-local addresses. The firewall
    rule (`scripts\allow_collector_firewall.ps1` as Administrator, or
    `sudo sh install.sh --firewall`) also limits the port to the local subnet.
  - 10 failed authentications from one address within 5 minutes block it for 15
    minutes.
  - Bodies are capped at 16 MB (64 MB after gzip; gzip bombs are refused), with a
    per-agent rate limit (default 200 MB/min).
- **Agents:** the logs go through exactly the same parsers, rules and models as
  local files.

| Machine | Agent | Reads | Install |
|---|---|---|---|
| Windows (no Python) | `splitshield_agent.ps1` | Security, Sysmon, PowerShell, System | `install_agent.ps1` as Administrator (Task Scheduler, runs as SYSTEM) |
| Linux | `splitshield_agent.py` | systemd journal, auth.log / secure / syslog / messages | `sudo sh install_agent.sh` (systemd) |
| macOS | `splitshield_agent.py` | unified log (sign-ins, sudo, sshd, security), system.log | `sudo sh install_agent.sh` (launchd) |

  - Each agent saves where every source stopped **only after** the batch was sent or
    spooled.
  - Batches that cannot be sent wait in a local `spool/` and are re-sent in order.
  - Log files are followed through rotation.

- **Read-only API** (send `Authorization: Bearer <token>`):
  - `GET /api/v1/health`: events stored, SIEM queue.
  - `GET /api/v1/incidents?since=<unix time>&level=high&limit=200`
  - `GET /api/v1/alerts?since=&limit=`
  - `GET /api/v1/search?q=failed logins for bob`

  Nothing can be changed through the API. Agent keys cannot read it.

## 🗄️ Local retention

- **Default:** keep **30 days** in the live store. Older events and alerts are
  **archived** as gzip in `archive/` (they can be analysed again by uploading them),
  or **deleted** if you choose that.
- **Optional:** delete archives older than N days.
- **Safety:**
  - The store is rewritten in place under the same lock every writer uses, so no line
    arriving meanwhile is lost.
  - An interrupted run finishes on the next start.
  - The search index rebuilds itself after a run.
- Runs nightly (01:00–05:00) and from **Apply now**. The audit log is never trimmed.

## ✅ Compliance evidence (NIST CSF 2.0, ISO/IEC 27001:2022)

- The *Compliance evidence* report section lists 20 controls. Each is marked
  **evidence**, **partial** or **gap**, with the facts the SOC recorded in the period
  and, for gaps, what to do. The facts include:
  - events per sensor family, incidents, multi-sensor correlation;
  - analyst reviews, accounts and roles, sign-ins;
  - audit-chain integrity, retention, backups;
  - SIEM delivery, model update cycles.
- **NIST CSF 2.0:** DE.CM-01/03/09, DE.AE-02/03/06, RS.AN-03, RS.MA-02, PR.AA-05,
  PR.PS-04, PR.DS-11, ID.IM-01.
- **ISO 27001:2022 Annex A:** 5.15, 5.25, 5.26, 5.28, 8.2, 8.13, 8.15, 8.16.
- It **supports an audit; it is not a certification**. The *Weekly compliance*
  schedule produces it every Monday.

## 🗂️ Incident cases

Every incident can be worked as a case on Incidents & review:
- **status:** New → Investigating → Contained → Resolved or False positive;
- an **assignee** (assigning moves New to Investigating);
- **notes**.

Changing a case needs the `review` permission, and every change is in the audit log.
The case status is shown in reports and returned by the API (`case` field). Marking a
case "False positive" never changes a model: teaching the detector stays the separate
Real / False alarm review. A status filter (open cases by default) keeps the list short.

## 🔕 Silent sources and agent clocks

- **Silent agents:** an agent that sends nothing for *Alert when an agent sends nothing
  for* minutes (default 30) raises **one** alert, ATT&CK T1562 *Impair Defenses*,
  because an attacker who stops logging looks like "no attacks". It raises one "source
  is back" event when the agent sends again. The receiver checks every minute.
- **Clock skew:** agents send their clock time with every batch. The receiver keeps
  the difference ("clock off by" on Agents & API) and warns above 2 minutes, because
  wrong clocks move events outside the correlation windows.

## 🕶️ Privacy: pseudonymised names in the SIEM

*Connect SOC → Privacy → Replace user and host names in everything sent out* (off by
default). Names become `u-…` / `h-…` codes: a keyed SHA-256 with a secret made once on
this computer. The same name always gets the same code, so the SIEM can still
correlate. The dashboard keeps showing real names.

## ⏱️ Alert delay

`python -m scripts.latency_benchmark` measures each path end to end, from the event
to the alert at a (mock) SIEM: a log line, an agent over HTTPS, a live flow, building
and forwarding an incident, and a dataset upload. Worst case = measured processing +
the waiting built into each path:
- this computer's logs are read every 2 s (was 10 s; Settings → live delay);
- agents send every 2 s (was 10 s; `--every`);
- live flows wait up to 0.5 s for their batch (was 1 s);
- incidents are forwarded every 1 s (was 5 s).

Results: `results/extra/latency.md`.

## 🏎️ Fast mode for retraining

`python -m scripts.fast_mode --batch 512 -- run_pipeline.py --experiment … --seed …`
(run in a copy) trains with a larger batch. No research file is edited: a start-up
hook overrides the batch size. Fast-mode runs get their own run ids, and the
registered results always come from the normal mode. A batch size is accepted only if
every hypothesis verdict is unchanged and no run's held-out precision, recall or
macro-F1 drops (`--compare`).

## 💻 Windows, macOS and Linux

The whole SOC runs on all three:
- **Install:** `INSTALL.cmd` on Windows, `sh install.sh` on Linux / macOS, then
  `START.cmd` / `sh START.sh`.
- **Background jobs** (`python -m src.soc.services install | remove | status`):
  autopilot (hourly), backup (03:30), update agent (04:00), this computer's live logs,
  and the log receiver when switched on. They use Task Scheduler, launchd agents or
  systemd user units and run as you, without admin rights. On Linux, run
  `loginctl enable-linger $USER` once so they also run while you are signed out.
- **This computer's logs:**
  - Windows: Security / Sysmon / PowerShell; needs the "Event Log Readers" group.
  - Linux: journal + auth logs; `sudo usermod -aG adm,systemd-journal $USER`.
  - macOS: unified log; Full Disk Access for the .venv Python if files are unreadable.
- **Windows `.evtx` exports** can be analysed on macOS / Linux too. On 10,255 attack
  sample events they give the same detections as on Windows (4 differ, where the
  portable reader is *more* exact).
- **Blocking attackers:**
  - Windows Firewall (Administrator).
  - nftables / iptables on Linux (root).
  - pf on macOS (root, after `sudo sh install.sh --firewall`).
- **Pop-ups:** msg.exe / osascript / notify-send.
- **Tested:** on Windows directly. Linux and macOS are covered by automated tests of
  their units, plists and code paths. A real Linux run needs Docker or WSL on this PC,
  and no Mac was available.

## 🧪 How this is tested

- `tests/test_users_audit.py`, `test_dashboard.py` (sign-in, roles), `test_evolution.py`
- `test_report_builder.py`, `test_compliance.py`
- `test_delivery.py` (outage, order, dead letter, LEEF/ECS/OCSF, TLS)
- `test_collector.py` (pinning, lockout, gzip bombs, LAN-only, API)
- `test_agent.py` (agent ↔ receiver, outage spool, impostor certificate)
- `test_retention.py` (concurrent writers, crash recovery)
- `test_crossplatform.py`, `test_evtx_portable.py`
- `scripts/combo_stress.py`: every combination of the options (41,472 + per-model
  freezes) through the whole SOC pipeline, with mock SIEMs.
