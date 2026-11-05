# 🚨 The SOC: inputs, detection, correlation and the dashboard

[← Back to the main README](../README.md)

What the deployable security operations centre reads, how it detects and explains, and how to use the dashboard. Users and roles, audit log, reports, guaranteed SIEM delivery, agents, REST API, retention, compliance and Windows / macOS / Linux: [ENTERPRISE.md](ENTERPRISE.md).

## 🚨 What the SOC does

- 📥 **Ingests** CICFlowMeter / NetFlow flows, Windows Security, **Sysmon**,
  PowerShell, exported **`.evtx`**, syslog, auditd, Apache / Nginx / IIS, Exim /
  Postfix, Suricata (EVE and fast.log), Zeek, firewall logs, CloudTrail, Kubernetes
  audit, Azure / GCP / Okta / O365, `.zip` archives and folders up to 100 GB. The
  format is detected automatically and original timestamps are kept. Unknown formats
  are still checked (structure, indicators, rarity) and a parser is suggested for
  you to approve.
- 🛡️ **Detects** with 48 endpoint rules, Linux / web / cloud log rules and 2,507
  **SigmaHQ** community rules (release r2026-07-01, Detection Rule License 1.1,
  `third_party/sigma`), plus the ML models in the coverage table.
- 🧩 **Correlates** events per entity (IP, else user, else host) into incidents with an
  ATT&CK attack path and severity. Behaviour rules catch what single events cannot:
  password spraying, cloud reconnaissance, access-denied probing and web scans.
- 💡 **Explains** each alert: indicators for the class plus *why this event* was
  flagged (TreeSHAP for the tree models), with a step-by-step write-up of the
  technique.
- 📤 **Forwards** to Splunk (HEC), Wazuh / QRadar / Elastic (syslog, CEF) or a webhook,
  and exports an ATT&CK Navigator layer.
- 🩺 **Checks its input**: warns when flows are in different units or miss columns, and
  matches column names across CICFlowMeter versions.

## 🔗 How the models work together

- **Shared risk board** (`src/soc/riskboard.py`): every finding is posted per entity
  (IP, user, host, URL, domain, file) and entities seen together are linked (a logon
  ties a user to an IP). Each incident lists what other model families found about
  the same entity or one link away (busy hubs such as proxies are skipped). The
  severity boost starts **off** and is switched on only when your reviews show it
  does not lower precision.
- **Cross-checks at ingestion** (`src/soc/crosscheck.py`): URLs in logs go to the
  phishing model and threat intel; Sysmon file hashes go to threat intel
  (ThreatFox, MalwareBazaar); executables on this PC are scanned by the malware model
  (low confidence unless confirmed; 0 false alarms on 1,496 signed system files).
- **Place profiles** (`config/places.yaml`): home PC, small office, server or lab
  change only how much is forwarded and how strongly links raise severity, never
  what is detected; critical incidents are always forwarded.

## 🖥️ Watching this PC live

`src/soc/winlive.py` reads the new records of this PC's **Security**, **Sysmon** and
**PowerShell** logs every 2 seconds (was 10; adjustable) (Windows' own `wevtutil`) and stores them like any
uploaded log: same rules, incidents, cross-checks, allow-list and SIEM forwarding. It is
read-only, runs with your normal rights (task *SplitShield-XAI live logs*, at logon;
`scripts/install_winlive_task.ps1`) and continues from the last record it saw, so
nothing is read twice or missed. Security and Sysmon need one administrator step,
then sign out and in:

```powershell
net localgroup "Event Log Readers" $env:USERNAME /add
```

The **🏁 Getting started** page shows each log as reading ✅ or what it still needs.

## 🔎 Search and 📄 reports

- Ask in plain English on **💬 Ask SplitShield**: *show failed logins for bob
  yesterday*, *find powershell -enc*, *alerts from 203.0.113.45 in the last 2
  hours*. The answer says exactly which filters were used, then shows a table.
- **Speed at any size** (`src/soc/eventindex.py`): the stored events get an index on
  disk (`events.db` next to `events.jsonl`, rebuilt by itself if missing). Searches,
  "is X happening?" and incidents read only what they need, and only the entities with
  new events are re-correlated. Answers are the same as scanning every event (tested on
  random streams after every append). Above **10,000 matches** a search shows
  "10,000+" (newest first) with a **🔢 Count exactly** button. Measured medians are on
  **Settings → Performance** ([numbers](RESULTS.md)); the setting there only changes
  how much memory and CPU are used, never the answers.
- *make a report of today / yesterday / this week* writes a Markdown and a Word
  report to `results/reports/` (download buttons appear). A daily report is made
  every morning (and a weekly one on Mondays). Reports cover incidents, findings by
  model family, top entities, ATT&CK coverage, what to do, and system health.

## ⚙️ Customise

On the **⚙️ Customise** page:

- **Your own rules:** a simple form (field, contains / regex / equals, optional
  "N times in M minutes") or Sigma YAML. Every rule is tested on your newest 200,000
  stored events first; a rule that would fire on more than 5 % of them needs your
  explicit OK.
- **Allow-list:** IPs, users, hosts, files or domains you trust, each with a reason
  and an expiry. Their alerts are muted (not forwarded, not counted in reports) but
  stay stored, searchable and visible; critical incidents still notify, marked
  "allow-listed entity involved".
- **Sensitivity:** make a model stricter than its validated threshold (fewer
  alerts), never looser.

## 🎯 Deployed detector recipe (`deploy.py`)

✅ Every choice is made on **validation** data; test data is only reported.

1. 🧹 **Preprocessing:** standard vs robust (outlier clipping, log-scaling of
   heavy-tailed rates); the better one is kept.
2. 🔍 **Model search:** tuning grid, then the **fit doctor** retrains when the fit check
   finds over- or underfitting.
3. 🌱 **Self-sizing network:** Net2Net neuron growth (wider, then deeper), then pruning
   of weak neurons and connections, each step kept only if it helps.
4. 🤝 **Ensemble:** MLP, LightGBM and Random Forest, alone or averaged. LightGBM class
   weights are capped at 50× the median so rare classes cannot destabilise it.
5. 🌡️ **Calibration and alerting:** temperature scaling; two-tier alerts (low /
   confirmed, about 1 % false-alarm budget); per-class thresholds; conformal "model
   unsure" flags; an energy-based check for manipulated inputs.

## 🖥️ Dashboard guide

The menu is grouped into sections; every page also has its own address
(`http://localhost:8501/incidents`, or `?page=incidents`).

| Section | Page | Use it to |
|---|---|---|
| Overview | Home | See what needs attention: alerts today, open incidents, this PC's logs, stored events, analysis speed, health - each card links to its page |
| | Getting started | See what is set up (password, Sysmon, SIEM, tasks, live logs) and fix what is not |
| | Ask SplitShield | Ask or search in plain English: *what is a brute force attack?*, *show failed logins for bob yesterday* |
| Monitor | Live monitor | Start monitoring (demo replay, live capture or logs only); watch alerts arrive |
| | Incidents & review | Read correlated attack chains; mark alerts *Real* / *False alarm* (this teaches the model) |
| Investigate | Analyze logs | Drag in any logs, a `.zip` or a folder path (up to 100 GB); big files are analysed by several workers |
| | Explain attacks | Step-by-step explanation of any attack or ATT&CK technique and how to defend |
| | Check URL / file / sensors | Phishing check for a URL, malware check for a file, ICS sensor CSV, insider activity |
| Models & data | Test any dataset | Score the detectors on your own labelled dataset and get a report card |
| | New log formats | Review parser suggestions for unknown formats and approve them |
| | Add dataset | Fine-tune the detector on your own labelled flows |
| | Learning (reviews) | What the detector learned from your reviews; retrain; improve the assistant |
| | Drift & autopilot | How far live traffic is from the training data; automatic decisions |
| | Self-improvement & health | Tabs: health of all 9 models and every automatic part; keep / roll-back decisions; teach a model with new labelled data; speed and code speed-ups to approve |
| | Updates & threat intel | Switch the update agent on or off; feed status; look up an IP, domain or URL; roll back |
| Settings | Connect SOC | Send alerts to Splunk / Wazuh / QRadar / Elastic and receive logs back |
| | Customise | Your own rules, the allow-list and per-model sensitivity |
| | Performance | Automatic, a **memory budget in GB**, or a **target in logs per second**; shows the predicted and measured speed of this PC (calibration) |
| | Places & backup | Where the SOC runs (alerting budget) and where daily backups go |
| | Password | Set or change the dashboard password (only a salted hash is stored) |
| Research | Project results | Research report, extra datasets, log mapping, area models, blind tests |

### ⚡ Performance (memory and speed)

`src/soc/perf.py` turns the setting into a plan: how many log-analysis workers
(`src/soc/parallel.py` parses big line-by-line files in several processes, with the same
events as one process) and how much memory the search index may cache. **Calibrate** on
the page measures this PC once. On the development PC (12 cores, 16 GB): one worker
about 44,000 logs/s end to end, two workers about 61,000 logs/s; more workers do not add
speed there because storing the events becomes the slower step, so a bigger memory
budget then goes to the search cache (faster searches on millions of events).

### 🛡️ Hostile input

Log text is written by whoever produced the log, an attacker included. Values from logs
are escaped wherever the dashboard shows them as Markdown (no injected links or remote
images), over-deep JSON is read as plain text, and a custom regular-expression rule that
could backtrack catastrophically on a crafted line is refused when saved (shape check
plus a timed test in a separate process). Several processes may write the stores at the
same time (live watcher, monitor, uploads): a cross-process lock keeps every line whole.
`scripts/stress_test.py` checks all of this.

## 🔐 Privacy and security

- Dashboard bound to `localhost`; an optional password is stored only as a salted hash.
- Nothing is uploaded anywhere by default; SIEM forwarding is opt-in per output, and
  the update agent only downloads, over HTTPS, when you switch it on.
- Untrusted log content is never rendered as HTML or executed; archives are
  size-checked and unsafe paths rejected.
- Model files are loaded only from this project's `artifacts/` folder.
