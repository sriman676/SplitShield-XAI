# 📊 Results

[← Back to the main README](../README.md)

Every number here is on data the system never saw while being built: settings were chosen on a validation split and the test split was scored once. Blind tests are whole extra datasets, scored once at the very end. Per-model details: [MODEL_CARDS.md](MODEL_CARDS.md).

## 🗺️ Coverage: what is checked and how well

What each kind of data is checked with, and how well it did. **Held-out** means
rules were written, or models chosen, on one part of the data and scored on a part
they never saw. **Blind** means a whole extra dataset, scored once at the very end.
Model details are in [`docs/MODEL_CARDS.md`](MODEL_CARDS.md).

| Area | Detection | Data | Held-out result | Blind result |
|---|---|---|---|---|
| 🌐 Network flows (CICFlowMeter) | ML ensemble | CIC-IDS2017 (+ 5 research datasets) | ✅ 99.97 % attacks caught, 0.21 % false alarms | ⚠️ CIC-IDS2018: 44 % caught, 1.9 % false alarms (new DDoS tools missed) |
| 🌐 Network flows (NetFlow) | ML ensemble | NF-CSE-CIC-IDS2018-v2 | ✅ 96.3 % attacks caught, 0.07 % false alarms (macro-F1 0.981) | — |
| 🪟 Windows events | 48 rules + Sigma | EVTX-ATTACK-SAMPLES + OTRF | ✅ 53 % right tactic, 47 % right technique; 1.9 % alerts on normal logs | Splunk attack_data: 49 % (55 / 112 files) right tactic |
| 🐧 Linux / web / mail / IDS | log rules + Sigma | AIT-LDS (4 servers) | ✅ 67 / 77 attack steps (87 %) at 0.08 % false alarms | ⚠️ Splunk attack_data: 6 % (6 / 99 files) right tactic (Linux), 0 % (0 / 18 files) (web) |
| ☁️ AWS CloudTrail | rules + Sigma + behaviour correlation | Invictus (Stratus) + flaws.cloud | ✅ 17 attack actions caught (72 / 210 calls), 1.9 % false alarms; 6.8k events/s | Splunk attack_data: 40 % (16 / 40 files) right tactic |
| 🛰️ Zeek connections | ML (split by malware capture) | IoT-23 | ✅ F1 0.995 on unseen malware captures (99.9 % caught), 6.0 % false alarms on benign connections | — |
| 🎣 Phishing URLs | ML (hostname only) | PhiUSIIL + real popular sites | ✅ 1.8 % of real popular sites flagged, precision 0.99, recall 0.59 (threshold re-chosen on validation; before: F1 0.81) | ⚠️ OpenPhish + Tranco: 89 % caught, 12.6 % false positives (ROC-AUC 0.957) |
| 🦠 Malware files | ML (EMBER features) | EMBER 2018 (time-ordered) | ✅ 92.6 % detected at 1 % false positives, ROC-AUC 0.994 (regularised v2, 2026-10-22) | ✅ EMBER 2017: 93.9 % at 0.04 % false positives (ROC-AUC 0.9996) |
| 🏭 Industrial control (ICS) | ML anomaly detection, ≤ 5 false alarms/day budget | HAI 22.04 | ✅ 2.06 false alarms/day, 20 / 58 attacks found, precision 0.72, median delay 39 s | ❌ HAI 21.03: alarm on all the time (16 sensors missing in the older plant set-up); retrain per plant |
| 👤 Insider threat | ML per user-day | CMU CERT r4.2 | ✅ F1 0.943 (time split), 0.875 on unseen users; 0.84 alerts/day | CERT r4.1: 3/3 insiders flagged, 0.05 alerts/day |
| ☸️ Kubernetes · 🍎 macOS · 🔐 Okta · Azure AD · O365 | Sigma + parsers | Splunk attack_data | ✅ parsed and scored | see blind logs below |
| 🔥 Firewalls (Cisco ASA / Palo Alto / Fortinet / pfSense) | parsing + correlation | vendor formats | ✅ parsed into the common event model | — |
| ❓ Unknown log formats | structure detection, indicators, rarity, Sigma keywords | any text log | ✅ detected and flagged; parser suggestions for the owner to approve | — |
| 🔗 Threat intel (opt-in) | URLhaus, ThreatFox, Feodo, OpenPhish lookups | live feeds | ✅ 0 hits on 100 benign indicators | — |

## 🎯 Deployed flow detector (raw-unit CIC-IDS2017, held-out test data)

| | |
|---|---|
| 🧠 Model | MLP + LightGBM ensemble (chosen on validation), robust preprocessing, temperature-calibrated |
| 🚨 Attacks caught | **99.97 %** (8 missed of about 29,000) |
| 🔕 False alarms | **0.21 %** of benign flows |
| 🎯 Precision | 99.94 % |
| 📈 Macro-F1 train / validation / test | 0.992 / 0.942 / **0.985**: ✅ good fit (the validation dip comes from tiny Infiltration and Heartbleed counts) |
| 🧩 Per-class test F1 | 0.99–1.00 for every family; Infiltration 0.89 (only a handful of examples) |
| 🛡️ Evasion | A PGD attack flips 31–49 % of detected attacks, against 15–20 % for random noise of the same size; DDoS and DoS hold up |
| 🔌 SIEM delivery (self-test) | ✅ Splunk HEC and syslog verified locally |

## 🧬 ATT&CK mapping (fair held-out test)

The rules were written from half of EVTX-ATTACK-SAMPLES only. They were scored on the
other half plus OTRF Security-Datasets, which comes from a different lab. Sigma rules
by the authors of either collection, or that cite them, are excluded (122 rules).

| Rules | Mapped to a technique | ✅ Right tactic | Right technique |
|---|---|---|---|
| Original | 43 % | 19 % | 14 % |
| + 48 endpoint rules (`src/soc/winrules.py`) | 59 % | 37 % | 31 % |
| + 2,507 SigmaHQ community rules (`src/soc/sigma.py`) | 71 % | **53 %** | **47 %** |

On this PC's own normal logs (7,649 events), 1.9 % raised an alert. These were mostly
real activity (PowerShell downloads, service installs, Defender detections), not
noise.

- **Linux / web / mail / IDS (AIT-LDS)**: rules developed on two servers (cup,
  insect) and scored on two others (onion, spiral): **67 of 77 attack steps (87 %)**
  at **0.08 %** false alarms.
- **Cloud (Invictus)**: 17 attack API types caught (72 of 210 attack calls), 1.9 %
  false alarms on background activity. On 1.94 M real flaws.cloud events, 0.7 % alert
  rate.
- **ATT&CK knowledge**: all 706 Enterprise techniques with mitigations and detection
  analytics (`src/knowledge/attack_enterprise.json`, ATT&CK v20.0). Any technique or
  incident is explained in five parts: what it is, how it works step by step, signs
  in your logs, how to stop it, and how to confirm.

## 🙈 Blind tests (scored once, at the very end)

Each frozen model on a whole dataset it never saw and nothing was tuned on
(`python -m scripts.blind_test`, results in `results/extra/blind_tests.json`).

| Model | Blind dataset | Result | Verdict |
|---|---|---|---|
| 🌐 Flow detector | CSE-CIC-IDS2018, 8 days (other network, other year) | 44.4 % of attacks caught, 1.9 % false alarms, precision 0.94. Slowloris / SlowHTTPTest / FTP brute force ≥ 99 %; HOIC and LOIC-UDP DDoS 0 % (tools absent from 2017) | ⚠️ transfers partly: retrain or fine-tune on local traffic |
| 🦠 Malware | EMBER 2017 test set (193,108 files; 6,892 files also in EMBER 2018 removed) | ROC-AUC 0.9996; 93.9 % detected at 0.04 % false positives; 76.2 % at 0 % | ✅ (older malware than training, so easier than a future set) |
| 🎣 Phishing | OpenPhish live feed (300) + Tranco top sites (3,000), 2026-10-15 | 89 % of phishing caught, 12.6 % of popular sites flagged, precision 0.41, ROC-AUC 0.957 | ⚠️ ranks well, but its threshold is too eager on real top sites |
| 🏭 ICS | HAI 21.03 (older version of the same testbed) | All 50 attacks "found" only because the alarm is on all the time (point precision 0.022 = attack share); 16 of the model's sensors do not exist in 21.03 | ❌ does not transfer across plant set-ups: retrain per plant |
| 👤 Insider | CMU CERT r4.1 (another synthetic organisation, 3 insiders) | 3/3 insiders flagged; user-day precision 0.92, recall 0.77; 0.05 alerts/day | ✅ |
| 🪟🐧☁️ Logs | Splunk attack_data, 370 seeded files across 17 log types | 33 % of files raised an alert, 23 % right tactic, 16 % right technique. Windows 49 % (55 / 112 files), AWS 40 % (16 / 40 files); Linux 6 % (6 / 99 files), web 0 % (0 / 18 files), Okta / Azure AD / O365 6 % (3 / 54 files) right tactic | ⚠️ Windows and AWS carry over; Linux, web and identity logs need more rules |

Blind sets are scored once and nothing was tuned on them; the weak spots above are
reported as they came out.

## 🔁 Follow-ups after the blind tests (not blind)

The blind results above stay as registered. Weak spots they showed were then
improved using other data, and scored on held-out parts of that data:

| Area | What changed | Held-out result |
|---|---|---|
| 🐧 Linux / Azure AD / O365 rules | rules written from 82 Splunk attack_data files **not** in the blind sample | right tactic on 76 held-out files: **7 % → 32 %**; benign Loghub logs unchanged; AIT held-out still 67 / 77 at 0.08 % (`results/extra/attackdata_followup.md`) |
| 🎣 Phishing threshold | re-chosen on validation incl. real popular sites (Tranco ranks 20k-100k; the blind pool was never used) | popular sites flagged **14.7 % → 1.8 %** (Tranco 100k-200k); PhiUSIIL test precision 0.94 → 0.99, recall 0.71 → 0.59 (`results/extra/phishing_threshold.md`) |
| ⚡ Speed | Sigma matching and the phishing check | Sigma 7.8k → 16.3k lines/s (identical results); phishing 0.3k → 1.9k URLs/s (`results/extra/speed_followup.md`) |
| 🦠 File scan (new) | LIEF features → malware model | **0 false alarms on 1,496** Microsoft-signed system files; detection on real malware not measured (none downloaded), so hits are low-confidence (`results/extra/pescan_validation.json`) |
| 🔗 Risk board (new) | cross-model severity boost | could not be validated on public data (AIT has no cross-model links): starts as context only and switches on from your reviews (`results/extra/riskboard_validation.json`) |

## ⚡ Speed (this PC: 12 cores, 16 GB RAM, CPU only)

Measured with `scripts/speed_benchmark.py`, `perf.calibrate` and the profiling runs of
2026-10-16/17. Every speed-up was checked to give **identical results**.

| What | Measured |
|---|---|
| Log analysis (parse + rules + store), 1 / 2 workers | 43.9k / **60.7k logs/s** (more workers are slower: one SQLite writer) |
| Live flow scoring (monitor) | 58 → **~2,300 flows/s** (batches of ≤ 256, ≤ 1 s wait) |
| Deployed flow detector, batched | ~34–36k flows/s (with the DDoS specialist) |
| Search at 1 million events | 3–46 ms per query; exact counts on request |
| Assistant answer at 1 million events | 16–195 ms |
| Incidents at 1 million events | cached 2.5 ms; first build 23 s |
| Incidents after 1,000 new events (211k-event store, PC under load) | 6.5 s → **1.0 s** |
| SIEM outage | alerts queued at once instead of waiting up to 5 s each |
| Research training (identical numbers, 1,615 values checked) | E4 69.5 → 61.3 min, E8 46.4 → 35.4 min, T1 6.8 → 5.3 min |

Disk: the search index is about 4.4× the raw store (full-text 30 %, column indexes
20 %, links and incidents 20 %, events 30 %); retention (default 30 days) bounds it. The
10-million-event benchmark is re-run after the final runs (the first attempt stopped).
