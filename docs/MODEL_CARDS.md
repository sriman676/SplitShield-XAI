# 🪪 Model cards

[← Back to the main README](../README.md)

One card per deployed model: what it is for, what it was trained and tested on, how
well it works, and where it should **not** be trusted. Every number is on data the
model never saw while being built: settings were chosen on a validation split and
the test split was scored once. *Blind* = a whole other dataset, scored once at the
very end (see the bottom of this page and `results/extra/blind_tests.json`).

---

## 🌐 Network-flow detector (CICFlowMeter): `artifacts/deploy/`

| | |
|---|---|
| Purpose | Flag attack flows in CICFlowMeter-style flow records and name the attack family |
| Model | Flow stack v4 (deployed 2026-10-25, owner exception): the MLP + LightGBM ensemble (robust scaling, temperature-calibrated) with the DDoS specialist cascade, plus a multi-dataset generalist LightGBM; a stack LightGBM trained with PGD-perturbed attacks combines them, confirmed alert at attack score ≥ 0.85. "Model unsure" and energy "unlike training data" flags. Diagram: Figure 2 of the final report |
| Deployed test result (stack v4) | CIC-IDS2017: precision 0.9992, attack recall 0.9982, 0.24 % false alarms; CIC-DDoS2019 recall 0.9996; other networks: CIC-IDS2018 recall 0.873, IoT-DIAD 0.882 (`results/extra/improve_flows_v4.md`) |
| Training data | CIC-IDS2017 original (raw units), 70 / 15 / 15 split, duplicates removed |
| Test result (ensemble alone, before the stack) | 99.97 % of attacks caught, 0.21 % false alarms, precision 99.94 %, macro-F1 0.985 |
| Fit | train / val / test macro-F1 0.992 / 0.942 / 0.985 (the validation dip comes from tiny Infiltration and Heartbleed counts) |
| Robustness | Deployed flow stack v4 (adversarial training, owner exception 2026-10-25): PGD flips 29 % (ε 0.1) / 31 % (ε 0.3) of detected CIC-IDS2017 attacks (before: 41 % / 62 %). Earlier models: PGD evasion flips 31–49 % of detected attacks, against 15–20 % for random noise of the same size; DDoS and DoS are robust (`results/extra/robust_deployed.md`) |
| DDoS specialist (2026-10-17) | A small LightGBM (attack vs benign, trained on CIC-IDS2017 + CIC-DDoS2019 day 03-11) adds a DDoS alert when the champion calls a flow benign and the specialist scores it ≥ 0.519 (threshold chosen on validation). CIC-DDoS2019 test: attack recall **0.0003 → 0.9998**, precision 0.314 → 0.999. CIC-IDS2017 test: recall 0.99969 → 0.99986, precision 0.99945 → 0.99917, false alarms 0.178 % → 0.267 %; scoring 3–10 % slower. Added as an **owner-approved exception** to the precision rule (`results/extra/improve_flows.json`, key `cascade`). Every other flow is scored exactly as before; previous detector in `artifacts/deploy_backup/20261017-0949_pre_ddos_specialist` |
| Do not trust it for | Traffic from a different flow exporter or network without checking the 📈 Drift page first; Infiltration (few training examples) |

## 🌐 NetFlow detector: `artifacts/deploy_netflow/`

| | |
|---|---|
| Purpose | Binary attack / benign verdict on NetFlow v9 records (39 features) |
| Model | LightGBM + MLP + Random Forest ensemble, robust scaling, threshold floor 0.5 |
| Training data | NF-CSE-CIC-IDS2018-v2; 3 seeds |
| Test result | macro-F1 0.981 ± 0.001; 96.3 % of attacks caught at 0.07 % false alarms |
| Limits | 293 "Infilteration" test flows are byte-identical to benign flows, so they cannot be caught. **It does not transfer to other networks** without retraining: macro-F1 0.35 on NF-UNSW-NB15 and 0.44 on NF-ToN-IoT. |

## 🛰️ Zeek connection model: `artifacts/zeekml/`

| | |
|---|---|
| Purpose | Score Zeek `conn.log` records as malicious or benign |
| Model | LightGBM or Random Forest on connection features (whichever is better on validation); threshold chosen on validation by Youden's J |
| Training data | IoT-23, split **by malware capture**, so test captures come from malware families the model never saw |
| Test result | Random Forest chosen on validation (threshold 0.67). Mean of 3 seeds: **F1 0.995**, 99.9 % of malicious connections caught, precision 0.991, **6.0 % false alarms**; the deployed seed: precision 0.996, recall 0.9995, **2.8 % false alarms** (`results/extra/logs_zeek.json`, `improve_zeek.json`) |
| Retest 2026-10-29 | Zeek 9.0.1 converted the third benign IoT-23 honeypot capture (7-1) and it was added to training only: just 359 connections; test precision 0.9956 → 0.9931, false alarms 2.8 % → 4.5 %, so the deployed model is kept |
| Limits | In IoT-23 the "benign" traffic comes from the same infected devices. A strict 1 % alert budget left nothing detectable, so expect more false alarms than the flow detectors. |

## 🎣 Phishing URL model and page model: `artifacts/mlareas/phishing/`

| | |
|---|---|
| Purpose | Rate a URL as phishing or legitimate from the URL text alone (no page fetch) |
| Model | LightGBM on hostname features only (v4, 2026-10-16: `host2` features - brand edit distance, top-level-domain statistics, 2/3/4-letter-gram hashes, letter-pair rarity) |
| Training data | PhiUSIIL (UCI 967), stratified 70 / 15 / 15, 3 seeds |
| Test result | F1 0.807, precision 0.946, recall 0.704, 2.5 % of legitimate URLs flagged |
| Why hostname only | In PhiUSIIL every legitimate URL is a bare homepage, so full-URL models score F1 0.998 by learning URL *shape*. That is leakage, not skill, so the full-URL model is not deployed. |
| Threshold (v3) | 0.955, re-chosen on validation with real popular sites (Tranco ranks 20k-100k): 1.8 % of real popular sites flagged; PhiUSIIL test precision 0.994, recall 0.590, F1 0.740. Previous version kept for rollback |
| v4 (2026-10-16) | Trained with 300,000 real legitimate domains (Tranco ranks 200k-1M) and the richer hostname features; threshold 0.927 chosen on validation. PhiUSIIL test precision 0.9941 → **0.9959**, recall 0.590 → **0.592**; real popular sites flagged as phishing (Tranco test 100k-200k) 1.84 % → **0.02 %** (`results/extra/improve_phishing_host2.json`). v3 kept in `prev/` |
| v5 (deployed 2026-10-24) | v4 training data + Mendeley 2021 training-domain hosts; threshold 0.916 chosen on validation. PhiUSIIL test precision 0.997, recall 0.602; Tranco 100k-200k test 0.15 % flagged; blind CompPhish recall 0.71 at 2.0 % false alarms (`results/extra/improve_phishing_v5.json`) |
| Page model v2 (2026-10-25, owner exception) | A second LightGBM reads uploaded pages and e-mail attachments (forms, links, brands, scripts; no page is ever fetched). Retrained with CompPhish 70 %: on the 30 % check recall 0.39 → 0.64, false alarms 4.0 % → 1.2 %, precision 0.88 → 0.98 (`results/extra/improve_phishing_compphish.json`). Diagram: Figure 3 of the final report |
| "Suspicious" band (2026-10-17) | Starts at 0.477: the lowest probability that at most 2 % of real popular sites (Tranco VALIDATION 20k-100k) reach - 1.8 % there, 1.5 % on the Tranco test sample (`results/extra/phishing_band.json`) |
| Limits | Phishing on trusted shared hosts (Google Drive, GitHub pages and similar) looks like its host; the strict threshold misses phishing (recall 0.59). A few very large brand domains score high (google.com 0.79, "suspicious"): PhiUSIIL phishing often sits on Google-hosted pages; use the allow-list for your own trusted domains |

## 🦠 Malware file model: `artifacts/mlareas/malware/champion/`

| | |
|---|---|
| Purpose | Score a Windows PE file as malicious from EMBER features |
| Model | LightGBM, regularised v2 (num_leaves 256, learning rate 0.05, min 200 rows per leaf, L2 10, 30 % features and 70 % rows per tree; 510k training rows); thresholds for 1 % and 0.1 % FPR chosen on validation. v1 kept in `prev/` |
| Training data | EMBER 2018 v2 features; train up to 2018-09, validation 2018-09/10, test 2018-11/12 (time-ordered) |
| Test result | ROC-AUC 0.994; **92.6 %** detected at 1 % false positives (v1: 91.7 %), precision 0.990; 69.3 % at 0.1 % (v1: 68.7 %). Promoted 2026-10-22: detection and precision both better, smaller train-test gap |
| Speed | about 99k files/s from feature vectors; about 7.5k files/s including feature extraction |
| Limits | Static features only, so packed or brand-new families are harder; it never runs the file |

## 🏭 ICS anomaly model: `artifacts/mlareas/ics/current/`

| | |
|---|---|
| Purpose | Find attacks in industrial sensor streams without attack labels for training |
| Model | Autoencoder trained on normal operation; the score is the largest per-sensor reconstruction error |
| Training data | HAI 22.04 + HAI 23.05 normal periods (v3); the last 15 % of each is validation |
| Alarm budget | At most **5 false alarms/day** on normal validation data (owner's rule); the threshold is the most sensitive one that stays within it |
| Test result | **2.06 false alarms/day**, 20 of 58 attack episodes found, precision 0.72, median delay 39 s |
| Trade-off | Without the budget it finds 47 / 58 episodes but raises about 130 false alarms/day (`results/extra/ics_unbudgeted.md`) |
| v2 (2026-10-17) | Owner switched to **time-split attack validation**: HAI 22.04 attack files test1+test2 (24 episodes) choose the threshold and variant, test3+test4 (34 episodes) are the test. Each sensor's error is now divided by its own 99th percentile on normal validation data (per-sensor normalisation). On the new test part: precision 0.893 → **0.895**, recall 0.166 → **0.167**, 15 of 34 episodes, 0 false alarms/day (3-seed mean 0.896 / 0.167). Variants with higher recall (0.39-0.66) were rejected: 21-46 false alarms/day on the test part (`results/extra/improve_ics_attack_split.json`). v1 kept in `prev/` |
| **v3 (2026-10-22, deployed)** | More data: **HAI 23.05** added (same plant, newer run; 7 renamed setpoint tags matched by value range, `P1_B4005` dropped; 74 sensors). Same autoencoder, trained on the normal data of both versions; threshold on HAI 22.04 test2 within 5 false alarms/day. 22.04 test3+4: precision 0.895 → 0.896, recall 0.167 → 0.139 (14 vs 15 of 34 attacks). **HAI 23.05 test2: v2 alarmed almost nonstop (precision 0.04); v3 precision 0.70, recall 0.23, 8 of 38 attacks, 5.1 false alarms/day.** Deployed as an **owner exception** to the no-drop rule (works on both plant versions). v2 backup: `artifacts/deploy_backup/20261022-2344_pre_ics_v3`. Script: `scripts/train_ics_v3.py`; results `results/extra/ics_v3*.json` |
| Tried, not deployed (v3) | Hybrid autoencoder errors + 10/60/300 s trend features → LightGBM trained on labelled attacks (22.04 test1, 23.05 test1): precision 0.94 but recall 0.11 and severe overfitting (train recall 1.00, validation 0.10) - each HAI attack hits different sensors, so labelled attacks do not generalise to new ones. A sensor-independent version (top-5 deviations across sensors) was worse (recall 0.04). HAIEnd 23.05 internal DCS tags and BATADAL (a water network, test F1 0.31-0.43, report only) were also tried |
| Fit | Underfitting (train and validation F1 on injected faults both < 0.05; more data does not help - `results/extra/fit_report.md`): the autoencoder cannot separate most HAI attacks from normal drift |
| Limits | Slow, subtle manipulations stay under the threshold; retrain per plant |

## 👤 Insider-threat model: `artifacts/mlareas/insider/current/`

| | |
|---|---|
| Purpose | Daily risk score per user (user-day), with plain-English reasons |
| Model | LightGBM (15 leaves, 558 trees), threshold 0.899 chosen on validation |
| Training data | CMU CERT r4.2 (synthetic organisation, 1,000 users, 70 insiders), 82 features per user-day |
| Test result | Time split: F1 0.943, PR-AUC 0.980, 0.84 alerts/day. Unseen users: F1 0.875 |
| Limits | CERT is **synthetic**. Some web-volume features are partly a simulation artefact, so expect lower accuracy on real organisations. Use it as a triage aid, never as a verdict on a person. |

## 💬 Assistant question-understanding model: `artifacts/nlp/intent.joblib`

| | |
|---|---|
| Purpose | Work out what a question asks (explain an attack, check logs, how to defend, follow-up and so on) |
| Model | TF-IDF (word and character) + linear classifier, with topic memory for follow-ups |
| Test result | **87.6 %** on question templates never seen in training (grouped CV 88.0 % ± 1.9) |
| Safety | Answers come from the knowledge base and your logs. The small language model only words incident narratives, and any narrative that fails the faithfulness check is replaced by a template. |

## 🪟🐧☁️ Log rules (not ML): `src/soc/winrules.py`, `logrules.py`, `sigma.py`

| Area | Fair held-out result |
|---|---|
| Windows (EVTX-ATTACK-SAMPLES half + OTRF) | 53 % right tactic, 47 % right technique; 1.9 % alerts on this PC's normal logs |
| Linux / web / mail / IDS (AIT-LDS held-out servers) | 67 of 77 attack steps (87 %) at 0.08 % false alarms |
| AWS CloudTrail (Invictus) | 72 of 210 attack calls (17 attack API types), 1.9 % false alarms; about 6.8k events/s on flaws.cloud |

---

## 🙈 Blind tests (scored once, at the very end)

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
