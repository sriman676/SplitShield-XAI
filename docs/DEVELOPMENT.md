# 🛠️ Development, layout and changelog

[← Back to the main README](../README.md)

For working on the code.

## 🗂️ Files

```
INSTALL.cmd / START.cmd   one-step install / start the SOC dashboard
LAUNCH.cmd                full rebuild from datasets (1_SETUP_AND_RUN.ps1 → 2_AI_ENGINEER.ps1)
dashboard.py              SOC dashboard (Streamlit)
deploy.py                 trains the deployed flow detector
run_pipeline.py           research stages 0–9         run_matrix.py   H4 skew matrix
improve.py                self-improvement            ai_engineer.py  orchestrates a full run
config/                   experiments (YAML), mutability maps, defence knowledge base
src/soc/                  parsers, logformats, winrules, logrules, sigma, correlate, engine,
                          sinks, drift, auth, columns, zeekml, netflow_model, unknown,
                          evolve, updater, intel, controller, codeopt,
                          model_updates
src/mlareas/              phishing, malware (EMBER), ics (HAI), insider (CERT)
src/nlp/                  intent model, ShieldLM, RLCD, compression, log-status answers,
                          feedback (per-prompt learning)
src/knowledge/            ATT&CK Enterprise (706 techniques) and attack write-ups
src/data/ models/ training/ attack/ defense/ explain/ eval/   research code
artifacts/                trained models (deploy/, deploy_netflow/, mlareas/, zeekml/, nlp/)
results/                  FINAL_REPORT.md, extra/ (held-out and blind results)
scripts/                  installers, autopilot, update agent, backup, train_<area>,
                          extra_logs, blind_test, test_dataset, make_portable
docs/                     INSTALL, SOC, RESULTS, MODEL_CARDS, AUTOMATION, ASSISTANT,
                          UPDATE_AGENT, RESEARCH, DATASETS, LIMITATIONS, GLOSSARY,
                          ENTERPRISE, SECURITY, PREREG_H2b, PREREG_H2c,
                          DEVELOPMENT (.md), architecture.png
third_party/sigma/        SigmaHQ rules r2026-07-01 (Detection Rule License 1.1)
tests/                    778 tests
```

## ➕ Adding a new model

Every specialist model is declared in **`config/models.yaml`**: its folder, family,
speed probe, update step, inbox pattern and the entities its findings are about. The
self-improvement controller, the 🧬 health page, the shared risk board, the inbox and
the model cards all read that file, so nothing else needs editing.

```powershell
.venv\Scripts\python.exe -m scripts.new_model dns_tunnel --title "DNS tunnelling model" `
    --family network --entities ip,domain --inputs "labelled DNS log CSV"
```

This writes a template module (`src/mlareas/dns_tunnel.py` with `load`, `score`,
`speed` and `improve` to fill in), a wiring test (`tests/test_mlarea_dns_tunnel.py`)
and the registry entry. Train the model, save it under
`artifacts/mlareas/dns_tunnel/current/` with a fixed `holdout.npz` for its update gate
and a `probe.npz` for its speed probe, then run the tests. Its findings join the risk
board under the chosen family automatically.

No paths are hard-coded: everything is relative to the project folder, so the project
runs from any drive or a pendrive. The backup folder is a dashboard setting
(🏁 Getting started).

## 🛠️ Development

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt   # full (research) dependencies
.venv\Scripts\python.exe -m pytest -q                         # 778 tests
.venv\Scripts\python.exe -m ruff check .                      # lint
.venv\Scripts\python.exe -m mypy                              # types
```

## 🧪 How it was tested

- **778 automated tests**; **every combination of the options (41,483)** run end to end;
- a **30-minute soak** (10 agents → receiver → SIEM, with a SIEM outage and a receiver
  restart): 360,100 / 360,100 lines stored once, 35,963 / 35,963 alerts delivered once;
- hostile input, concurrency, faults, 1 M and 10 M events; a real Linux host (WSL Ubuntu);
- a penetration test plus a re-check after the freeze ([SECURITY](SECURITY.md));
- the research re-run twice and compared number by number with the registered results, and
  pre-registered confirmatory tests of H2 ([PREREG_H2b](PREREG_H2b.md), [PREREG_H2c](PREREG_H2c.md)).

## 📝 Changelog

| Date | Change |
|---|---|
| 2026-10-15 | Shared risk board and cross-model checks (phishing URLs, hash intel, LIEF file scan); place profiles; search by prompt; daily / on-request reports (Markdown + Word); custom rules, allow-list and stricter-only sensitivity; model registry (`config/models.yaml`) + `scripts/new_model.py`; assistant answer-first / steps / adaptive style / typo tolerance; Sigma 2.1x and phishing 6x faster; phishing threshold v3; Linux / Azure AD / O365 rules |
| 2026-10-15 | One self-improvement controller over all 9 models (held-out + speed gate, rollback, inbox); per-prompt assistant learning; nightly speed profiling with owner-approved code proposals; health watch and notifications; Sysmon installed; docs split into docs/*.md |
| 2026-10-15 | Update agent (Sigma, ATT&CK, threat intel; off by default, gated); blind tests on 6 datasets; ICS alarm budget; H2 and T1 follow-ups; NetFlow detector; Zeek model; phishing, malware, ICS and insider models; Linux / web / mail / cloud rules; generalised Sigma engine; unknown-format handling; evolve loop; one-step installer; model cards |
| 2026-10-14 | Extra datasets (NF-ToN-IoT, CIC-DDoS2019, NF-UNSW multi-class); Windows log-mapping fair test (19 % → 53 %); dashboard UX (any-format upload, help, getting started); real-world log samples |
| 2026-10-14 | Full unattended run complete: registered research (H1–H5), corrected follow-up, raw-unit deployed detector, autonomy (autopilot, backups, resume) |

## 📌 How to cite

```bibtex
@software{splitshield_xai_2026,
  title  = {SplitShield-XAI: split-learning intrusion detection with explanations
            and a self-improving SOC},
  author = {SplitShield-XAI contributors},
  year   = {2026},
  note   = {Version 0.2.0}
}
```

Please also cite the datasets you use (see *Sources and citations*).

## 🧾 Third-party notices

- **SigmaHQ rules** (`third_party/sigma`): © their authors, Detection Rule License
  (DRL) 1.1; authors are credited in each rule file.
- **MITRE ATT&CK®** (`src/knowledge/attack_enterprise.json`): © The MITRE Corporation,
  used under the ATT&CK terms of use; ATT&CK is a registered trademark of MITRE.
- **Threat-intel feeds** (downloaded only when the update agent is on): URLhaus,
  ThreatFox and Feodo Tracker by abuse.ch; OpenPhish community feed. Each is subject to
  its provider's terms.
- Datasets are © their respective authors; see *Sources and citations*.
