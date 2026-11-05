# 🔐 Security review and penetration test

[← Back to the main README](../README.md)

An internal penetration test of the whole project, run before the final code freeze
(2026-10-17), plus a re-test after the freeze (2026-10-23, below). The scope is this computer's SOC
(dashboard on localhost), the LAN log receiver and REST API, SIEM delivery, agents,
file uploads and parsers, the update agent, installers and scheduled jobs, secrets on
disk and third-party libraries. This is an internal review by the developer, **not an
independent audit**.

## Findings and fixes

| # | Area | Finding | Severity | Fix | Test |
|---|---|---|---|---|---|
| 1 | Log receiver | The TLS handshake ran in the accept loop: one client that connected and never handshook stopped every other agent (denial of service from the LAN) | High | Handshake moved into each connection's thread; every connection has a 30 s time limit | `test_collector.py::test_idle_or_slow_clients_cannot_block_the_receiver` |
| 2 | Log receiver | Slow clients could hold threads indefinitely (slowloris) | Medium | 30 s socket timeout per connection | same |
| 3 | Log receiver / API | An unexpected error dropped the connection with no answer | Low | Clean `500 internal error`, never a traceback | `test_collector.py` |
| 4 | Dashboard sign-in | The failed sign-in back-off was per browser session, so new sessions reset it | Medium | Per-username lockout: 5 failures in 5 min lock that user for 15 min (audited) | `test_users_audit.py::test_username_lockout_after_repeated_failures` |
| 5 | Search / API | `last 999999999 hours` crashed the search (date overflow); found by fuzzing 320 hostile queries (SQL / FTS injection, control characters, random input) | Low | Relative windows clamped to 100 years | `test_search_reports.py::test_hostile_searches_never_crash` |
| 6 | Report schedules | A schedule's folder was not checked | Low | Must be an existing folder given as a full path | `test_report_builder.py` |
| 7 | SIEM delivery | A kept-open TCP / TLS connection the receiver had closed could swallow the next alert | Medium | Closed connections are detected before each send and reopened | `test_sinks_live.py::test_syslog_tcp_newline_framed` |
| 8 | Test isolation | The default alert outbox path was fixed at import time, so test runs wrote alerts into the real outbox (never sent anywhere; quarantined, not deleted) | Medium | The default follows the sandbox | `test_delivery.py::test_default_outbox_follows_the_sandbox` |
| 9 | Libraries | `cryptography` 44.0.2: 6 advisories (certificate verification, OpenSSL in wheels) | Medium | 50.0.0 | collector / agent tests |
| 10 | Libraries | `lief` 0.16.6: crash on a malformed ELF file (could stop a file scan) | Low | 0.17.2; PE features identical on 200 Windows system binaries | malware scan tests |

## Re-check after the freeze (2026-10-23)

Scope: everything added or changed since the first test (cases, silent agents and clock
skew, batch ids, pseudonymisation, live-delay settings, per-source "unknown" entities,
ICS v3 scoring, the assistant), every place that shows log content, and the libraries.

| # | Area | Finding | Severity | Fix | Test |
|---|---|---|---|---|---|
| 11 | Dashboard / incident write-up | Log content (user and file names, event details, findings) went into Markdown unescaped: a log line could put a live link or a **remote image** (which tells an attacker an analyst opened the incident) into the SOC dashboard; incident titles, the alert pop-up, findings, narratives and insider rows too. Missed by the first test | Medium | Every log-derived value escaped (`md()`); incident write-ups escape entity, sources, findings, event kind and detail, id | `test_pentest_recheck.py::test_incident_markdown_escapes_log_content` |
| 12 | SOC reports | The same values were written raw into the Markdown report (a link or remote image in a Markdown viewer; a `\|` broke tables in all three formats) | Low | Log values cleaned to one line without pipes or link / image syntax (plain text in Markdown, Word and HTML) | `test_pentest_recheck.py::test_reports_keep_log_content_inert` |
| 13 | REST API | Incidents returned the full case, including analysts' note text, to read-only API tokens | Low | The API returns status, assignee, last update and the number of notes only | `test_pentest_recheck.py::test_api_case_has_no_note_text` |
| 14 | Receiver | An agent could send `inf` / `nan` as its clock | Low | Non-finite clock values are ignored | `test_pentest_recheck.py::test_agent_clock_must_be_finite` |
| 15 | Receiver / agents | A batch re-sent after a lost answer was stored twice (integrity, found under load) | Medium | Batch ids; the receiver stores a batch once (14 days of ids) | `test_agent.py::test_a_batch_whose_answer_was_lost_is_stored_once` |
| 16 | Libraries | `pillow` 12.3.0 (35 advisories), `scapy` 2.5.0 (1), `certifi` (1), `black` (1, developer tool) | Medium | pillow 13.0.0 (with streamlit 1.66.1, which allows it), scapy 2.8.0 (live capture checked), certifi 2026.10.19, black 26.10.1; SBOM regenerated | full test suite |

Checked, no new finding: batch ids are per agent (one agent cannot suppress another's
batches) and bounded by the rate limit; cases need the `review` permission and limit note
size; live-delay settings are clamped to at least 0.5 s; the pseudonym key is a random
32-byte secret next to the settings; the ICS column map and models are written only by
the project's own training; the assistant only reads the store.

## Accepted risks

| Risk | Why accepted | Mitigation |
|---|---|---|
| `lightgbm` 4.5.0 advisory PYSEC-2024-231 (code execution when loading a *crafted model file*) | Upgrading the library could change the registered research training; the SOC only loads its own model files | Model files are written only by the project's own training, and anyone able to replace them already controls the account |
| Agents and the receiver use a self-signed certificate | Agents pin its SHA-256 fingerprint, which for this use is stronger than trusting any CA | Using your own PKI certificate is future work |
| UDP syslog cannot confirm delivery | True of every syslog client | TCP / TLS / webhook are guaranteed |
| The update agent trusts GitHub over HTTPS (no signature on Sigma / ATT&CK releases) | Neither project signs its releases | HTTPS only, size limit, zip-slip guard, only rule files kept, the whole test suite must pass, rollback |

## Checked, no finding

- **Dashboard:**
  - Bound to `localhost`; Streamlit XSRF protection on.
  - No `unsafe_allow_html`; log text is Markdown-escaped.
  - Pages hidden per role; asking for a forbidden page opens Home.
  - Disabled or removed users are signed out on their next click.
- **Code:** no `shell=True`, `os.system`, `eval` / `exec` or unsafe YAML loading;
  subprocess arguments are lists. Paths reach PowerShell through environment
  variables, never into the command.
- **Uploads:**
  - Only the base file name is used (no path traversal).
  - Zip archives are limited in file count and total size; nested JSON depth is limited.
  - Custom regular expressions are checked for catastrophic backtracking and timed
    before they are saved.
- **Receiver:**
  - Agent keys and API tokens are compared in constant time and stored as SHA-256.
  - Agent keys cannot read the API.
  - LAN-only address check; gzip bombs refused; per-agent rate limit.
- **Secrets on disk:**
  - Passwords are salted PBKDF2; keys and tokens are hashes.
  - The webhook token can come from the `SOC_TOKEN` environment variable instead of
    `settings.json`.
  - The agent's key file is readable only by its owner / SYSTEM.
- **Audit:** every change is in a SHA-256 hash chain; editing the history is detected.
