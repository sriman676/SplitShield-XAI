"""Extra ML detection areas for the SOC (owner 2026-10-15).

Self-contained modules with a small, stable API that the dashboard and the SOC
engine call:

  * ``src.mlareas.phishing`` - ``check_url`` / ``check_urls`` / ``improve`` / ``blind_test``
  * ``src.mlareas.ics``      - ``score_window`` / ``improve`` / ``blind_test``
  * ``src.mlareas.malware``  - ``check_file_features`` / ``improve`` / ``blind_test``

Train / validation / test results (3 seeds) come from ``python -m scripts.train_<area>``.
"""
