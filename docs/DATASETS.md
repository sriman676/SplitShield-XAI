# 📦 Datasets and sources

[← Back to the main README](../README.md)

Datasets are only needed to rebuild the models, not to run the SOC.

## 📦 Datasets

Every dataset was downloaded from its public source **without sign-up**. Datasets are
not needed to *run* the SOC, only to rebuild the models.

| Dataset | Size | Used for |
|---|---|---|
| CIC-IDS2017 corrected / original / with flow IDs | 2.7 M / 2.8 M / 3.1 M rows | research, deployed flow detector, time-ordered test |
| NF-CSE-CIC-IDS2018-v2 / NF-UNSW-NB15-v2 / NF-ToN-IoT-v2 | 17.1 M / 2.0 M / 16.9 M rows | research, NetFlow detector, extra datasets |
| CIC-IoT-2023 / CIC-IoT-DIAD-2024 / CIC-DDoS2019 | 3.8 M / 2.0 M / 20.4 M rows | research and extra datasets |
| EVTX-ATTACK-SAMPLES / OTRF Security-Datasets | 277 / 153 files | Windows rules (half dev, half held-out) |
| AIT-LDS v1.1 | 4 servers, 5.3 GB | Linux / web / mail rules |
| Loghub SSH / Linux / Apache | 655 k lines | benign server logs, evolve benchmark |
| IoT-23 (Zeek) | 14 captures | Zeek model |
| Invictus aws_dataset / flaws.cloud | 2.9 k / 1.9 M events | cloud rules |
| PhiUSIIL / EMBER 2018 / HAI 22.04 / CERT r4.2 | | phishing / malware / ICS / insider models |
| **Blind:** CIC-IDS2018, EMBER 2017, HAI 21.03, CERT r4.1, OpenPhish + Tranco, Splunk attack_data | | scored once at the end |

Data-quality notes: the corrected CIC-IDS2017 is released **min-max normalised** (raw
units cannot be recovered); the original has negative durations and 8.3 % duplicates
(removed); CIC-IoT-2023 has **17 % contradictory labels**.

## 📚 Sources and citations

Licences and terms are set by each dataset's authors. Please check the source page
before reuse and cite the original work:

| Dataset | Cite | Source |
|---|---|---|
| CIC-IDS2017 / CSE-CIC-IDS2018 | Sharafaldin, Habibi Lashkari, Ghorbani, *ICISSP* 2018 | unb.ca/cic/datasets |
| CIC-DDoS2019 | Sharafaldin, Habibi Lashkari, Hakak, Ghorbani, *ICCST* 2019 | unb.ca/cic/datasets |
| CIC-IoT-2023 | Neto et al., *Sensors* 2023 | unb.ca/cic/datasets |
| CIC IoT-DIAD-2024 | Canadian Institute for Cybersecurity, 2024 | unb.ca/cic/datasets |
| NF-*-v2 NetFlow sets | Sarhan, Layeghy, Portmann, *Mobile Netw. Appl.* 2022 | staff.itee.uq.edu.au/marius/NIDS_datasets (mirrors used) |
| UNSW-NB15 / ToN-IoT | Moustafa & Slay 2015 / Moustafa 2021 | research.unsw.edu.au |
| IoT-23 | Garcia, Parmisano, Erquiaga, Stratosphere Lab 2020 | stratosphereips.org |
| AIT Log Data Set v1.1 | Landauer et al., AIT Austrian Institute of Technology | zenodo.org/records/4264796 |
| EVTX-ATTACK-SAMPLES | Samir Bousseaden | github.com/sbousseaden/EVTX-ATTACK-SAMPLES |
| OTRF Security-Datasets | Roberto Rodriguez, Open Threat Research Forge | github.com/OTRF/Security-Datasets |
| Loghub | Zhu et al., *ISSRE* 2023 | github.com/logpai/loghub |
| Splunk attack_data | Splunk Threat Research Team | github.com/splunk/attack_data |
| Invictus aws_dataset / flaws.cloud logs | Invictus Incident Response / Scott Piper (Summit Route) | github.com/invictus-ir/aws_dataset · summitroute.com |
| PhiUSIIL | Prasad & Chandra, *Computers & Security* 2024 | UCI ML Repository #967 |
| OpenPhish / Tranco | OpenPhish community feed / Le Pochat et al., *NDSS* 2019 | openphish.com · tranco-list.eu |
| EMBER | Anderson & Roth, arXiv:1804.04637, 2018 | github.com/elastic/ember |
| HAI | Shin et al., National Security Research Institute (KR) | github.com/icsdataset/hai |
| CERT Insider Threat | Lindauer, CMU SEI, 2020 | kilthub.cmu.edu (figshare 12841247) |

## ♻️ Re-running the research from the shipped copy

The shipped copy holds the code, the 9 trained models and every result, but **not**
the datasets (they are large and each has its own licence). To re-run the research
or retrain a model, download each dataset from the source above and put it under
`data/raw/<folder>`. The loaders (`src/data/sources.py`) find the files by name
inside the folder, so archive layouts can stay as downloaded.

| Folder in `data/raw/` | Dataset |
|---|---|
| `cicids2017-original`, `cicids2017-improved`, `cicids2017-flows` | CIC-IDS2017 original / corrected / with flow IDs |
| `cic-ddos2019`, `cic-iot-2023`, `cic-iot-diad-2024` | CIC-DDoS2019, CIC-IoT-2023, CIC IoT-DIAD-2024 |
| `nf-cse-cic-ids2018-v2`, `nf-unsw-nb15-v2`, `nf-ton-iot-v2` | NetFlow v2 sets |
| `iot-23` | IoT-23 (Zeek conn.log) |
| `evtx-attack-samples`, `otrf-security-datasets` | Windows attack recordings |
| `ait-lds-v1.1`, `loghub`, `splunk-attack-data` | Linux / web / mail logs, benign server logs, blind log set |
| `invictus-aws`, `flaws-cloudtrail` | CloudTrail |
| `phiusiil`, `tranco` | phishing (+ the blind benign domains) |
| `ember2018`, `hai`, `cert` | malware, ICS, insider |

Then run the research with `LAUNCH.cmd` (Windows) or
`.venv/bin/python ai_engineer.py` (Linux / macOS). The registered configurations
reproduce the registered numbers (see `docs/RESULTS.md`, reproducibility report).
