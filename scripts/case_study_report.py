"""Build results/SplitShield-XAI_case_study.docx: the academic case study (2026-10-29).

python -m scripts.case_study_report
"""

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "SplitShield-XAI_case_study.docx"
ACCENT = RGBColor(0x1F, 0x3A, 0x5F)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.3))
sec.top_margin = sec.bottom_margin = Cm(2.2)

st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15
for lvl, size in ((1, 15), (2, 12.5)):
    h = doc.styles[f"Heading {lvl}"]
    h.font.name, h.font.size, h.font.color.rgb = "Calibri", Pt(size), ACCENT
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
    h.paragraph_format.space_after = Pt(4)

# page numbers in the footer
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
    r = fp.add_run()
    if kind:
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), kind)
    else:
        el = OxmlElement("w:instrText")
        el.set(qn("xml:space"), "preserve")
        el.text = text
    r._r.append(el)
    r.font.size = Pt(9)


def para(
    text: str,
    bold_lead: str = "",
    italic: bool = False,
    size: float | None = None,
    align=None,
) -> None:
    p = doc.add_paragraph()
    if bold_lead:
        p.add_run(bold_lead).bold = True
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if align is not None:
        p.alignment = align


def bullets(items: list[str]) -> None:
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        if "::" in it:
            lead, rest = it.split("::", 1)
            p.add_run(lead if lead.endswith(".") else lead + ":").bold = True
            p.add_run(rest)
        else:
            p.add_run(it)
        p.paragraph_format.space_after = Pt(2)


def shade(cell, hex_fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def table(rows: list[list[str]], widths_cm: list[float], caption: str = "") -> None:
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Cm(widths_cm[j])
            c.text = ""
            r = c.paragraphs[0].add_run(val)
            r.font.size = Pt(9)
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
            if i == 0:
                r.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                shade(c, "1F3A5F")
            elif i % 2 == 0:
                shade(c, "EEF2F7")
    if caption:
        p = doc.add_paragraph()
        r = p.add_run(caption)
        r.italic, r.font.size = True, Pt(9)
        p.paragraph_format.space_before = Pt(3)


def figure(path: Path, width_cm: float, caption: str) -> None:
    if not path.exists():
        return
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    r = p.add_run(caption)
    r.italic, r.font.size = True, Pt(9)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


# ---- title page -------------------------------------------------------------------
for _ in range(5):
    doc.add_paragraph()
para("CASE STUDY", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("SplitShield-XAI")
r.bold, r.font.size, r.font.color.rgb = True, Pt(30), ACCENT
para(
    "Building an explainable detection layer next to a SIEM, and testing the security "
    "of split learning on the way",
    size=14,
    align=WD_ALIGN_PARAGRAPH.CENTER,
)
doc.add_paragraph()
para("Sriman Rutvik Voddiraju", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
para("August – October 2026", size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(6):
    doc.add_paragraph()
para(
    "Companion documents: SplitShield-XAI_final_report.docx (full technical report), "
    "docs/RESEARCH.md, docs/LIMITATIONS.md, docs/MODEL_CARDS.md.",
    italic=True,
    size=9,
    align=WD_ALIGN_PARAGRAPH.CENTER,
)
doc.paragraphs[-1].runs[0].add_break(WD_BREAK.PAGE)

# ---- abstract -----------------------------------------------------------------------
doc.add_heading("Abstract", 1)
para(
    "Security teams collect large volumes of logs in a SIEM but still struggle to tell which "
    "events belong to one attack, why an alert fired, and whether a machine-learning detector "
    "can be trusted on their own network. SplitShield-XAI is a project with two halves that "
    "answer these questions together. The research half asks whether split learning, a way to "
    "train intrusion detectors without pooling raw data, leaks information, how explanations "
    "behave under attack, and how training schemes cope with uneven data (hypotheses H1–H5, "
    "five seeds each, every metric paired with a control). The product half is a working "
    "detection and response companion for a SIEM: nine models and more than 3,600 rules cover "
    "network flows, Windows, Linux, web, cloud, phishing, malware, industrial control and "
    "insider activity, link findings into incidents along the MITRE ATT&CK chain, explain them "
    "in plain English and forward them to any SIEM."
)
para(
    "Four of the five registered hypotheses were supported. The fifth (H2, explanation drift "
    "under attack) was not supported as registered, turned out to be affected by a "
    "normalisation artefact, failed a first confirmatory test for lack of statistical power, "
    "and was finally supported by a pre-registered, adequately powered test (H2c, 5 of 5 "
    "seeds) with a small effect. On the product side, held-out results are strong where the "
    "training and test data come from similar environments (99.7 % of network attacks caught "
    "at 0.17 % false alarms; malware 92.6 % at 1 % false positives) and weaker on unseen "
    "networks and log sources, which two rounds of blind testing exposed and which are reported "
    "as measured. The main lesson of the case is methodological: pre-registration, controls, "
    "held-out and blind data, and a strict rule that a model change is kept only if precision "
    "and recall both hold, caught several errors that would otherwise have been reported as "
    "results."
)

# ---- 1 context ----------------------------------------------------------------------
doc.add_heading("1. Context and problem", 1)
para(
    "A Security Information and Event Management system (SIEM) stores and searches logs from "
    "many sources. Detection inside a SIEM is mostly rule-based, alerts arrive one by one, and "
    "an analyst has to decide which alerts belong together and what the attacker is doing. "
    "Machine-learning intrusion detectors promise to find what rules miss, but three problems "
    "limit their use in practice:"
)
bullets(
    [
        "Trust:: a score without a reason is hard to act on, and explanations themselves may be "
        "manipulable.",
        "Transfer:: detectors trained on public datasets often fail on a different network, and "
        "published accuracy figures rarely show this.",
        "Data sharing:: organisations cannot pool raw traffic to train a better shared model. "
        "Split learning and federated learning avoid pooling, but their security and their "
        "behaviour on uneven (non-IID) data need evidence.",
    ]
)
para(
    "The project set out to build a tool that a small security team could actually run next to "
    "its SIEM, and to use the same data and code to answer research questions about split "
    "learning, explanations and attacks."
)

# ---- 2 objectives -------------------------------------------------------------------
doc.add_heading("2. Objectives and research questions", 1)
para("Product objectives:", bold_lead="")
bullets(
    [
        "Detect attacks across network, endpoint, web, cloud, e-mail/URL, file, industrial and "
        "insider data, and map each finding to MITRE ATT&CK.",
        "Correlate findings from different sensors into incidents and explain each one in plain "
        "English.",
        "Connect to any SIEM (CEF, LEEF, ECS, OCSF; syslog, TLS or webhook) with guaranteed delivery.",
        "Stay current without code changes (update agent for Sigma, ATT&CK, OWASP) and let users "
        "change every rule, with proposals they approve.",
    ]
)
para("Research hypotheses (registered before the main runs):")
table(
    [
        ["", "Prediction"],
        [
            "H1",
            "Activation leakage falls as the split cut moves deeper; accuracy stays roughly flat.",
        ],
        [
            "H2",
            "Adversarial (PGD) inputs change SHAP explanations more than random noise of the same size.",
        ],
        [
            "H3",
            "Adversarial training cuts attack success at a bounded cost in clean accuracy.",
        ],
        [
            "H4",
            "Sequential split learning degrades fastest under label skew; SplitFed does not.",
        ],
        [
            "H5",
            "The original CIC-IDS2017 release scores higher than its corrected release.",
        ],
    ],
    [1.5, 14.9],
    "Table 1. Registered research hypotheses.",
)

# ---- 3 approach ---------------------------------------------------------------------
doc.add_heading("3. Approach", 1)
doc.add_heading("3.1 Research pipeline", 2)
para(
    "A ten-stage pipeline (run_pipeline.py) preprocesses each dataset (deduplication, invalid "
    "values, class-capped sampling, 70/15/15 split, scaling), trains baselines (random forest, "
    "gradient boosting, MLP) and four training schemes (centralised, sequential split learning, "
    "FedAvg, SplitFed), explains them with SHAP (LIME as a cross-check), attacks them (FGSM, "
    "PGD, GAN, activation inversion, client attribution), and runs the ablations behind H1–H5. "
    "Rules are enforced in code rather than by convention: a leakage, attack-success or drift "
    "metric raises an error without its control; runs whose configurations differ outside the "
    "declared variables are never put in the same table; model selection uses validation data "
    "only; headline numbers use five seeds."
)
doc.add_heading("3.2 Product architecture", 2)
para(
    "Inputs (local logs every 2 s, agents on other machines over pinned HTTPS, network flows, "
    "uploads) pass through rule engines (the full SigmaHQ bundle of 3,161 rules, OWASP Core "
    "Rule Set, 352 LOLBAS/GTFOBins patterns, own rules) and nine models (flows, NetFlow, Zeek, "
    "phishing URLs and pages, malware, industrial control, insider, question understanding, "
    "and ShieldLM, a small transformer that words the assistant's answers). Every detector "
    "posts what it finds about an IP address, user, host, URL or file to a shared risk board. "
    "Correlation groups an entity's events into one incident, raises its severity when "
    "independent sensors agree, and lays out the ATT&CK path. Incidents go to the SIEM and to "
    "a dashboard with an assistant that answers questions from the stored data."
)
figure(
    ROOT / "docs" / "architecture.png",
    15.5,
    "Figure 1. How the parts connect: sources, detectors, risk board, correlation and outputs.",
)
doc.add_heading("3.3 Evaluation design", 2)
bullets(
    [
        "Held-out data:: never used to build or tune a model; reported once.",
        "Blind data:: separate datasets scored once at the end. Round 1 (CIC-IDS2018, EMBER 2017, "
        "HAI 21.03, CERT r4.1, OpenPhish, Splunk attack_data) and, after the improvements, round 2 "
        "on new sets (HIKARI, AIT-LDS v2, EVTX-to-MITRE, CompPhish), frozen in a tagged copy "
        "before scoring.",
        "Keep-or-revert rule:: a model change is deployed only if held-out precision and recall "
        "both do not drop. Exceptions were decided by the project owner and recorded as such.",
        "Fair rule tests:: rules were written from a development split only; Sigma rules written "
        "by a test set's authors or citing it were excluded.",
        "Engineering tests:: unit tests, every combination of the options run end to end, a "
        "30-minute soak test with a SIEM outage, hostile-input, concurrency and fault tests, a "
        "real Linux host, and a penetration test.",
    ]
)

# ---- 4 results ----------------------------------------------------------------------
doc.add_heading("4. Results", 1)
doc.add_heading("4.1 Research", 2)
table(
    [
        ["", "Verdict", "Evidence"],
        [
            "H1",
            "Supported",
            "Leakage 0.981 → 0.958 → 0.911 for cut depth 1→3 (Spearman ρ = −0.64, "
            "BH-adjusted p = 0.020); macro-F1 0.85–0.87",
        ],
        [
            "H2",
            "Not supported (0/5)",
            "PGD drift 0.865 vs random-matched control 0.967",
        ],
        [
            "H2b",
            "Not supported (2/5)",
            "Pre-registered confirmatory test, corrected pipeline, 16 instances "
            "per seed (≈ 35 % power)",
        ],
        [
            "H2c",
            "Supported (5/5)",
            "Pre-registered with a power analysis, 128 instances per seed: every "
            "seed p < 0.0001; median paired difference +0.020 … +0.051 (small)",
        ],
        [
            "H3",
            "Supported (3/5)",
            "PGD success → 0 in every seed; clean-F1 change −0.014 … +0.095",
        ],
        [
            "H4",
            "Supported",
            "Macro-F1 drop from α = 10 to α = 0.1: split-sequential −0.083, SplitFed −0.007",
        ],
        ["H5", "Supported", "MLP 0.875 vs 0.853; RF 0.962 vs 0.956"],
    ],
    [1.4, 3.4, 11.6],
    "Table 2. Research verdicts (five seeds per hypothesis, controls applied).",
)
figure(
    ROOT / "results" / "figures" / "F4_alpha_skew.png",
    12.5,
    "Figure 2. H4: macro-F1 of each training scheme as label skew grows (smaller α = more skew).",
)

doc.add_heading("4.2 Detection", 2)
table(
    [
        ["Area", "Held-out", "Blind round 1", "Blind round 2"],
        [
            "Network flows",
            "99.7 % caught, 0.17 % false alarms",
            "CIC-IDS2018: 56.1 %, 3.85 % FA",
            "HIKARI: 13.2 %, 14.0 % FA",
        ],
        [
            "Windows events",
            "53.9 % right ATT&CK tactic",
            "—",
            "EVTX-to-MITRE: 55.2 % detected, 42.7 % right tactic",
        ],
        [
            "Linux / cloud / identity logs",
            "40.9 % right tactic",
            "43.8 % detected, 31.9 % right tactic",
            "AIT-LDS v2: 5 of 22 attack steps",
        ],
        [
            "Phishing URLs",
            "precision 0.997, recall 0.60",
            "precision 1.00, recall 0.74",
            "CompPhish: recall 0.71, 2.0 % FA",
        ],
        ["Malware files", "92.6 % at 1 % false positives", "EMBER 2017: 95.0 %", "—"],
        [
            "Industrial control",
            "precision 0.90 / 0.70",
            "HAI 21.03: does not transfer",
            "—",
        ],
        [
            "Insider threat",
            "F1 0.95, 0.86 alerts/day",
            "3 of 3 insiders, 0.05 alerts/day",
            "—",
        ],
    ],
    [3.4, 4.3, 4.3, 4.4],
    "Table 3. Detection results. FA = false alarms. Nothing was tuned on a blind set.",
)
para(
    "Engineering results: more than 740 automated tests; all 41,483 option combinations run "
    "end to end; in the soak test 360,100 of 360,100 log lines were stored exactly once and "
    "35,963 of 35,963 alerts delivered exactly once despite a SIEM outage and a receiver "
    "restart; about 3 ms of processing from a log line to the SIEM; log analysis at about "
    "140,000 lines per second with 2–6 workers and about 190,000 with 8 on the development PC."
)

# ---- 5 key episodes -----------------------------------------------------------------
doc.add_heading("5. Key episodes", 1)
para(
    "The most instructive parts of the project were the points where a first result turned out "
    "to be wrong or incomplete. Five are summarised here."
)

doc.add_heading('5.1 H2: from "not supported" to a powered test', 2)
para(
    "The registered H2 runs showed explanations drifting almost completely (about 0.9–1.0) under "
    "both PGD and random noise, so the hypothesis was not supported. Investigation found that "
    "derived features were recomputed on data that had already been normalised, which scrambled "
    "every perturbed explanation. On a corrected pipeline an exploratory follow-up suggested a "
    "small effect. Rather than replace the registered verdict, a confirmatory test (H2b) was "
    "pre-registered on new seeds: it failed (2 of 5). A power analysis then showed H2b had only "
    "about 35 % power at 16 instances per seed. H2c repeated the same test and decision rule on "
    "new seeds with 128 instances per seed (power ≈ 0.98) and was supported in 5 of 5 seeds. All "
    "three verdicts are reported side by side. The conclusion is that PGD does move SHAP "
    "explanations more than equal-size noise, but only by a few hundredths on a 0–1 scale."
)

doc.add_heading('5.2 The risk board that "did not help"', 2)
para(
    "The first validation of the shared risk board showed no measurable gain from linking "
    "findings across sensors. The cause was a bug: the validation script scored an empty board. "
    "Fixing it uncovered further problems (year-less syslog dated in the wrong year, which lost "
    "29 February; events not recording which machine logged them; loopback addresses treated as "
    "entities; failed logons linking unrelated entities). After the fixes, settings chosen on "
    "development servers and checked on held-out servers kept entity precision at 1.00 and "
    "raised recall from 0.23 to 0.37."
)

doc.add_heading("5.3 Blind round 2 and transfer between networks", 2)
para(
    "After improvements that raised recall on other networks (CIC-IDS2018 0.37 → 0.85, "
    "IoT-DIAD 0.14 → 0.84), the flow detector caught only 13.2 % of attacks on the blind HIKARI "
    "dataset, at 14.0 % false alarms. This was reported as measured. Afterwards, and labelled as "
    "not blind, training on 70 % of HIKARI gave 19.4 % at 0.009 % false alarms on the remaining "
    "30 %, and per-site calibration was added so that each site can adapt the threshold to its "
    "own traffic. The lesson is that flow detectors must be calibrated or retrained on the site "
    "where they run, and that a single held-out split from the training network overstates "
    "real-world performance."
)

doc.add_heading("5.4 A data leak in the phishing data", 2)
para(
    "The legitimate URLs in the phishing training set were almost all homepages, so a model "
    'could learn "has a path → phishing". Version 4 trains on 300,000 real popular domains '
    "(Tranco ranks 200,001–1,000,000) and reads only the hostname. Precision rose from 0.9941 to "
    "0.9959, and popular real sites flagged fell from 1.84 % to 0.02 %. The remaining gap, "
    "phishing hosted on trusted shared platforms, is handled by threat-intelligence feeds and by "
    "a separate page model for uploaded pages and e-mail attachments."
)

doc.add_heading("5.5 Speed: measuring before optimising", 2)
para(
    "Log analysis did not get faster beyond two worker processes. The first assumption, that "
    "the SQLite store was the bottleneck, was wrong. Profiling showed parsing was CPU-bound and "
    "that the OWASP Core Rule Set took 62 % of the time, because every rule decoded the same "
    "request values again. Sharing decoded values between the rules of one request gave a "
    "1.64× speed-up with identical results on 12,510 real requests. Together with workers "
    "storing their own share, throughput rose from about 64,000 to about 140,000–190,000 lines "
    "per second."
)

# ---- 6 limitations ------------------------------------------------------------------
doc.add_heading("6. Limitations", 1)
bullets(
    [
        "Transfer:: flow and industrial-control detectors do not fully transfer to unseen networks "
        "or plants. Each site needs calibration, and an industrial plant needs about a week of its "
        "own normal data.",
        "Evasion:: after adversarial training, a white-box PGD attacker can still push about 29–31 % "
        "of detected network attacks past the detector (down from 41–62 %).",
        "Log coverage:: on unseen log sources only part of each attack chain is recognised (for "
        "example 5 of 22 steps on the blind AIT-LDS v2 servers before new rules were written). Some "
        "tactics need endpoint telemetry such as Sysmon.",
        "Data realism:: the insider data (CERT) is synthetic, malware detection on live samples was "
        "not measured (live malware is never downloaded), and macOS was tested only by unit tests.",
        "Research scope:: H2's effect is small and dataset-dependent (it did not replicate on raw-unit "
        "CIC-IDS2017). Registered numbers on the pre-normalised CIC-IDS2017 release are approximate "
        "about attack realism.",
    ]
)

# ---- 7 lessons ----------------------------------------------------------------------
doc.add_heading("7. Lessons learned", 1)
bullets(
    [
        "Pre-register, then power the test.:: The H2 story shows that a confirmatory test can fail "
        "simply because it is too small; a power analysis is part of the design, not an "
        "afterthought.",
        "Keep the registered result and add the correction next to it.:: Replacing a verdict after "
        "seeing the data would have hidden two real errors (the normalisation artefact and the "
        "underpowered test).",
        "Blind data is worth the cost.:: Every held-out number looked good; the blind rounds were "
        "where transfer failures appeared.",
        '"No effect" can be a bug.:: The risk board and several other surprises were pipeline '
        "errors, found because a null result was investigated rather than accepted.",
        "Profile before optimising.:: The obvious suspect (the database) was not the bottleneck.",
        "A precision-and-recall rule keeps deployment honest.:: Several improvements that raised one "
        "metric and lowered the other were rejected; the exceptions were decided by the owner and are "
        "documented.",
    ]
)

# ---- 8 conclusion -------------------------------------------------------------------
doc.add_heading("8. Conclusion and future work", 1)
para(
    "SplitShield-XAI shows that one codebase can serve both a careful research study and a "
    "usable security tool. The research supports four of five hypotheses outright and gives a "
    "properly powered, small-effect answer to the fifth. The product detects and explains "
    "attacks across many data sources and connects to any SIEM, and its weak spots are measured "
    "and published rather than hidden. The next steps are a new blind check for industrial "
    "control on SWaT/WADI (requested from iTrust), live Zeek traffic, real (non-synthetic) "
    "insider and macOS data, and site-specific retraining of the flow detectors."
)

doc.add_heading("Note on the development process", 2)
para(
    "The project was built by the author with an AI coding assistant (Claude Code). The "
    "assistant wrote and ran code under rules set by the author (pre-registration, no tuning on "
    "blind data, the precision-and-recall rule, no deletion or download without approval); the "
    "author decided every research and deployment question that the rules did not settle.",
    italic=True,
)

# ---- references ---------------------------------------------------------------------
doc.add_heading("References", 1)
refs = [
    "Sharafaldin, I., Habibi Lashkari, A., Ghorbani, A. A. Toward generating a new intrusion "
    "detection dataset and intrusion traffic characterization. ICISSP 2018. (CIC-IDS2017, "
    "CSE-CIC-IDS2018)",
    "Sharafaldin, I., Habibi Lashkari, A., Hakak, S., Ghorbani, A. A. Developing realistic "
    "distributed denial of service (DDoS) attack dataset and taxonomy. ICCST 2019. (CIC-DDoS2019)",
    "Neto, E. C. P., et al. CICIoT2023: a real-time dataset and benchmark for large-scale attacks "
    "in IoT environment. Sensors, 2023.",
    "Sarhan, M., Layeghy, S., Portmann, M. Towards a standard feature set for network intrusion "
    "detection system datasets. Mobile Networks and Applications, 2022. (NF-v2 NetFlow sets)",
    "Moustafa, N., Slay, J. UNSW-NB15. 2015; Moustafa, N. ToN-IoT. 2021.",
    "Landauer, M., et al. AIT Log Data Set v1.1. AIT Austrian Institute of Technology. "
    "zenodo.org/records/4264796",
    "Zhu, J., et al. Loghub: a large collection of system log datasets. ISSRE 2023.",
    "Prasad, A., Chandra, S. PhiUSIIL: a diverse security profile empowered phishing URL "
    "detection framework. Computers & Security, 2024.",
    "Le Pochat, V., et al. Tranco: a research-oriented top sites ranking hardened against "
    "manipulation. NDSS 2019.",
    "Anderson, H. S., Roth, P. EMBER: an open dataset for training static PE malware machine "
    "learning models. arXiv:1804.04637, 2018.",
    "Shin, H.-K., et al. HAI security dataset. National Security Research Institute. "
    "github.com/icsdataset/hai",
    "Lindauer, B. Insider Threat Test Dataset. Carnegie Mellon University, SEI, 2020.",
    "MITRE ATT&CK®. The MITRE Corporation. attack.mitre.org",
    "SigmaHQ detection rules (Detection Rule License 1.1); OWASP Core Rule Set (Apache-2.0); "
    "OWASP Top 10 (CC BY-SA 4.0).",
]
for i, ref in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.first_line_indent = Cm(-0.8)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(f"[{i}]  {ref}")
    r.font.size = Pt(9.5)

doc.save(str(OUT))
print(OUT)
