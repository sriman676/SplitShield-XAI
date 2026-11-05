"""Attack knowledge base for the interactive assistant (educational / research).

Extends ``config/defense_kb.py`` (mitigations) with, for each attack the detector
knows, four things the assistant answers on demand:

  * what   - what the attack is,
  * attacker - how an attacker carries it out (mechanism),
  * howto  - how it is reproduced *inside this research sandbox* (the modules that
             already implement it, and the safe way to exercise them),
  * defend - how to defend (delegated to the curated defence KB).

Scope note (authorized NIDS research): the ``howto`` entries point at techniques
already implemented in this repository (FGSM/PGD/GAN evasion, activation inference)
and at benign, local generators. They are for understanding and reproducing results
on the project's own datasets/host, not for use against third-party systems.
"""

from __future__ import annotations

from typing import Any, cast

from config.defense_kb import defense_for

# Per detector-class mechanism + in-repo/lab reproduction. Defences come from
# config/defense_kb.py so there is one source of truth for mitigations.
_ATTACKS: dict[str, dict[str, str]] = {
    "ATTACK": {
        "attacker": "Any of the NetFlow-v2 attack families (DoS/DDoS, scanning, "
        "brute force, exploits, backdoors, worms...). The binary model flags the flow "
        "as malicious from its NetFlow features without naming the family.",
        "howto": "Studied at the flow level on NF-CSE-CIC-IDS2018-v2 / NF-UNSW-NB15-v2 "
        "(E8 cross-dataset transfer); no traffic is generated.",
    },
    "DoS": {
        "attacker": "A single host sends a flood of half-open or malformed "
        "connections (e.g. a SYN flood) so the target's connection table and CPU "
        "are exhausted and legitimate clients are refused.",
        "howto": "Reproduce locally against your own loopback service with a packet "
        "tool such as hping3/scapy issuing SYNs to a port you own. In this repo the "
        "*effect* on the detector is studied via the flow features, not by attacking "
        "a real host.",
    },
    "DDoS": {
        "attacker": "Many compromised hosts (a botnet) send traffic to one victim "
        "simultaneously; volume, not cleverness, overwhelms links or servers. Sources "
        "are often IP-spoofed and use reflection/amplification (DNS, NTP).",
        "howto": "Studied here at the flow level only. A safe lab analogue is a "
        "multi-thread request generator against your own server behind a rate limiter; "
        "never point volumetric traffic at infrastructure you do not own.",
    },
    "PortScan": {
        "attacker": "Recon: probe many ports/hosts (SYN/FIN/NULL scans) to map which "
        "services are open before choosing an exploit.",
        "howto": "Reproduce on your own subnet with `nmap -sS <your-host>`; the "
        "resulting flows carry the short-duration, many-destination signature this "
        "detector keys on.",
    },
    "BruteForce": {
        "attacker": "Repeated credential guesses against a login service (SSH/FTP/HTTP "
        "form) using wordlists until one succeeds.",
        "howto": "Reproduce against a disposable local account with a tool like "
        "hydra/patator pointed at 127.0.0.1; observe the many-short-session flow "
        "pattern. Keep it to accounts and hosts you own.",
    },
    "WebAttack": {
        "attacker": "Injects malicious input at the web layer - SQL injection, XSS, "
        "path traversal - to read data or run code the app never intended.",
        "howto": "Practise on an intentionally vulnerable app (OWASP Juice Shop / DVWA) "
        "in a local container; those are built for this. The detector sees the "
        "resulting HTTP flow anomalies.",
    },
    "WebBased": {
        "attacker": "Web-app compromise family: SQLi, XSS, command injection, malicious "
        "upload, backdoor - same idea as WebAttack across more classes.",
        "howto": "Same safe path: OWASP Juice Shop / DVWA locally. No runnable exploit "
        "is shipped here.",
    },
    "Bot": {
        "attacker": "An already-compromised host beacons to command-and-control (C2) at "
        "regular intervals and awaits instructions; the periodic beaconing is the tell.",
        "howto": "Emulate benign beaconing with a scripted fixed-interval HTTP GET to "
        "your own server to study the periodicity feature - no real malware involved.",
    },
    "Infiltration": {
        "attacker": "Post-compromise lateral movement: a foothold host pivots east-west "
        "using stolen credentials to reach higher-value systems.",
        "howto": "Study in a segmented lab (two local VMs) by moving a file over SMB/SSH "
        "between hosts you own; the detector keys on unusual internal flows.",
    },
    "Heartbleed": {
        "attacker": "Sends a malformed TLS heartbeat requesting more bytes than it "
        "supplied, so a vulnerable OpenSSL returns adjacent memory (keys, sessions).",
        "howto": "Reproduce only against a deliberately unpatched OpenSSL in an isolated "
        "VM; the flow shows an oversized heartbeat response. Patch-testing use only.",
    },
    "Mirai": {
        "attacker": "IoT worm: scans for Telnet/SSH with default creds, logs in, pulls a "
        "bot binary, then joins DDoS floods.",
        "howto": "Flow-level study only. A safe analogue is a Telnet default-cred check "
        "against your own lab IoT device to see the scan/login pattern.",
    },
    "Recon": {
        "attacker": "IoT reconnaissance: host discovery and OS/port/vuln scanning to "
        "find weak devices.",
        "howto": "`nmap`/mDNS discovery on your own IoT VLAN reproduces the burst-scan "
        "signature.",
    },
    "Spoofing": {
        "attacker": "Forges identity at L2/L3 - ARP or DNS spoofing / MITM - to "
        "intercept or redirect traffic between two parties.",
        "howto": "Reproduce with ettercap/bettercap between two VMs you control on an "
        "isolated switch; observe the ARP-reply anomalies the detector flags.",
    },
}

# Adversarial-ML techniques implemented in THIS repo (against the detector itself).
# These are the ones the assistant can both explain and actually run in-sandbox.
ADVERSARIAL: dict[str, dict[str, str]] = {
    "FGSM": {
        "what": "Fast Gradient Sign Method: a one-step evasion that nudges every "
        "mutable feature along the sign of the loss gradient so a malicious flow is "
        "misclassified as benign.",
        "attacker": "Needs gradient access (white-box) to the model; adds an "
        "epsilon-bounded perturbation constrained to features an attacker could really "
        "change (constraints in src/attack/constraints.py).",
        "howto": "Implemented at src/attack/fgsm_pgd.py:fgsm - run it on the test flows "
        "via `python run_pipeline.py --stage 4` (Stage 4 attacks).",
        "defend": "Adversarial training (src/defense/adversarial_training.py, D1) and "
        "activation noise (D2); see Stage 7.",
    },
    "PGD": {
        "what": "Projected Gradient Descent: the iterative, stronger version of FGSM - "
        "many small steps, re-projected into the epsilon ball and the feasible feature "
        "region each step.",
        "attacker": "White-box, multi-step; the headline evasion in this project.",
        "howto": "src/attack/fgsm_pgd.py:pgd, exercised by Stage 4 and evaluated for "
        "explanation drift in Stage 6 (H2).",
        "defend": "Same as FGSM; robustness is measured as ASR drop after D1.",
    },
    "GAN": {
        "what": "GAN-based evasion: a generator learns to emit malicious-yet-benign-"
        "looking flows that fool the detector while staying realistic.",
        "attacker": "Trains a generator against the (proxy) detector; higher effort, "
        "produces novel samples rather than perturbing existing ones.",
        "howto": "src/attack/gan_attack.py:gan_evasion, run in Stage 4; realism is "
        "reported jointly with attack-success-rate.",
        "defend": "Adversarial training on GAN samples; monitor feature realism bounds.",
    },
    "ACTIVATION_INFERENCE": {
        "what": "Privacy attack in split learning: an honest-but-curious server "
        "reconstructs client inputs from the activations sent across the cut layer.",
        "attacker": "Trains an inversion network on the cut activations "
        "(src/attack/activation_inference.py) - a leakage attack, not an evasion one.",
        "howto": "Stage 4 (T3) and the cut-depth study Stage 5 (H1) quantify leakage vs "
        "a mean-predictor baseline.",
        "defend": "Cut deeper (fewer client layers exposed) and add activation noise "
        "(D2, src/defense/activation_noise.py).",
    },
}


def explain(name: str) -> dict[str, Any] | None:
    """Return a structured explanation for a detector class or adversarial technique.

    Keys: ``name``, ``kind`` ('attack_class'|'adversarial'), ``what``, ``attacker``,
    ``howto``, ``defenses`` (list). Returns ``None`` if the name is unknown.
    """
    key = name.strip()
    upper = key.upper().replace(" ", "_").replace("-", "_")
    if upper in ADVERSARIAL:
        a = ADVERSARIAL[upper]
        return {
            "name": upper,
            "kind": "adversarial",
            "what": a["what"],
            "attacker": a["attacker"],
            "howto": a["howto"],
            "defenses": [a["defend"]],
        }
    # detector class: case-insensitive match against the defence KB / attack table
    for cls in {*_ATTACKS, "BENIGN"}:
        if cls.lower() == key.lower():
            kb = defense_for(cls)
            atk = _ATTACKS.get(cls, {})
            return {
                "name": cls,
                "kind": "attack_class",
                "what": str(kb["summary"]),
                "attacker": atk.get("attacker", "n/a"),
                "howto": atk.get("howto", "Studied at the flow level in this project."),
                "defenses": list(cast("list[str]", kb["defenses"])),
            }
    return None


def known_names() -> list[str]:
    """All names the assistant can explain (classes + adversarial techniques)."""
    return sorted({*_ATTACKS, "BENIGN", *ADVERSARIAL})


def format_explanation(exp: dict[str, Any]) -> str:
    """Human-readable block for the CLI."""
    lines = [
        f"=== {exp['name']}  ({exp['kind']}) ===",
        f"What it is    : {exp['what']}",
        f"How it works  : {exp['attacker']}",
        f"Reproduce     : {exp['howto']}",
        "How to defend :",
        *[f"  - {d}" for d in exp["defenses"]],
    ]
    return "\n".join(lines)
