"""Attack -> defence knowledge base (curated domain guidance).

Maps each detector class (CIC 9-family and CIC-IoT `ciot` names) to a short
explanation of the attack and concrete mitigations. This is domain knowledge, not
something to infer from the data, so it is a reviewed constant. The explanation
module pairs each entry with the model's own top SHAP features for that class, so a
report says both *why the model flagged it* and *what to do about it*.
"""

from __future__ import annotations

DEFENSE_KB: dict[str, dict[str, object]] = {
    "BENIGN": {
        "summary": "Normal traffic. No action; the baseline the others deviate from.",
        "defenses": ["No mitigation required."],
    },
    "ATTACK": {
        "summary": "Malicious flow of an unspecified type: the binary NetFlow models "
        "(E8) only separate benign from attack, so the attack family is unknown.",
        "defenses": [
            "Triage: look up the source IP's other alerts and the destination port to "
            "identify the family (scan, brute force, flood, web exploit, C2).",
            "Contain first: block or rate-limit the source at the edge while triaging.",
            "Deploy a multi-class model (CIC-IDS2017 / CIC-IoT) for a named family "
            "and its specific mitigations.",
        ],
    },
    "DoS": {
        "summary": "Denial of service: one source exhausts a target's resources.",
        "defenses": [
            "Rate-limit / SYN-cookie at the edge firewall.",
            "Per-source connection caps and timeouts.",
            "Autoscale or shed load; drop half-open connections early.",
        ],
    },
    "DDoS": {
        "summary": "Distributed DoS: many sources flood the target at once.",
        "defenses": [
            "Upstream scrubbing / CDN absorption (Anycast).",
            "BCP38 ingress filtering to stop spoofed sources.",
            "Blackhole/RTBH the victim prefix during volumetric peaks.",
        ],
    },
    "PortScan": {
        "summary": "Reconnaissance probing many ports/hosts to map the network.",
        "defenses": [
            "Port-scan detection + temporary source block.",
            "Default-deny firewall; expose only required ports.",
            "Deploy honeypots/tarpits to slow and flag scanners.",
        ],
    },
    "BruteForce": {
        "summary": "Repeated credential guessing against a login service.",
        "defenses": [
            "Account lockout / exponential backoff after failures.",
            "MFA and strong password policy.",
            "fail2ban-style IP blocking; move admin ports off defaults.",
        ],
    },
    "WebAttack": {
        "summary": "Web-layer exploitation (injection, XSS, path traversal).",
        "defenses": [
            "WAF with OWASP core ruleset.",
            "Parameterised queries and output encoding.",
            "Input validation and least-privilege DB accounts.",
        ],
    },
    "WebBased": {
        "summary": "Web-app attacks (SQLi/XSS/command injection/upload/backdoor).",
        "defenses": [
            "WAF + OWASP ruleset; block on anomaly score.",
            "Parameterised queries, CSP, and file-type/content validation.",
            "Patch app dependencies; sandbox upload handling.",
        ],
    },
    "Bot": {
        "summary": "Compromised hosts under C2 control (botnet members).",
        "defenses": [
            "Block known C2 domains/IPs; monitor beaconing periodicity.",
            "Egress filtering and DNS sinkholing.",
            "Isolate and reimage infected endpoints.",
        ],
    },
    "Infiltration": {
        "summary": "Post-compromise lateral movement inside the network.",
        "defenses": [
            "Network segmentation / micro-segmentation.",
            "EDR on endpoints; alert on unusual east-west flows.",
            "Least-privilege access and just-in-time credentials.",
        ],
    },
    "Heartbleed": {
        "summary": "OpenSSL memory-disclosure via malformed TLS heartbeat.",
        "defenses": [
            "Patch OpenSSL; revoke and reissue affected certificates.",
            "Rotate secrets that may have been in memory.",
            "IDS signature for oversized heartbeat requests.",
        ],
    },
    "Mirai": {
        "summary": "IoT botnet spreading via default credentials, then flooding.",
        "defenses": [
            "Change default IoT creds; disable Telnet/unused services.",
            "Segment IoT onto its own VLAN with egress filtering.",
            "Firmware updates; block known Mirai C2 and scan patterns.",
        ],
    },
    "Recon": {
        "summary": "IoT reconnaissance: host discovery, OS/port/vuln scanning.",
        "defenses": [
            "Default-deny + minimal exposed services on IoT segments.",
            "Detect and rate-limit scan bursts; honeypots.",
            "Disable mDNS/UPnP discovery where not needed.",
        ],
    },
    "Spoofing": {
        "summary": "Identity spoofing (ARP/DNS/MITM) to intercept or redirect.",
        "defenses": [
            "Dynamic ARP Inspection + DHCP snooping on switches.",
            "DNSSEC / trusted resolvers; static ARP for critical hosts.",
            "802.1X port auth to bind identity to the port.",
        ],
    },
}


def defense_for(cls: str) -> dict[str, object]:
    """Return the KB entry for a class, or a safe unknown-class placeholder."""
    return DEFENSE_KB.get(
        cls,
        {
            "summary": f"Unmapped class {cls!r}; add it to config/defense_kb.py.",
            "defenses": ["No curated guidance yet — review and add."],
        },
    )
