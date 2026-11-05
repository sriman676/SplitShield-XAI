"""Detailed, defender-oriented attack knowledge for the assistant (owner 2026-10-14).

For every attack family the detector reports: its main TYPES, HOW IT UNFOLDS (the
phases an analyst should recognise - descriptive, no runnable instructions), the
SIGNS it leaves in flows and logs, and for every MITRE ATT&CK tactic: what it means
and which techniques this SOC can actually see. Answers are built from this text,
so facts and ATT&CK ids are exact; the language model only words incident
narratives.
"""

from __future__ import annotations

# family -> {"types": [(name, description)], "steps": [...], "signs": [...]}
DETAILS: dict[str, dict[str, list]] = {
    "BruteForce": {
        "types": [
            (
                "Simple brute force",
                "tries every combination of characters; slow, only works on short passwords",
            ),
            (
                "Dictionary attack",
                "tries words from a list of common or leaked passwords",
            ),
            (
                "Credential stuffing",
                "replays username/password pairs leaked from other breaches, betting on password reuse",
            ),
            (
                "Password spraying",
                "tries one or a few common passwords against many accounts, staying under lockout limits",
            ),
            (
                "Hybrid attack",
                "dictionary words plus common changes (Summer2024!, p@ssw0rd)",
            ),
            (
                "Offline cracking",
                "guesses against stolen password hashes on the attacker's own machine - invisible to network monitoring until the cracked password is used",
            ),
        ],
        "steps": [
            "The attacker finds an exposed login service (SSH, RDP, FTP, VPN, a web or mail login).",
            "They collect likely usernames (default accounts, staff names, leaked lists).",
            "An automated tool submits login attempts at high speed, sometimes from many IP addresses to dodge rate limits and lockouts.",
            "When a guess succeeds, the attacker logs in with a valid account (ATT&CK T1078) and continues the intrusion.",
        ],
        "signs": [
            "Many failed logons in a short time (Windows 4625/4771, SSH 'Failed password', FTP/HTTP 401s).",
            "One source trying many accounts (spraying) or many attempts on one account; account lockouts (Windows 4740).",
            "Many short, repeated connections to ports 22, 3389, 21 or a login URL.",
            "A successful logon right after a burst of failures from the same IP - treat as a likely compromised account.",
        ],
    },
    "DoS": {
        "types": [
            (
                "Volumetric flood",
                "sends more traffic (UDP, ICMP) than the link or server can absorb",
            ),
            (
                "Protocol / SYN flood",
                "opens connections without finishing the TCP handshake until the server's connection table is full",
            ),
            (
                "Application-layer flood",
                "sends many expensive requests (e.g. HTTP floods such as Hulk or GoldenEye) that exhaust CPU or worker threads",
            ),
            (
                "Slow-rate attack",
                "holds connections open by sending data very slowly (Slowloris, Slow HTTP) so legitimate users cannot connect",
            ),
        ],
        "steps": [
            "The attacker picks a target service and the resource to exhaust (bandwidth, connection table, CPU, worker threads).",
            "From one machine they send a flood of packets or requests - or many deliberately slow connections.",
            "The service slows down, times out or stops answering legitimate users.",
        ],
        "signs": [
            "A sudden spike of flows or packets from a single source to one service.",
            "Many half-open (SYN without ACK) or very long, almost idle connections.",
            "Server errors and timeouts; the detector labels the flows DoS.",
        ],
    },
    "DDoS": {
        "types": [
            ("Botnet flood", "thousands of infected devices flood the target at once"),
            (
                "Reflection / amplification",
                "spoofed requests to open DNS, NTP or memcached servers make them send large replies to the victim",
            ),
            (
                "Application-layer DDoS",
                "many sources send legitimate-looking requests (e.g. LOIC HTTP floods)",
            ),
        ],
        "steps": [
            "The attacker controls many machines (a botnet) or abuses open reflectors.",
            "All sources are told to send traffic to the victim at the same time, often with spoofed addresses.",
            "The combined volume overwhelms the victim's link or servers; blocking one IP does not help.",
        ],
        "signs": [
            "A traffic surge from many different IP addresses with the same pattern.",
            "Large UDP replies from ports 53 (DNS), 123 (NTP) or 11211 (memcached) that nobody requested.",
            "Upstream link saturation; the detector labels the flows DDoS.",
        ],
    },
    "PortScan": {
        "types": [
            ("Host discovery (ping sweep)", "finds which addresses are alive"),
            (
                "TCP SYN (half-open) scan",
                "sends SYNs and reads SYN-ACK/RST answers without completing connections",
            ),
            ("TCP connect scan", "completes full connections to each port"),
            ("UDP scan", "probes UDP services such as DNS and SNMP"),
            (
                "Stealth scans (FIN, NULL, Xmas)",
                "use unusual TCP flags to slip past simple filters",
            ),
            (
                "Service / version detection",
                "fingerprints what software runs on each open port",
            ),
        ],
        "steps": [
            "The attacker discovers which hosts are online.",
            "They probe many ports on those hosts to learn which services are open.",
            "They fingerprint the services and versions to choose an exploit for the next stage.",
        ],
        "signs": [
            "One source contacting many ports and/or hosts within seconds.",
            "Many very short flows with one or two packets, lots of RSTs or firewall drops.",
            "Suricata 'scan' signatures; the correlator adds ATT&CK T1046 when a source touches many host:port pairs.",
        ],
    },
    "Recon": {
        "types": [
            ("Ping sweep", "finds live hosts"),
            ("Port scan", "finds open services"),
            ("OS scan", "guesses the operating system from packet behaviour"),
            ("Vulnerability scan", "checks services against known weaknesses"),
        ],
        "steps": [
            "The attacker maps the network: live hosts, open ports, operating systems.",
            "They check the discovered services for known vulnerabilities.",
            "The results decide which devices are attacked next.",
        ],
        "signs": [
            "One source touching many hosts/ports quickly; unusual probe packets.",
            "Vulnerability-scanner signatures in Suricata; ATT&CK T1595 / T1046.",
        ],
    },
    "WebAttack": {
        "types": [
            (
                "SQL injection",
                "smuggles database commands through a form or URL parameter to read or change data",
            ),
            (
                "Cross-site scripting (XSS)",
                "plants script in a page so it runs in other users' browsers",
            ),
            ("Command injection", "makes the server run operating-system commands"),
            ("Path traversal", "uses ../ sequences to read files outside the web root"),
            ("Web login brute force", "automated password guessing against a web form"),
        ],
        "steps": [
            "The attacker maps the application: pages, forms, parameters, APIs.",
            "They send crafted input to find a field the application does not validate.",
            "They exploit it to read data, take over sessions or run code on the server.",
        ],
        "signs": [
            "Odd characters in URLs or form data (quotes, OR 1=1, <script>, ../).",
            "Bursts of HTTP 4xx/5xx errors from one client; WAF or Suricata web-attack alerts.",
            "Unexpected database errors in application logs; ATT&CK T1190.",
        ],
    },
    "WebBased": {
        "types": [
            ("SQL injection", "database commands through user input"),
            ("Cross-site scripting", "script injected into pages"),
            ("Command injection", "operating-system commands through the app"),
            (
                "Malicious upload / backdoor",
                "an uploaded file gives the attacker lasting access",
            ),
            ("Browser hijacking", "the victim's browser is redirected or controlled"),
        ],
        "steps": [
            "The attacker probes the web application of the device or service.",
            "They inject malicious input or upload a malicious file.",
            "They use the foothold to steal data or keep access.",
        ],
        "signs": [
            "Malformed or script-laden HTTP requests; new unexpected files on the web server.",
            "ATT&CK T1190 alerts from Suricata or the detector.",
        ],
    },
    "Bot": {
        "types": [
            ("Centralised C2 (HTTP/IRC)", "bots poll one command server"),
            (
                "Peer-to-peer botnet",
                "bots relay commands to each other, no single server to block",
            ),
            (
                "Domain-generation (DGA) botnet",
                "bots compute new domain names daily to find their controller",
            ),
        ],
        "steps": [
            "A host is infected (phishing, exploit, weak password).",
            "The bot 'beacons' to its command-and-control server at regular intervals.",
            "It receives tasks: flood a target, send spam, steal data or download more malware.",
        ],
        "signs": [
            "Regular, periodic small connections from an internal host to the same external address.",
            "DNS lookups of random-looking domains; ATT&CK T1071 (application-layer C2).",
        ],
    },
    "Mirai": {
        "types": [
            ("GRE floods (GRE-ETH, GRE-IP)", "floods wrapped in GRE tunnels"),
            ("UDP / SYN / ACK floods", "classic volumetric floods from IoT devices"),
            ("HTTP flood", "request floods from infected cameras and routers"),
        ],
        "steps": [
            "Infected devices scan the internet for IoT devices with Telnet (ports 23/2323) open.",
            "They log in with factory-default passwords and install the bot.",
            "The new bot reports to C2 and later joins DDoS floods on command.",
        ],
        "signs": [
            "Telnet scanning and logins with default credentials; IoT devices making outbound floods.",
            "ATT&CK T1071 (C2) and T1498 (network denial of service).",
        ],
    },
    "Infiltration": {
        "types": [
            ("Malicious attachment", "the victim opens a booby-trapped document"),
            ("Drive-by download", "visiting a compromised site installs malware"),
            ("Infected removable media", "malware runs from a USB stick"),
        ],
        "steps": [
            "A user runs or opens something malicious, giving the attacker a foothold inside the network.",
            "The attacker downloads tools onto that machine (ATT&CK T1105).",
            "From inside, they scan and move towards more valuable systems.",
        ],
        "signs": [
            "An internal host suddenly downloading tools or contacting unusual external servers.",
            "Internal scanning that starts from a workstation; new processes or services on that host.",
        ],
    },
    "Heartbleed": {
        "types": [
            (
                "OpenSSL heartbeat over-read (CVE-2014-0160)",
                "malformed TLS heartbeat requests make a vulnerable server return up to 64 KB of its memory per request",
            ),
        ],
        "steps": [
            "The attacker opens a TLS connection to a server running a vulnerable OpenSSL version.",
            "They send heartbeat requests that claim a larger payload than they send.",
            "The server leaks memory - possibly private keys, session cookies or passwords.",
        ],
        "signs": [
            "Many TLS heartbeat messages with oversized length fields; large server responses to tiny requests.",
            "IDS signatures for CVE-2014-0160; ATT&CK T1190.",
        ],
    },
    "Spoofing": {
        "types": [
            (
                "ARP spoofing",
                "the attacker claims another device's IP address on the LAN to intercept its traffic",
            ),
            (
                "DNS spoofing",
                "forged DNS answers send users to attacker-controlled servers",
            ),
            ("IP spoofing", "forged source addresses hide the real sender"),
        ],
        "steps": [
            "The attacker sends forged ARP or DNS replies on the local network.",
            "Victims' traffic now flows through the attacker (man-in-the-middle).",
            "The attacker reads or alters the traffic and captures credentials.",
        ],
        "signs": [
            "One MAC address claiming several IPs; bursts of unsolicited ARP replies.",
            "DNS answers from unexpected servers; ATT&CK T1557 (adversary-in-the-middle).",
        ],
    },
}

# detector class -> plain-English name used in answers
DISPLAY: dict[str, str] = {
    "BruteForce": "brute force",
    "PortScan": "port scanning",
    "Recon": "reconnaissance scanning",
    "WebAttack": "web application attacks",
    "WebBased": "web-based attacks",
    "DoS": "denial of service (DoS)",
    "DDoS": "distributed denial of service (DDoS)",
    "Bot": "botnet activity",
    "Mirai": "Mirai botnet activity",
    "Infiltration": "infiltration",
    "Heartbleed": "Heartbleed exploitation",
    "Spoofing": "spoofing",
    "ATTACK": "malicious traffic",
}


def display(name: str) -> str:
    return DISPLAY.get(name, name)


# ATT&CK tactic slug -> id, display name, meaning
TACTICS: dict[str, dict[str, str]] = {
    "reconnaissance": {
        "id": "TA0043",
        "name": "Reconnaissance",
        "what": "gathering information about the target (scanning, probing) to plan an attack",
    },
    "resource-development": {
        "id": "TA0042",
        "name": "Resource Development",
        "what": "acquiring infrastructure, accounts or tools for the attack",
    },
    "initial-access": {
        "id": "TA0001",
        "name": "Initial Access",
        "what": "getting a first foothold - exploiting a public-facing app, phishing, valid stolen accounts",
    },
    "execution": {
        "id": "TA0002",
        "name": "Execution",
        "what": "running attacker code - scripts, command interpreters, malicious binaries",
    },
    "persistence": {
        "id": "TA0003",
        "name": "Persistence",
        "what": "keeping access across reboots - new accounts, services, scheduled tasks",
    },
    "privilege-escalation": {
        "id": "TA0004",
        "name": "Privilege Escalation",
        "what": "gaining higher permissions such as administrator or SYSTEM",
    },
    "defense-evasion": {
        "id": "TA0005",
        "name": "Defense Evasion",
        "what": "avoiding detection - clearing logs, abusing trusted system binaries, disabling security tools",
    },
    "credential-access": {
        "id": "TA0006",
        "name": "Credential Access",
        "what": "stealing or guessing passwords and keys - brute force, credential dumping, man-in-the-middle",
    },
    "discovery": {
        "id": "TA0007",
        "name": "Discovery",
        "what": "learning the internal environment - network service and account discovery",
    },
    "lateral-movement": {
        "id": "TA0008",
        "name": "Lateral Movement",
        "what": "moving from the first host to others - remote services, pass-the-hash",
    },
    "collection": {
        "id": "TA0009",
        "name": "Collection",
        "what": "gathering the data the attacker wants",
    },
    "command-and-control": {
        "id": "TA0011",
        "name": "Command and Control",
        "what": "remote control of compromised hosts - beaconing, tool downloads",
    },
    "exfiltration": {
        "id": "TA0010",
        "name": "Exfiltration",
        "what": "stealing data out of the network",
    },
    "impact": {
        "id": "TA0040",
        "name": "Impact",
        "what": "disrupting or destroying - denial of service, deleting backups, ransomware",
    },
}

# every tactic each detectable technique serves (ATT&CK lists several for some)
TECHNIQUE_TACTICS: dict[str, list[str]] = {
    "T1595": ["reconnaissance"],
    "T1046": ["discovery"],
    "T1110": ["credential-access"],
    "T1078": [
        "initial-access",
        "persistence",
        "privilege-escalation",
        "defense-evasion",
    ],
    "T1190": ["initial-access"],
    "T1059": ["execution"],
    "T1136": ["persistence"],
    "T1543": ["persistence", "privilege-escalation"],
    "T1098": ["persistence", "privilege-escalation"],
    "T1070": ["defense-evasion"],
    "T1218": ["defense-evasion"],
    "T1557": ["credential-access", "collection"],
    "T1071": ["command-and-control"],
    "T1105": ["command-and-control"],
    "T1048": ["exfiltration"],
    "T1498": ["impact"],
    "T1499": ["impact"],
    "T1490": ["impact"],
    "T1003": ["credential-access"],
    "T1055": ["defense-evasion", "privilege-escalation"],
    "T1053": ["execution", "persistence", "privilege-escalation"],
    "T1547": ["persistence", "privilege-escalation"],
    "T1021": ["lateral-movement"],
    "T1562": ["defense-evasion"],
    "T1027": ["defense-evasion"],
    "T1548": ["privilege-escalation", "defense-evasion"],
    "T1134": ["defense-evasion", "privilege-escalation"],
    "T1546": ["privilege-escalation", "persistence"],
    "T1574": ["persistence", "privilege-escalation", "defense-evasion"],
    "T1047": ["execution"],
    "T1569": ["execution"],
    "T1033": ["discovery"],
    "T1087": ["discovery"],
    "T1069": ["discovery"],
    "T1016": ["discovery"],
    "T1082": ["discovery"],
    "T1018": ["discovery"],
    "T1558": ["credential-access"],
    "T1555": ["credential-access"],
    "T1550": ["defense-evasion", "lateral-movement"],
    "T1564": ["defense-evasion"],
    "T1222": ["defense-evasion"],
    "T1112": ["defense-evasion"],
    "T1036": ["defense-evasion"],
    "T1220": ["defense-evasion"],
    "T1197": ["defense-evasion", "persistence"],
    "T1505": ["persistence"],
    "T1572": ["command-and-control"],
}

# phrases -> tactic slug (longest match wins; checked before attack names, so
# "defense evasion" is the ATT&CK tactic, not the ML evasion attack)
TACTIC_SYNONYMS: dict[str, str] = {
    "reconnaissance": "reconnaissance",
    "resource development": "resource-development",
    "initial access": "initial-access",
    "execution": "execution",
    "code execution": "execution",
    "persistence": "persistence",
    "privilege escalation": "privilege-escalation",
    "priv esc": "privilege-escalation",
    "defense evasion": "defense-evasion",
    "defence evasion": "defense-evasion",
    "defense-evasion": "defense-evasion",
    "evading detection": "defense-evasion",
    "credential access": "credential-access",
    "credential theft": "credential-access",
    "discovery": "discovery",
    "lateral movement": "lateral-movement",
    "collection": "collection",
    "command and control": "command-and-control",
    "command-and-control": "command-and-control",
    "c2": "command-and-control",
    "c&c": "command-and-control",
    "exfiltration": "exfiltration",
    "data exfiltration": "exfiltration",
    "data theft": "exfiltration",
    "impact": "impact",
}


# technique phrases -> ATT&CK id (a question naming one asks about that technique,
# not its whole tactic)
TECHNIQUE_SYNONYMS: dict[str, str] = {
    "credential dumping": "T1003",
    "dumping credentials": "T1003",
    "lsass": "T1003",
    "mimikatz": "T1003",
    "process injection": "T1055",
    "scheduled task": "T1053",
    "autostart": "T1547",
    "run key": "T1547",
    "remote desktop": "T1021",
    "rdp": "T1021",
    "psexec": "T1021",
    "log clearing": "T1070",
    "cleared logs": "T1070",
    "clearing logs": "T1070",
    "disabling antivirus": "T1562",
    "disabled defender": "T1562",
    "obfuscated": "T1027",
    "encoded powershell": "T1027",
    "new account": "T1136",
    "account creation": "T1136",
}

# named attacks whose ATT&CK technique is shared with others (Heartbleed and a SQL
# injection are both T1190): log events count as evidence only when they name it
FAMILY_KEYWORDS: dict[str, str] = {
    "Heartbleed": r"heartbleed|cve-2014-0160",
    "Mirai": r"mirai",
}


# current ATT&CK splits Defense Evasion (TA0005) into "stealth" and "defense
# impairment"; this SOC and its test sets use the classic 14 tactics
_TACTIC_COMPAT = {"stealth": "defense-evasion", "defense-impairment": "defense-evasion"}


def _enterprise() -> dict[str, tuple[str, list[str]]]:
    """Every ATT&CK Enterprise technique (src/knowledge/attack_enterprise.json, from
    MITRE's github.com/mitre/cti): id -> (name, tactics)."""
    global _ENTERPRISE
    if _ENTERPRISE is None:
        import json
        from pathlib import Path

        p = Path(__file__).with_name("attack_enterprise.json")
        raw = json.loads(p.read_text("utf-8"))["techniques"] if p.is_file() else {}
        _ENTERPRISE = {
            t: (
                v["name"],
                list(dict.fromkeys(_TACTIC_COMPAT.get(x, x) for x in v["tactics"])),
            )
            for t, v in raw.items()
        }
    return _ENTERPRISE


_ENTERPRISE: dict[str, tuple[str, list[str]]] | None = None


def tactics_of(tid: str) -> list[str]:
    """Tactics of a technique: the curated table first, then full ATT&CK."""
    base = tid.split(".")[0]
    return TECHNIQUE_TACTICS.get(base) or _enterprise().get(base, ("", []))[1]


def technique_info(tid: str) -> tuple[str, str]:
    """(name, primary tactic) for any ATT&CK technique id; (id, "?") if unknown."""
    from src.soc.correlate import TECHNIQUES

    base = tid.split(".")[0]
    if base in TECHNIQUES:
        return TECHNIQUES[base]
    name, tactics = _enterprise().get(base, (base, []))
    return name, (tactics[0] if tactics else "?")


def techniques_for_tactic(tactic: str) -> list[str]:
    return sorted(t for t, ts in TECHNIQUE_TACTICS.items() if tactic in ts)
