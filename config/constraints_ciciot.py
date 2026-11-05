"""Mutability map for CIC-IoT-2023 (recency arm, 39 features) — 02_THREAT_MODEL 4.1.

Its own schema (NOT the CICFlowMeter 78/47 layout): protocol one-hot indicators,
TCP-flag number/count columns, and packet-length aggregate statistics. Class
assignment follows the owner's session-3 sign-off guidance:
  * protocol one-hots + Protocol Type -> IMMUTABLE (dictated by the target service);
  * *_flag_number / *_count + byte/packet totals + header length -> INCREASE_ONLY
    (an attacker can add flagged/padded packets, not remove protocol-required ones);
  * Min/Max/AVG/Std/IAT/TTL -> FREE_BOUNDED (controllable within the train range;
    not recomputable from any other column present in this schema);
  * Variance -> DERIVED (= Std**2, recomputed; see src/attack/constraints.py);
  * Rate -> DERIVED with no closed form here (packet-rate inputs are ambiguous in
    this layout) -> frozen+flagged, never freely perturbed.

String values (not the IMMUTABLE/... constants) to avoid an import cycle, matching
constraints_cicv2.py. Merged into MUTABILITY_MAP by config/constraints.py.
NEEDS HUMAN SIGN-OFF against a dtype-verified schema before Stage 4 attack runs.
"""

from __future__ import annotations

MUTABILITY_MAP_CICIOT: dict[str, str] = {
    "Header_Length": "INCREASE_ONLY",
    "Protocol Type": "IMMUTABLE",
    "Time_To_Live": "FREE_BOUNDED",
    "Rate": "DERIVED",
    "fin_flag_number": "INCREASE_ONLY",
    "syn_flag_number": "INCREASE_ONLY",
    "rst_flag_number": "INCREASE_ONLY",
    "psh_flag_number": "INCREASE_ONLY",
    "ack_flag_number": "INCREASE_ONLY",
    "ece_flag_number": "INCREASE_ONLY",
    "cwr_flag_number": "INCREASE_ONLY",
    "ack_count": "INCREASE_ONLY",
    "syn_count": "INCREASE_ONLY",
    "fin_count": "INCREASE_ONLY",
    "rst_count": "INCREASE_ONLY",
    "HTTP": "IMMUTABLE",
    "HTTPS": "IMMUTABLE",
    "DNS": "IMMUTABLE",
    "Telnet": "IMMUTABLE",
    "SMTP": "IMMUTABLE",
    "SSH": "IMMUTABLE",
    "IRC": "IMMUTABLE",
    "TCP": "IMMUTABLE",
    "UDP": "IMMUTABLE",
    "DHCP": "IMMUTABLE",
    "ARP": "IMMUTABLE",
    "ICMP": "IMMUTABLE",
    "IGMP": "IMMUTABLE",
    "IPv": "IMMUTABLE",
    "LLC": "IMMUTABLE",
    "Tot sum": "INCREASE_ONLY",
    "Min": "FREE_BOUNDED",
    "Max": "FREE_BOUNDED",
    "AVG": "FREE_BOUNDED",
    "Std": "FREE_BOUNDED",
    "Tot size": "INCREASE_ONLY",
    "IAT": "FREE_BOUNDED",
    "Number": "INCREASE_ONLY",
    "Variance": "DERIVED",
}

# Rounded after projection. Set from the VERIFIED int64 columns in the real CSVs
# (Merged01.csv sample): the *_count columns, packet Min/Max, Tot sum, and Number.
# Deliberately excluded despite the name: Header_Length and Tot size read back as
# float64 (per-flow averages, not integer byte counts), and the *_flag_number
# columns are per-flow fractions in [0, 1] — rounding any of these would be wrong.
INTEGER_FEATURES_CICIOT: frozenset[str] = frozenset(
    {
        "ack_count",
        "syn_count",
        "fin_count",
        "rst_count",
        "Min",
        "Max",
        "Tot sum",
        "Number",
    }
)
