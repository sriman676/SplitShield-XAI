"""DRAFT mutability map for NetFlow v2 (NF-CSE-CIC-IDS2018-v2, NF-UNSW-NB15-v2).

STATUS: IN USE, NOT YET HUMAN-REVIEWED (owner decision 2026-10-13: use the NetFlow
datasets for attack testing now). Merged by config/constraints.py; every attack-stage
result on these features carries an "unreviewed_mutability_map" deviation
(02_THREAT_MODEL 4.1 / 09_TASKS Task 5 still ask for a human sign-off). After review,
remove the deviation by emptying UNREVIEWED_MAP_FEATURES in config/constraints.py.

Reasoning follows the signed-off CIC-V2 map (constraints_cicv2.py):
  * endpoint identity the attacker does not choose per flow (destination port,
    protocol, L7 protocol, ICMP type, DNS query type) -> IMMUTABLE; the source
    port is OS-chosen and spoofable -> FREE_BOUNDED;
  * volumes the attacker can pad but not remove (bytes, packets, retransmissions,
    duration, size-bucket packet counts, flag ORs) -> INCREASE_ONLY;
  * per-packet extremes and stack settings the attacker controls (TTL, packet
    lengths, TCP window, DNS TTL/query id) -> FREE_BOUNDED within train min/max;
  * quantities computed from others (per-second bytes, average throughput) ->
    DERIVED (frozen until a recomputation formula is added);
  * server responses the attacker cannot set (FTP return code) -> IMMUTABLE.
"""

from __future__ import annotations

MUTABILITY_MAP_NETFLOW_DRAFT: dict[str, str] = {
    "L4_SRC_PORT": "FREE_BOUNDED",
    "L4_DST_PORT": "IMMUTABLE",
    "PROTOCOL": "IMMUTABLE",
    "L7_PROTO": "IMMUTABLE",
    "IN_BYTES": "INCREASE_ONLY",
    "IN_PKTS": "INCREASE_ONLY",
    "OUT_BYTES": "INCREASE_ONLY",
    "OUT_PKTS": "INCREASE_ONLY",
    "TCP_FLAGS": "INCREASE_ONLY",  # cumulative OR of flags: bits can be added
    "CLIENT_TCP_FLAGS": "INCREASE_ONLY",
    "SERVER_TCP_FLAGS": "INCREASE_ONLY",
    "FLOW_DURATION_MILLISECONDS": "INCREASE_ONLY",
    "DURATION_IN": "INCREASE_ONLY",
    "DURATION_OUT": "INCREASE_ONLY",
    "MIN_TTL": "FREE_BOUNDED",
    "MAX_TTL": "FREE_BOUNDED",
    "LONGEST_FLOW_PKT": "FREE_BOUNDED",
    "SHORTEST_FLOW_PKT": "FREE_BOUNDED",
    "MIN_IP_PKT_LEN": "FREE_BOUNDED",
    "MAX_IP_PKT_LEN": "FREE_BOUNDED",
    "SRC_TO_DST_SECOND_BYTES": "DERIVED",
    "DST_TO_SRC_SECOND_BYTES": "DERIVED",
    "RETRANSMITTED_IN_BYTES": "INCREASE_ONLY",
    "RETRANSMITTED_IN_PKTS": "INCREASE_ONLY",
    "RETRANSMITTED_OUT_BYTES": "INCREASE_ONLY",
    "RETRANSMITTED_OUT_PKTS": "INCREASE_ONLY",
    "SRC_TO_DST_AVG_THROUGHPUT": "DERIVED",
    "DST_TO_SRC_AVG_THROUGHPUT": "DERIVED",
    "NUM_PKTS_UP_TO_128_BYTES": "INCREASE_ONLY",
    "NUM_PKTS_128_TO_256_BYTES": "INCREASE_ONLY",
    "NUM_PKTS_256_TO_512_BYTES": "INCREASE_ONLY",
    "NUM_PKTS_512_TO_1024_BYTES": "INCREASE_ONLY",
    "NUM_PKTS_1024_TO_1514_BYTES": "INCREASE_ONLY",
    "TCP_WIN_MAX_IN": "FREE_BOUNDED",
    "TCP_WIN_MAX_OUT": "FREE_BOUNDED",
    "ICMP_TYPE": "IMMUTABLE",
    "ICMP_IPV4_TYPE": "IMMUTABLE",
    "DNS_QUERY_ID": "FREE_BOUNDED",
    "DNS_QUERY_TYPE": "IMMUTABLE",
    "DNS_TTL_ANSWER": "FREE_BOUNDED",
    "FTP_COMMAND_RET_CODE": "IMMUTABLE",
}

# count / size / port / flag / TTL fields hold whole numbers (rounded after attacks)
INTEGER_FEATURES_NETFLOW_DRAFT: frozenset[str] = frozenset(
    k
    for k in MUTABILITY_MAP_NETFLOW_DRAFT
    if not k.endswith(("_SECOND_BYTES", "_AVG_THROUGHPUT"))
)
