"""Feature mutability map (02_THREAT_MODEL §4.1) — SKELETON.

This file holds the *data*: which flow feature belongs to which mutability class.
The projection *logic* lives in ``src/attack/constraints.py`` (Task 5) and imports
this map; it is never reimplemented.

REVIEW REQUIRED (09_TASKS Task 5): the concrete per-column mapping is a domain
judgement, not a code decision. It must be filled in against the *verified* real
schema and reviewed by the human before any Stage 4 attack runs. The heuristics
below are a labelling starting point only, not the final map.
"""

from __future__ import annotations

# Mutability classes (02_THREAT_MODEL §4.1)
IMMUTABLE = "IMMUTABLE"  # attacker cannot control; frozen, excluded from mask
INCREASE_ONLY = "INCREASE_ONLY"  # can pad not remove; raw-space delta clipped >= 0
FREE_BOUNDED = "FREE_BOUNDED"  # controllable within train min/max
DERIVED = "DERIVED"  # algebraic function of others; recomputed, never perturbed

MUTABILITY_CLASSES = (IMMUTABLE, INCREASE_ONLY, FREE_BOUNDED, DERIVED)

# Explicit per-column map. Keys are canonical (whitespace-stripped) names.
#
# STATUS (Task 5): filled for the 9-column fixture mini-schema only. The real
# ~91-column CIC schema is still unverified (download blocked on Kaggle creds,
# 11_LIMITATIONS §3), so the full map is completed against the verified schema in
# the Task 1 revisit. HEURISTIC_PATTERNS below proposes it; the human confirms.
#
# HUMAN REVIEW REQUIRED before any Stage 4 attack runs (09_TASKS Task 5). Two
# judgement calls to check explicitly:
#   * "Flow IAT Mean" / "Idle Mean" are named as *means* (which the DERIVED
#     heuristic matches), but 02_THREAT_MODEL §4.1 lists "inter-arrival time
#     statistics" and "idle/active times" as FREE_BOUNDED examples. They are also
#     not recomputable from any other column present in this schema. => classified
#     FREE_BOUNDED, not DERIVED. Confirm.
#   * "Destination Port" is IMMUTABLE here (dictated by the target service). This
#     is the retain-vs-drop shortcut feature (04_DATA §4 step 5); freezing it for
#     attacks is separate from the keep_dest_port modelling decision. Confirm.
MUTABILITY_MAP: dict[str, str] = {
    "Destination Port": IMMUTABLE,
    "Flow Duration": INCREASE_ONLY,
    "Total Fwd Packets": INCREASE_ONLY,
    "Total Backward Packets": INCREASE_ONLY,
    "Fwd Header Length": INCREASE_ONLY,
    "Flow IAT Mean": FREE_BOUNDED,
    "Idle Mean": FREE_BOUNDED,
    "Flow Bytes/s": DERIVED,
    "Flow Packets/s": DERIVED,
}

# Integer-valued raw features: rounded after projection, before the model is
# queried (02_THREAT_MODEL §4.2 rule 2). Counts, lengths, and microsecond
# durations are integers; rates and time-statistics are continuous.
INTEGER_FEATURES: frozenset[str] = frozenset(
    {
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Fwd Header Length",
    }
)

# Merge the signed-off real CIC-IDS2017-V2 schema (78 underscore-named features).
# The fixture (space-named) and CIC-V2 (underscore-named) keys never collide, so a
# union is safe; resolve_classes only checks features present in a given run.
from config.constraints_ciciot import (  # noqa: E402
    INTEGER_FEATURES_CICIOT,
    MUTABILITY_MAP_CICIOT,
)
from config.constraints_cicv2 import (  # noqa: E402
    INTEGER_FEATURES_CICV2,
    MUTABILITY_MAP_CICV2,
)
from config.constraints_diad import (  # noqa: E402
    INTEGER_FEATURES_DIAD,
    MUTABILITY_MAP_DIAD,
)
from config.constraints_netflow_draft import (  # noqa: E402
    INTEGER_FEATURES_NETFLOW_DRAFT,
    MUTABILITY_MAP_NETFLOW_DRAFT,
)

# CIC-IoT-2023 uses its own 39-feature schema; its bare-name keys (Rate, AVG, Std,
# Min, Max, IAT, Number, Variance) don't collide with the CIC/fixture underscore or
# space names, so the union stays unambiguous (resolve_classes only checks the
# features present in a given run).
MUTABILITY_MAP = {
    **MUTABILITY_MAP,
    **MUTABILITY_MAP_CICV2,
    **MUTABILITY_MAP_CICIOT,
    **MUTABILITY_MAP_DIAD,
    **MUTABILITY_MAP_NETFLOW_DRAFT,
}
# Classes in use without a human sign-off: results that depend on them are flagged.
UNREVIEWED_MAP_FEATURES: frozenset[str] = frozenset(MUTABILITY_MAP_NETFLOW_DRAFT)
INTEGER_FEATURES = (
    INTEGER_FEATURES
    | INTEGER_FEATURES_CICV2
    | INTEGER_FEATURES_CICIOT
    | INTEGER_FEATURES_DIAD
    | INTEGER_FEATURES_NETFLOW_DRAFT
)

# Starting-point heuristics used to PROPOSE a map for human review — substrings
# matched case-insensitively against column names. Not authoritative.
HEURISTIC_PATTERNS: dict[str, tuple[str, ...]] = {
    IMMUTABLE: ("protocol", "destination port"),
    INCREASE_ONLY: (
        "total",
        "packet",
        "byte",
        "header length",
        "flow duration",
        "count",
    ),
    FREE_BOUNDED: ("iat", "idle", "active"),
    DERIVED: ("/s", "mean", "std", "ratio", "average", "variance"),
}
