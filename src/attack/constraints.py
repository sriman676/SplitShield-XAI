"""Feature-constraint projection for attack realism (02_THREAT_MODEL §4).

Research purpose: perturbing tabular flow features is not like perturbing image
pixels. This module projects a perturbed feature vector back onto the set of
domain-plausible flows so that an attack success rate is not inflated by flows
that could not exist on a network. It operates ONLY on this project's own model
objects and already-public benchmark data (02_THREAT_MODEL §5); it is not a
generic, retargetable tool.

The mutability *map* (which column is which class) is data and lives in
``config/constraints.py`` — imported here, never reimplemented (§4). This module
holds the projection *logic* and the ``DERIVED`` recomputation *formulas*.

Projection order for one perturbed sample (02_THREAT_MODEL §4.2):
    1. work in RAW feature space (invert the scaler first) — an epsilon in
       StandardScaler units has no physical meaning and is not comparable across
       features (rule 1);
    2. apply per-class constraints in raw units:
         IMMUTABLE     -> frozen to the original value (rule: excluded from mask);
         INCREASE_ONLY -> raw delta clipped to >= 0 (can pad, not remove);
         FREE_BOUNDED  -> clipped to the training-split observed [min, max];
         DERIVED       -> not perturbed; recomputed from its inputs (rule 3);
    3. round integer features (rule 2), before the model is queried, not after;
    4. re-scale for the model.

``DERIVED`` recomputation formulas are uncertain for the real CICFlowMeter schema
(exact definitions vary by extractor release). Only the columns present in the
fixture mini-schema have a known formula below; an unrecognised ``DERIVED`` column
is frozen to its original value and flagged, never silently perturbed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from config.constraints import (
    DERIVED,
    FREE_BOUNDED,
    IMMUTABLE,
    INCREASE_ONLY,
    INTEGER_FEATURES,
    MUTABILITY_CLASSES,
    MUTABILITY_MAP,
)
from utils.logging_config import get_logger

_log = get_logger(__name__)


# --- DERIVED recomputation formulas -----------------------------------------
#
# Each takes the raw feature matrix (rows = samples, cols = features) and a
# name->column-index map, and returns the recomputed column as a 1-D array.
# Documented per 02_THREAT_MODEL §4.2 rule 3. FIXTURE-SCHEMA ONLY and marked
# uncertain: the real CICFlowMeter "Flow Bytes/s" divides *byte* counts (absent
# from the mini-schema) by duration; here the fixture defines both rates from
# packet counts, so both recompute identically. Replace against the verified
# schema in the Task 1 revisit.


def _rate_from_packets(raw: np.ndarray, col: dict[str, int]) -> np.ndarray:
    """(Total Fwd + Total Backward Packets) per second; duration is microseconds.

    UNCERTAIN vs. the real schema (see module docstring). Guards divide-by-zero
    the way step-4 cleaning already removes (zero-duration rows produce Inf).
    """
    pkts = raw[:, col["Total Fwd Packets"]] + raw[:, col["Total Backward Packets"]]
    dur_us = raw[:, col["Flow Duration"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(dur_us > 0, pkts * 1e6 / dur_us, 0.0)
    return out


def _safe_div(num: np.ndarray, den: np.ndarray, scale: float = 1.0) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num * scale / den, 0.0)


# Real CIC-IDS2017-V2 (underscore-named) formulas. The V2 schema carries byte
# counts (absent from the fixture), so rates and segment/packet sizes recompute
# exactly. Flow Duration is in microseconds (scale 1e6). Down/Up_Ratio has no
# reliable closed form across CICFlowMeter releases -> left unknown (freeze+flag).
def _v2_bytes_per_s(r: np.ndarray, c: dict[str, int]) -> np.ndarray:
    b = r[:, c["Total_Length_of_Fwd_Packets"]] + r[:, c["Total_Length_of_Bwd_Packets"]]
    return _safe_div(b, r[:, c["Flow_Duration"]], 1e6)


def _v2_pkts_per_s(r: np.ndarray, c: dict[str, int]) -> np.ndarray:
    p = r[:, c["Total_Fwd_Packets"]] + r[:, c["Total_Backward_Packets"]]
    return _safe_div(p, r[:, c["Flow_Duration"]], 1e6)


def _v2_fwd_pkts_per_s(r: np.ndarray, c: dict[str, int]) -> np.ndarray:
    return _safe_div(r[:, c["Total_Fwd_Packets"]], r[:, c["Flow_Duration"]], 1e6)


def _v2_bwd_pkts_per_s(r: np.ndarray, c: dict[str, int]) -> np.ndarray:
    return _safe_div(r[:, c["Total_Backward_Packets"]], r[:, c["Flow_Duration"]], 1e6)


def _v2_avg_pkt_size(r: np.ndarray, c: dict[str, int]) -> np.ndarray:
    b = r[:, c["Total_Length_of_Fwd_Packets"]] + r[:, c["Total_Length_of_Bwd_Packets"]]
    p = r[:, c["Total_Fwd_Packets"]] + r[:, c["Total_Backward_Packets"]]
    return _safe_div(b, p)


def _v2_avg_fwd_seg(r: np.ndarray, c: dict[str, int]) -> np.ndarray:
    return _safe_div(
        r[:, c["Total_Length_of_Fwd_Packets"]], r[:, c["Total_Fwd_Packets"]]
    )


def _v2_avg_bwd_seg(r: np.ndarray, c: dict[str, int]) -> np.ndarray:
    return _safe_div(
        r[:, c["Total_Length_of_Bwd_Packets"]], r[:, c["Total_Backward_Packets"]]
    )


def _v2_pkt_len_variance(r: np.ndarray, c: dict[str, int]) -> np.ndarray:
    return r[:, c["Packet_Length_Std"]] ** 2


# CIC-IoT-2023 (recency arm) — packet-length variance is Std**2, the only column in
# that schema with a reliable closed form. "Rate" is DERIVED but has no dependable
# formula in this layout, so it is left unregistered -> frozen+flagged.
def _ciot_variance(r: np.ndarray, c: dict[str, int]) -> np.ndarray:
    return r[:, c["Std"]] ** 2


DERIVED_FORMULAS: dict[str, Callable[[np.ndarray, dict[str, int]], np.ndarray]] = {
    # fixture (space-named) mini-schema
    "Flow Bytes/s": _rate_from_packets,
    "Flow Packets/s": _rate_from_packets,
    # real CIC-IDS2017-V2 (underscore-named) schema
    "Flow_Bytes/s": _v2_bytes_per_s,
    "Flow_Packets/s": _v2_pkts_per_s,
    "Fwd_Packets/s": _v2_fwd_pkts_per_s,
    "Bwd_Packets/s": _v2_bwd_pkts_per_s,
    "Average_Packet_Size": _v2_avg_pkt_size,
    "Avg_Fwd_Segment_Size": _v2_avg_fwd_seg,
    "Avg_Bwd_Segment_Size": _v2_avg_bwd_seg,
    "Packet_Length_Variance": _v2_pkt_len_variance,
    # CIC-IoT-2023 (own 39-feature schema)
    "Variance": _ciot_variance,
}


# --- corrected follow-up formulas (owner decision 2026-10-14) ----------------
#
# Used only when a run sets ``corrections: [derived_v2]``; registered runs keep
# DERIVED_FORMULAS above. Keyed by a canonical name (lower case, spaces and
# underscores folded, CIC-DIAD spellings mapped), so the same verified formula
# applies to the space-named original CIC-IDS2017, CIC-IoT-DIAD-2024 and the
# underscore-named V2 schema. Each was checked against the extractor's own values
# on 120k real rows (CIC-IDS2017-original and CIC-IoT-DIAD-2024): 100 % agreement
# within 1 %. Down/Up Ratio is CICFlowMeter's integer division bwd/fwd. Average
# Packet Size matched only 38-47 % of rows and stays frozen + flagged.
_CANON_SYNONYMS = {
    "total_fwd_packet": "total_fwd_packets",
    "total_bwd_packets": "total_backward_packets",
    "total_length_of_fwd_packet": "total_length_of_fwd_packets",
    "total_length_of_bwd_packet": "total_length_of_bwd_packets",
    "fwd_segment_size_avg": "avg_fwd_segment_size",
    "bwd_segment_size_avg": "avg_bwd_segment_size",
}


def canonical(name: str) -> str:
    """Schema-independent feature key (see the corrected formulas above)."""
    key = re.sub(r"[ _]+", "_", name.strip()).lower()
    return _CANON_SYNONYMS.get(key, key)


def _c_bytes(r: np.ndarray, c: dict[str, int]) -> np.ndarray:
    return (
        r[:, c["total_length_of_fwd_packets"]] + r[:, c["total_length_of_bwd_packets"]]
    )


def _c_pkts(r: np.ndarray, c: dict[str, int]) -> np.ndarray:
    return r[:, c["total_fwd_packets"]] + r[:, c["total_backward_packets"]]


CORRECTED_FORMULAS: dict[str, Callable[[np.ndarray, dict[str, int]], np.ndarray]] = {
    "flow_bytes/s": lambda r, c: _safe_div(
        _c_bytes(r, c), r[:, c["flow_duration"]], 1e6
    ),
    "flow_packets/s": lambda r, c: _safe_div(
        _c_pkts(r, c), r[:, c["flow_duration"]], 1e6
    ),
    "fwd_packets/s": lambda r, c: _safe_div(
        r[:, c["total_fwd_packets"]], r[:, c["flow_duration"]], 1e6
    ),
    "bwd_packets/s": lambda r, c: _safe_div(
        r[:, c["total_backward_packets"]], r[:, c["flow_duration"]], 1e6
    ),
    "avg_fwd_segment_size": lambda r, c: _safe_div(
        r[:, c["total_length_of_fwd_packets"]], r[:, c["total_fwd_packets"]]
    ),
    "avg_bwd_segment_size": lambda r, c: _safe_div(
        r[:, c["total_length_of_bwd_packets"]], r[:, c["total_backward_packets"]]
    ),
    "packet_length_variance": lambda r, c: r[:, c["packet_length_std"]] ** 2,
    "down/up_ratio": lambda r, c: np.floor(
        _safe_div(r[:, c["total_backward_packets"]], r[:, c["total_fwd_packets"]])
    ),
    "variance": lambda r, c: r[:, c["std"]] ** 2,  # CIC-IoT-2023
}


def is_prenormalized(x_train_raw: np.ndarray) -> bool:
    """True when every feature already lies in [0, 1] (a min-max scaled release,
    e.g. the Kaggle CIC-IDS-2017-V2 CSV): raw units are unrecoverable, so integer
    rounding and DERIVED recomputation would corrupt the values, not repair them."""
    x = np.asarray(x_train_raw, dtype=np.float64)
    return (
        bool(x.size)
        and float(np.nanmin(x)) >= -1e-9
        and float(np.nanmax(x)) <= 1 + 1e-9
    )


def _alias(name: str) -> str:
    """Original CIC-IDS2017 spells the V2 features with spaces ("Total Length of Fwd
    Packets" = "Total_Length_of_Fwd_Packets"). Owner decision 2026-10-13: such a name
    reuses the signed-off V2 classification of the identical feature; an exact entry
    always wins."""
    return name.strip().replace(" ", "_")


def resolve_classes(feature_names: list[str]) -> dict[str, str]:
    """Return name -> mutability class for the given schema, or fail loudly.

    Every feature must be classified (§4.1: "exactly one class"). A missing or
    unknown-valued entry is a review gap, not something to default silently.
    """
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for name in feature_names:
        cls = MUTABILITY_MAP.get(name) or MUTABILITY_MAP.get(_alias(name))
        if cls is None:
            missing.append(name)
        elif cls not in MUTABILITY_CLASSES:
            raise ValueError(f"Feature {name!r} has invalid mutability class {cls!r}")
        else:
            resolved[name] = cls
    if missing:
        raise ValueError(
            "Unclassified features (fill config/constraints.MUTABILITY_MAP and have "
            f"the human review, 09_TASKS Task 5): {missing}"
        )
    return resolved


@dataclass
class FeatureConstraints:
    """Projects perturbed samples onto domain-plausible flows for one schema.

    Built once per run from the fitted scaler and the raw training split (for
    ``FREE_BOUNDED`` bounds). ``mask`` is the boolean perturbation mask an attack
    applies (True where a feature may move); ``IMMUTABLE`` and ``DERIVED`` columns
    are False.
    """

    feature_names: list[str]
    classes: dict[str, str]
    scaler: Any
    lo: np.ndarray  # per-feature raw training min (FREE_BOUNDED bounds)
    hi: np.ndarray  # per-feature raw training max
    mask: np.ndarray  # bool, True = perturbable
    integer_cols: np.ndarray  # bool, True = round after projection
    _col: dict[str, int]
    unknown_derived: tuple[str, ...]  # DERIVED cols with no formula (frozen + flagged)
    corrected: bool = False  # follow-up formulas + pre-normalized guard
    normalized: bool = (
        False  # corrected run on [0,1]-scaled data: no rounding/recompute
    )

    # --- construction -------------------------------------------------------

    @classmethod
    def from_run(
        cls,
        feature_names: list[str],
        scaler: Any,
        x_train_raw: np.ndarray,
        corrected: bool = False,
    ) -> FeatureConstraints:
        classes = resolve_classes(feature_names)
        normalized = corrected and is_prenormalized(x_train_raw)
        if corrected:
            col = {canonical(n): i for i, n in enumerate(feature_names)}
        else:
            col = {n: i for i, n in enumerate(feature_names)}
        lo = x_train_raw.min(axis=0)
        hi = x_train_raw.max(axis=0)
        mask = np.array(
            [classes[n] in (INCREASE_ONLY, FREE_BOUNDED) for n in feature_names]
        )
        integer_cols = np.array(
            [
                n in INTEGER_FEATURES or _alias(n) in INTEGER_FEATURES
                for n in feature_names
            ]
        )
        if normalized:
            integer_cols[:] = False
            _log.warning(
                "pre-normalized [0,1] features: integer rounding and DERIVED "
                "recomputation disabled (raw units unrecoverable); DERIVED columns "
                "frozen, realism approximate"
            )
        known = (
            {n for n in feature_names if canonical(n) in CORRECTED_FORMULAS}
            if corrected and not normalized
            else set(DERIVED_FORMULAS)
            if not corrected
            else set()
        )
        unknown = tuple(
            n for n in feature_names if classes[n] == DERIVED and n not in known
        )
        if unknown:
            _log.warning(
                "DERIVED features with no recomputation formula (frozen to original, "
                "not recomputed; add a formula before relying on their realism): %s",
                list(unknown),
            )
        return cls(
            feature_names=feature_names,
            classes=classes,
            scaler=scaler,
            lo=lo,
            hi=hi,
            mask=mask,
            integer_cols=integer_cols,
            _col=col,
            unknown_derived=unknown,
            corrected=corrected,
            normalized=normalized,
        )

    # --- scaled <-> raw -----------------------------------------------------

    def to_raw(self, x_scaled: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(x_scaled)

    def to_scaled(self, x_raw: np.ndarray) -> np.ndarray:
        return self.scaler.transform(x_raw)

    # --- projection ---------------------------------------------------------

    def project_raw(self, x_adv_raw: np.ndarray, x_orig_raw: np.ndarray) -> np.ndarray:
        """Project perturbed raw samples onto the constraint set (steps 2-3).

        Args:
            x_adv_raw: Perturbed samples in RAW feature units, shape (n, d).
            x_orig_raw: The originals they derive from, same shape.

        Returns:
            The projected raw samples. Not modified in place.
        """
        x = np.array(x_adv_raw, dtype=np.float64, copy=True)
        orig = np.asarray(x_orig_raw, dtype=np.float64)
        for j, name in enumerate(self.feature_names):
            c = self.classes[name]
            if c == IMMUTABLE:
                x[:, j] = orig[:, j]
            elif c == INCREASE_ONLY:
                x[:, j] = orig[:, j] + np.clip(x[:, j] - orig[:, j], 0.0, None)
            elif c == FREE_BOUNDED:
                x[:, j] = np.clip(x[:, j], self.lo[j], self.hi[j])
            elif c == DERIVED:
                x[:, j] = orig[:, j]  # frozen now; recomputed below if known

        # Round integers before recomputing DERIVED (rates depend on rounded counts).
        if self.integer_cols.any():
            x[:, self.integer_cols] = np.rint(x[:, self.integer_cols])

        self._recompute_derived(x)
        return x

    def _recompute_derived(self, x: np.ndarray) -> None:
        """Apply each DERIVED formula whose column is present, in place.

        A formula whose *input* columns are absent in this schema (e.g. a shared
        rate name that indexes a differently-named count column) is skipped, leaving
        that DERIVED column frozen to its pre-set value — the freeze+flag contract,
        never a crash. resolve_classes still guarantees every feature is classified.
        """
        if self.normalized:
            return  # raw units unrecoverable: DERIVED stay frozen (flagged)
        table = CORRECTED_FORMULAS if self.corrected else DERIVED_FORMULAS
        for name, formula in table.items():
            if name in self._col:
                try:
                    x[:, self._col[name]] = formula(x, self._col)
                except KeyError:
                    continue  # inputs not in this schema; column stays frozen

    def project_scaled(
        self, x_adv_scaled: np.ndarray, x_orig_scaled: np.ndarray
    ) -> np.ndarray:
        """Same projection, entering and leaving in scaled (model) space."""
        return self.to_scaled(
            self.project_raw(self.to_raw(x_adv_scaled), self.to_raw(x_orig_scaled))
        )

    def violation_rate(
        self, x_adv_raw: np.ndarray, x_orig_raw: np.ndarray, atol: float = 1e-6
    ) -> float:
        """Fraction of samples changed by projection (pre-repair violation, §4.2 r4).

        A sample "violates" if projecting it moves any feature by more than
        ``atol`` in raw units — i.e. it was outside the constraint set.
        """
        projected = self.project_raw(x_adv_raw, x_orig_raw)
        moved = np.abs(projected - np.asarray(x_adv_raw, dtype=np.float64)) > atol
        return float(np.mean(moved.any(axis=1)))

    # --- absolute (from-scratch) validity, for the generative attack (T2) ---
    #
    # The projection above is perturbation-relative (needs an original). Generated
    # flows have no original, so realism is judged against absolute physical bounds:
    # the training-split [min, max] range, integer-valuedness, and DERIVED
    # consistency (01_RESEARCH_DESIGN §4.5 domain-constraint violation rate).

    def repair_raw(self, x_raw: np.ndarray) -> np.ndarray:
        """Clip generated samples to physical bounds, round ints, recompute DERIVED.

        IMMUTABLE features are left as generated (the attacker does emit a value;
        it is simply not something they perturb relative to a victim flow).
        """
        x = np.array(x_raw, dtype=np.float64, copy=True)
        for j, name in enumerate(self.feature_names):
            c = self.classes[name]
            if c in (INCREASE_ONLY, FREE_BOUNDED):
                x[:, j] = np.clip(x[:, j], self.lo[j], self.hi[j])
        if self.integer_cols.any():
            x[:, self.integer_cols] = np.rint(x[:, self.integer_cols])
        self._recompute_derived(x)
        return x

    def absolute_violation_rate(self, x_raw: np.ndarray, atol: float = 1e-6) -> float:
        """Fraction of generated samples repair would change (§4.5, pre-repair)."""
        repaired = self.repair_raw(x_raw)
        moved = np.abs(repaired - np.asarray(x_raw, dtype=np.float64)) > atol
        return float(np.mean(moved.any(axis=1)))
