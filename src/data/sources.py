"""Dataset registry: id -> loader, schema contract, file list, label handling.

Adding a dataset means adding a ``DatasetSpec`` here, never editing
``preprocess.py`` (04_DATA §2). Every URL/slug is UNVERIFIED at kit time
(11_LIMITATIONS §3); ``download.py`` checks them at download time and fails loud.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import assert_no_duplicate_columns, strip_column_names
from utils.logging_config import get_logger
from utils.paths import FIXTURES_DIR, RAW_DIR

_log = get_logger(__name__)


@dataclass(frozen=True)
class DatasetSpec:
    """Everything preprocess needs to load and validate one dataset."""

    id: str
    label_col: str = "Label"
    expected_columns: frozenset[str] | None = None  # None => infer (fixtures)
    id_columns: tuple[str, ...] = (
        "Flow ID",
        "Source IP",
        "Src IP",
        "Destination IP",
        "Dst IP",
        "Source Port",
        "Src Port",
        "Timestamp",
    )
    timestamp_col: str | None = None
    kaggle_slug: str | None = None
    url: str | None = None
    files: tuple[str, ...] = ()  # relative to raw/<id>/
    label_scheme: str = "family"  # 'family' (9-class CIC) | 'binary' (BENIGN/ATTACK)
    drop_labels: frozenset[str] = frozenset()  # raw labels dropped before mapping
    # Header-less CSV support (some CICFlowMeter exports ship no header row):
    read_header: int | None = 0  # None => file has no header; use column_names
    column_names: tuple[str, ...] = ()  # names to assign when read_header is None
    label_from_path: bool = False  # derive label_col from the folder tree, not the file
    # Columns known to be shipped twice; the later copy is dropped ONLY after its
    # values are verified identical to the first (otherwise loading fails loud).
    duplicate_columns: tuple[str, ...] = ()
    # >0: keep a seeded random sample of at most this many rows per source file
    # (memory guard for datasets too large to hold whole; a recorded deviation).
    max_rows_per_file: int = 0
    encoding: str = ""  # CSV text encoding when not UTF-8 (e.g. "cp1252")
    ts_parser: str = ""  # "cic2017": day-first, 12-hour clock without AM/PM
    raw_id: str = ""  # read data/raw/<raw_id>/ instead of data/raw/<id>/ (same files)
    # sample while reading (chunks), so a multi-GB CSV is never held whole; only the
    # extra datasets use it (registered datasets keep their original read path)
    stream_sample: bool = False
    notes: str = ""


# Canonical CICFlowMeter v4 (2023/2024) 84-column layout, in file order. Used for
# header-less exports (CIC IoT-DIAD 2024): 7 identifier cols + 76 features + a
# trailing label col. Names ASSUMED from the standard extractor and flagged for
# verification against a headered sample before the mutability map is built.
_CICFLOWMETER_84: tuple[str, ...] = (
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp",
    "Flow Duration", "Total Fwd Packet", "Total Bwd packets",
    "Total Length of Fwd Packet", "Total Length of Bwd Packet",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
    "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWR Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Fwd Segment Size Avg", "Bwd Segment Size Avg", "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg", "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "FWD Init Win Bytes", "Bwd Init Win Bytes", "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std",
    "Idle Max", "Idle Min", "NeedManualLabel",
)  # fmt: skip

# CIC IoT-DIAD 2024 labels come from the folder tree; these 8 dir names are the
# classes (they line up 1:1 with the `ciot` scheme).
DIAD_CLASS_DIRS: tuple[str, ...] = (
    "Benign", "Brute Force", "DDOS", "DOS", "Mirai", "Recon", "Spoofing", "Web-Based",
)  # fmt: skip


REGISTRY: dict[str, DatasetSpec] = {
    "fixture": DatasetSpec(
        id="fixture",
        label_col="Label",
        expected_columns=None,
        notes="Committed synthetic mini-schema for tests/CI/--fast.",
    ),
    "cicids2017-improved": DatasetSpec(
        id="cicids2017-improved",
        label_col="label",  # VERIFIED 2026-08-23: V2 CSV uses lowercase 'label'
        expected_columns=None,  # 79-col underscore schema, verified at extract time
        kaggle_slug=None,  # UNVERIFIED — supply via `download.py --slug`
        # 'Comb' (6.4% of rows) maps to no CIC family and its provenance is
        # uncertain; dropped with a logged deviation rather than guessed into a
        # family (04_DATA §3, 11_LIMITATIONS §3). Revisit if 'Comb' is identified.
        drop_labels=frozenset({"Comb"}),
        notes="Liu et al. 2022. PRIMARY (79-col underscore schema; 'Comb' dropped).",
    ),
    "cicids2017-original": DatasetSpec(
        id="cicids2017-original",
        label_col="Label",
        expected_columns=None,
        kaggle_slug=None,
        # 'Fwd Header Length' appears twice (cols 35 and 56); verified identical in
        # all 8 MachineLearningCVE files (2.8M rows).
        duplicate_columns=("Fwd Header Length",),
        notes="Original CIC-IDS2017. Control arm for H5.",
    ),
    # The same CIC-IDS2017 capture as cicids2017-original, from the
    # GeneratedLabelledFlows (TrafficLabelling) export, which KEEPS the Flow ID,
    # IPs and Timestamp. Not independent data: used only for the time-ordered test
    # (T1_temporal: train on earlier traffic, test on later), owner 2026-10-13.
    "cicids2017-flows": DatasetSpec(
        id="cicids2017-flows",
        label_col="Label",
        expected_columns=None,
        timestamp_col="Timestamp",
        duplicate_columns=("Fwd Header Length",),
        drop_labels=frozenset({"nan"}),  # 288k blank rows in the Thursday file
        encoding="cp1252",  # "Web Attack  Brute Force": a cp1252 en dash
        ts_parser="cic2017",
        notes="CIC-IDS2017 TrafficLabelling export (IPs + timestamps); temporal test.",
    ),
    "lycos-ids2017": DatasetSpec(
        id="lycos-ids2017",
        label_col="Label",
        expected_columns=None,
        url="https://lycos-ids.univ-lemans.fr/",
        notes="Rosay et al. Independent corrected re-extraction; ships own split.",
    ),
    # External validity (Task 9 / E8, E9). Shared 43-feature NetFlow v9 schema from
    # the same extractor (04_DATA §2) is what makes the transfer comparison honest.
    # All slugs UNVERIFIED (11_LIMITATIONS §3) — download.py fails loud.
    "nf-cse-cic-ids2018-v2": DatasetSpec(
        id="nf-cse-cic-ids2018-v2",
        label_col="Label",  # binary 0/1; 'Attack' category not used (E8 is binary)
        expected_columns=None,
        id_columns=("Attack",),  # drop the string category col (not a numeric feature)
        kaggle_slug=None,
        label_scheme="binary",  # E8 shares only benign-vs-attack across NetFlow sets
        notes="Sarhan et al. 2022. E8 source; 43 NetFlow v9 features; binary labels.",
    ),
    "nf-unsw-nb15-v2": DatasetSpec(
        id="nf-unsw-nb15-v2",
        label_col="Label",  # binary 0/1; UNSW families do not map to CIC (§3)
        expected_columns=None,
        id_columns=("Attack",),  # drop the string category col (not a numeric feature)
        kaggle_slug=None,
        label_scheme="binary",
        notes="E8 target — different network entirely; same 43 features; binary.",
    ),
    # Recency arm (04_DATA §3): modern IoT traffic, own 39-feature schema and native
    # 8-class taxonomy (label_scheme="ciot"). Extracted to data/raw/cic-iot-2023/
    # (63 CSVs). Rows with a NaN label read back as the string "nan" -> dropped.
    "cic-iot-2023": DatasetSpec(
        id="cic-iot-2023",
        label_col="Label",
        expected_columns=None,  # 39-feature IoT schema, verified at extract time
        kaggle_slug=None,
        label_scheme="ciot",  # 33 IoT attacks grouped into 7 classes + BENIGN
        drop_labels=frozenset({"nan"}),
        # 63 files, ~45M rows: seeded per-file sample keeps it under ~3.8M rows so it
        # fits a 16 GB machine (training is capped at 60k/class). See 11_LIMITATIONS.
        max_rows_per_file=60_000,
        notes="Neto et al. 2023. Recency arm; 39 features; native 8-class taxonomy.",
    ),
    # Recency arm 2 (04_DATA §3): CIC IoT-DIAD 2024. Header-less 84-col CICFlowMeter
    # exports; labels come from the folder tree (files carry only "NeedManualLabel").
    # Reuses the `ciot` scheme (its 8 top-level dirs are exactly the ciot classes).
    "cic-iot-diad-2024": DatasetSpec(
        id="cic-iot-diad-2024",
        label_col="Label",  # synthesised from the folder tree by load_raw
        expected_columns=None,
        # The CSVs DO carry a header; its 84 names match _CICFLOWMETER_84 1:1
        # (verified on all real files) except the last, literally "Label" holding
        # the "NeedManualLabel" placeholder -> renamed so it is dropped as an id col.
        read_header=0,
        column_names=_CICFLOWMETER_84,
        # 129 files, ~20M rows / 10 GB: cannot be held whole on a 16 GB machine.
        # Seeded per-file sample (<= ~3.9M rows); the training set is capped at
        # 60k/class anyway. Deviation recorded in 11_LIMITATIONS.
        max_rows_per_file=30_000,
        label_from_path=True,
        id_columns=(
            "Flow ID",
            "Src IP",
            "Src Port",
            "Dst IP",
            "Protocol",
            "Timestamp",
            "NeedManualLabel",  # the placeholder label column in-file, dropped
        ),  # fmt: skip
        kaggle_slug=None,
        label_scheme="ciot",
        notes="CIC 2024. Recency arm 2; 84-col CICFlowMeter, folder labels; ciot map.",
    ),
    # --- extra datasets (owner 2026-10-14): tested by scripts/extra_datasets.py,
    # outside the registered design; native multi-class labels for confusion
    # matrices. Downloaded without sign-up (Hugging Face mirrors of the originals).
    "nf-ton-iot-v2": DatasetSpec(
        id="nf-ton-iot-v2",
        label_col="Attack",  # 10 native classes; the binary 'Label' is dropped
        expected_columns=None,
        id_columns=("Label", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT"),
        label_scheme="ton",
        max_rows_per_file=400_000,  # 16.9M rows: <= 400k per class per file
        stream_sample=True,
        notes="Sarhan et al. 2022. NetFlow v2 ToN-IoT; HF mirror Nora9029/NF-ToN-IoT-v2.",
    ),
    "nf-unsw-nb15-v2-multi": DatasetSpec(
        id="nf-unsw-nb15-v2-multi",
        raw_id="nf-unsw-nb15-v2",  # same parquet as E8's binary spec
        label_col="Attack",
        expected_columns=None,
        id_columns=("Label", "L4_SRC_PORT"),
        label_scheme="unsw",
        notes="NF-UNSW-NB15-v2 with its 10 native classes (E8 uses it binary).",
    ),
    "cic-ddos2019": DatasetSpec(
        id="cic-ddos2019",
        label_col="Label",
        expected_columns=None,
        id_columns=(
            "Unnamed: 0",
            "Flow ID",
            "Source IP",
            "Destination IP",
            "Source Port",
            "Timestamp",
            "SimillarHTTP",
            "Fwd Header Length.1",  # shipped twice; the export renamed the copy
        ),  # fmt: skip
        label_scheme="ddos19",
        max_rows_per_file=300_000,  # ~20M rows over 7 files: <= 300k per class per file
        stream_sample=True,
        notes="Sharafaldin et al. 2019, test day 03-11 (CICFlowMeter); HF mirror.",
    ),
    "flnet2023": DatasetSpec(
        id="flnet2023",
        label_col="Label",
        expected_columns=None,
        kaggle_slug=None,
        notes="Kumar et al. 2023. E9 native federated partitions (per-router capture).",
    ),
}


def get_spec(dataset_id: str) -> DatasetSpec:
    """Return the registry entry for ``dataset_id`` or fail loud."""
    if dataset_id not in REGISTRY:
        raise KeyError(f"Unknown dataset {dataset_id!r}; register it in sources.py")
    return REGISTRY[dataset_id]


def _csv(path: Path, **kw: Any) -> pd.DataFrame:
    """pyarrow fast path; a file with malformed rows (seen in the CIC-IoT-2023
    merged CSVs: a 25-value row in a 40-column file) is re-read with the C parser,
    skipping only the bad rows, instead of failing the whole dataset."""
    try:
        return pd.read_csv(path, engine="pyarrow", **kw)
    except (pd.errors.ParserError, ValueError) as exc:
        # default low_memory chunking: low_memory=False OOMs on the 1.8 GB CIC-V2 CSV
        df = pd.read_csv(path, engine="c", on_bad_lines="skip", **kw)
        if "names" not in kw:  # undo the C parser's "X.1" renaming of repeated
            # headers so drop_verified_duplicates sees them like the pyarrow path
            with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
                header = next(csv.reader(fh), [])
            if len(header) == len(df.columns):
                df.columns = header
        _log.warning(
            "%s: malformed rows skipped (%s); %d rows kept",
            Path(path).name,
            str(exc).splitlines()[0][:120],
            len(df),
        )
        return df


def _shrink(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 and int64 -> smallest lossless int, column by column."""
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes("int64").columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def _stream_sample(path: Path, spec: DatasetSpec) -> pd.DataFrame:
    """Class-capped sample read in chunks: each raw label keeps about
    ``max_rows_per_file`` rows (all of them when rarer), so rare classes survive and
    a multi-GB CSV is never held whole. Seeded (same rows every run)."""
    import numpy as np
    import pyarrow.csv as pacsv

    enc = {"encoding": spec.encoding} if spec.encoding else {}
    with open(path, encoding=spec.encoding or "utf-8", errors="replace") as fh:
        header = next(csv.reader(fh))
    raw_label = next(c for c in header if c.strip() == spec.label_col)
    labels = pacsv.read_csv(
        path, convert_options=pacsv.ConvertOptions(include_columns=[raw_label])
    ).column(0)
    counts = pd.Series(labels.to_pandas()).astype(str).value_counts()
    frac = (spec.max_rows_per_file / counts).clip(upper=1.0)
    rng = np.random.default_rng(0)
    parts = []
    for chunk in pd.read_csv(path, chunksize=500_000, low_memory=True, **enc):
        f = chunk[raw_label].astype(str).map(frac).fillna(1.0).to_numpy()
        parts.append(_shrink(chunk[rng.random(len(chunk)) < f]))
    out = pd.concat(parts, ignore_index=True)
    _log.info("%s: kept %d of %d rows (<= %d per class)", Path(path).name, len(out),
              int(counts.sum()), spec.max_rows_per_file)  # fmt: skip
    return out


def _read_frame(path: Path, spec: DatasetSpec) -> pd.DataFrame:
    """Read one CSV or Parquet source file by suffix. CSV stays low-memory.

    Honours ``spec.read_header``/``spec.column_names`` for header-less CSV exports.
    """
    if Path(path).suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if spec.stream_sample and spec.max_rows_per_file:
        return _stream_sample(Path(path), spec)
    # pyarrow engine: lower peak memory + faster than the C parser on the big
    # (1.8 GB) CIC-V2 CSV, which OOMs under low_memory=False type inference.
    enc = {"encoding": spec.encoding} if spec.encoding else {}
    if spec.read_header is None:
        return _csv(path, header=None, names=list(spec.column_names), **enc)
    df = _csv(path, **enc)
    if spec.column_names:  # headered file whose names we override (verified 1:1)
        if len(df.columns) != len(spec.column_names):
            raise ValueError(
                f"{Path(path).name}: {len(df.columns)} columns, expected "
                f"{len(spec.column_names)} for {spec.id!r}"
            )
        df.columns = list(spec.column_names)
    return df


def drop_verified_duplicates(
    df: pd.DataFrame, names: tuple[str, ...], source: str = ""
) -> pd.DataFrame:
    """Drop later copies of the named duplicate columns, after checking each copy is
    value-identical to the first. Differing copies fail loud (never silently pick)."""
    keep: list[int] = []
    first: dict[str, int] = {}
    for i, c in enumerate(df.columns):
        if c in first and c in names:
            if not df.iloc[:, first[c]].equals(df.iloc[:, i]):
                raise ValueError(
                    f"{source}: duplicate column {c!r} differs from its first copy; "
                    "refusing to drop it (resolve in the registry)."
                )
            continue
        first.setdefault(c, i)
        keep.append(i)
    return df.iloc[:, keep]


def _label_from_path(path: Path, base: Path) -> str:
    """Derive a raw label from the first known class dir in the file's path.

    CIC IoT-DIAD 2024 encodes the class in the folder tree, not the CSV. Matching a
    known class dir anywhere in the relative path is robust to the extra nesting the
    archive ships with. Fails loud on an unrecognised path (a new class dir).
    """
    parts = Path(path).resolve().relative_to(base.resolve()).parts
    for part in parts:
        if part in DIAD_CLASS_DIRS:
            return part
    raise ValueError(
        f"No known class dir {DIAD_CLASS_DIRS} in {path!r}; add it (sources.py)."
    )


MEMORY_OVERRIDES = (
    Path(__file__).resolve().parents[2] / "config" / "memory_overrides.json"
)


def memory_row_scale(dataset_id: str) -> float:
    """Fraction of each source file to keep for ``dataset_id`` (1.0 = all).

    Written by scripts/supervisor.py only after that dataset ran out of memory: a
    recorded deviation (results/MEMORY_DEVIATIONS.md), never a silent change."""
    try:
        doc = json.loads(MEMORY_OVERRIDES.read_text(encoding="utf-8"))
        scale = float(doc.get("datasets", {}).get(dataset_id, 1.0))
    except (OSError, ValueError, TypeError, AttributeError):
        return 1.0
    return scale if 0.0 < scale < 1.0 else 1.0


def parse_timestamps(values: pd.Series, spec: DatasetSpec) -> pd.Series:
    """Timestamps for a temporal split, per the dataset's quirks.

    ``cic2017``: day-first dates in two formats ("6/7/2017 8:59",
    "03/07/2017 08:55:58") on a 12-hour clock WITHOUT AM/PM. Capture ran from about
    08:55 to 17:00, so an hour below 8 is afternoon (+12 h); otherwise afternoon
    flows would sort before the morning and break the time order.
    """
    if spec.ts_parser == "cic2017":
        ts = pd.to_datetime(values, dayfirst=True, format="mixed", errors="coerce")
        pm = ts.dt.hour < 8
        return ts.where(~pm, ts + pd.Timedelta(hours=12))
    return pd.to_datetime(values, errors="coerce")


def load_raw(spec: DatasetSpec) -> pd.DataFrame:
    """Load and concatenate a dataset's CSV files; strip names, reject dup columns.

    Args:
        spec: The dataset registry entry.

    Returns:
        The concatenated frame with stripped column names.

    Raises:
        FileNotFoundError: If no source files are found (with download hint).
    """
    if spec.id == "fixture":
        paths = [FIXTURES_DIR / "fixture.csv"]
    else:
        base = RAW_DIR / (spec.raw_id or spec.id)
        # Accept both CSV and Parquet (the NetFlow-v2 datasets ship as parquet).
        paths = (
            [base / f for f in spec.files]
            if spec.files
            # recursive: archives unpack into sub-folders (MERGED_CSV/, ...);
            # skip macOS resource forks (__MACOSX/, ._name.csv)
            else sorted(
                q
                for q in [*base.rglob("*.csv"), *base.rglob("*.parquet")]
                if "__MACOSX" not in q.parts and not q.name.startswith("._")
            )
        )
        # a second extract of the same file in another sub-folder (same name and
        # size) would be loaded twice; keep the first copy only
        seen: set[tuple[str, int]] = set()
        uniq = []
        for q in (Path(q) for q in paths if Path(q).exists()):
            if (q.name, q.stat().st_size) in seen:
                _log.warning("%s: duplicate copy of %s skipped", spec.id, q)
                continue
            seen.add((q.name, q.stat().st_size))
            uniq.append(q)
        paths = uniq
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        raise FileNotFoundError(
            f"No CSV/Parquet for {spec.id!r}. Run download.py first; the source in "
            "04_DATA §2 is unverified and must be supplied via --slug/--url."
        )
    scale = memory_row_scale(spec.id)
    if scale < 1.0:
        _log.warning(
            "%s: memory override keeps %.0f%% of each file's rows (out-of-memory "
            "deviation, see results/MEMORY_DEVIATIONS.md)",
            spec.id,
            100 * scale,
        )
    frames = []
    for p in paths:
        frame = strip_column_names(_read_frame(Path(p), spec))
        if spec.duplicate_columns:
            frame = drop_verified_duplicates(
                frame, spec.duplicate_columns, Path(p).name
            )
        if (
            spec.max_rows_per_file
            and not spec.stream_sample  # already capped per class while reading
            and len(frame) > spec.max_rows_per_file
        ):
            frame = frame.sample(n=spec.max_rows_per_file, random_state=0)
        if scale < 1.0 and len(frame) > 1:
            frame = frame.sample(n=max(1, int(len(frame) * scale)), random_state=0)
        if spec.label_from_path:
            frame[spec.label_col] = _label_from_path(Path(p), RAW_DIR / spec.id)
        frames.append(_shrink(frame))
    df = pd.concat(frames, ignore_index=True)
    assert_no_duplicate_columns(df)
    # concat can re-widen a column whose per-file dtypes differ; shrink again
    # (per-column, so peak memory grows by one column, never a whole block)
    return _shrink(df)
