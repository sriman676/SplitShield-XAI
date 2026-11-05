"""Dataset intake: find dataset archives/files, unpack them safely, and place each
under ``data/raw/<dataset-id>/`` where the loaders expect it.

    python -m src.data.intake            # scan project root + datasets/ folder

Robust by design — one bad archive never stops the others:
  * identification by exact archive name first, then by the column *signature* of
    the data inside (peeked straight from the archive, no extraction needed), so a
    renamed zip still lands in the right place; unrecognised archives are skipped
  * nested archives (zip inside zip) are unpacked in place, depth <= 3
  * path-traversal ("zip-slip") members are refused; tar uses the safe 'data' filter
  * integrity test before unpacking; free-disk-space check against the unpacked size
  * idempotent: an unchanged archive (same size + mtime) that is already unpacked
    is skipped, so re-running after an interruption is cheap
  * loose CSV/Parquet files dropped into ``datasets/`` are identified and copied
A report is written to ``data/raw/.intake.json`` and printed.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any

from src.data.sources import DIAD_CLASS_DIRS
from utils.logging_config import configure_logging, get_logger
from utils.paths import PROJECT_ROOT, RAW_DIR

_log = get_logger(__name__)

DATASETS_DIR = PROJECT_ROOT / "datasets"
REPORT = RAW_DIR / ".intake.json"
DATA_SUFFIXES = (".csv", ".parquet")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tgz", ".tar.gz")

KNOWN_ARCHIVES = {
    "cic-ids-2017-v2.zip": "cicids2017-improved",
    "machinelearningcsv.zip": "cicids2017-original",
    "merged_csv.zip": "cic-iot-2023",
    "nf-unsw-nb15-v2.parquet.zip": "nf-unsw-nb15-v2",
    "nf-cse-cic-ids2018-v2.parquet.zip": "nf-cse-cic-ids2018-v2",
}


# --- identification --------------------------------------------------------------


def identify(columns: list[str], name: str, parts: tuple[str, ...] = ()) -> str | None:
    """Dataset id from a file's header columns + its name/path, or None if unknown."""
    cols = {c.strip() for c in columns}
    low = name.lower()
    if {"L4_SRC_PORT", "L4_DST_PORT", "IN_BYTES"} <= cols:  # NetFlow v2 (shared schema)
        if "unsw" in low:
            return "nf-unsw-nb15-v2"
        if "2018" in low or "cse" in low:
            return "nf-cse-cic-ids2018-v2"
        return None  # same 43 columns for both; the name must disambiguate
    if {"Header_Length", "Protocol Type", "Tot size"} <= cols:
        return "cic-iot-2023"
    if {"Destination_Port", "Flow_Duration"} <= cols:
        return "cicids2017-improved"
    if {"Flow ID", "Src IP", "Dst IP"} <= cols and set(parts) & set(DIAD_CLASS_DIRS):
        return "cic-iot-diad-2024"
    if {"Flow ID", "Source IP", "Timestamp", "Destination Port", "Label"} <= cols:
        return "cicids2017-flows"  # TrafficLabelling export (keeps IPs + times)
    if {"Destination Port", "Flow Duration", "Label"} <= cols and "Flow ID" not in cols:
        return "cicids2017-original"
    return None


def _header_from_bytes(raw: bytes) -> list[str]:
    line = raw.decode("utf-8-sig", errors="replace").splitlines()[0] if raw else ""
    return next(csv.reader([line]), [])


def _parquet_columns(fh: Any) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_schema(fh).names)


def peek_zip(path: Path) -> str | None:
    """Identify a zip from the headers of the data files inside (no extraction)."""
    with zipfile.ZipFile(path) as zf:
        votes: dict[str, int] = {}
        members = [m for m in zf.infolist() if _is_data(m.filename)][:5]
        for m in members:
            try:
                with zf.open(m) as fh:
                    if m.filename.lower().endswith(".parquet"):
                        cols = _parquet_columns(io.BytesIO(fh.read()))
                    else:
                        cols = _header_from_bytes(fh.read(65536))
            except (OSError, zipfile.BadZipFile, ValueError):
                continue
            ds = identify(cols, f"{path.name}/{m.filename}", Path(m.filename).parts)
            if ds:
                votes[ds] = votes.get(ds, 0) + 1
        return max(votes, key=votes.__getitem__) if votes else None


def identify_file(path: Path) -> str | None:
    try:
        if path.suffix.lower() == ".parquet":
            cols = _parquet_columns(path)
        else:
            with open(path, "rb") as fh:
                cols = _header_from_bytes(fh.read(65536))
    except (OSError, ValueError):
        return None
    return identify(cols, path.name, path.parts)


def _is_data(name: str) -> bool:
    p = Path(name)
    return (
        p.suffix.lower() in DATA_SUFFIXES
        and "__MACOSX" not in p.parts
        and not p.name.startswith("._")
    )


def _is_archive(p: Path) -> bool:
    return p.name.lower().endswith(ARCHIVE_SUFFIXES)


# --- safe extraction -----------------------------------------------------------


def _check_space(needed: int, where: Path) -> None:
    free = shutil.disk_usage(where if where.exists() else where.anchor).free
    if needed * 1.05 > free:
        raise OSError(
            f"not enough disk space: need {needed / 1e9:.1f} GB, "
            f"{free / 1e9:.1f} GB free on {where.anchor}"
        )


def safe_extract(archive: Path, dest: Path) -> None:
    """Extract a zip/tar into ``dest``, refusing members that escape it."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    if archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad:
                raise zipfile.BadZipFile(f"corrupt member {bad!r}")
            _check_space(sum(m.file_size for m in zf.infolist()), RAW_DIR)
            for m in zf.infolist():
                if not (dest / m.filename).resolve().is_relative_to(root):
                    raise ValueError(f"unsafe path in archive: {m.filename!r}")
            zf.extractall(dest)
    else:
        with tarfile.open(archive) as tf:
            _check_space(sum(m.size for m in tf.getmembers()), RAW_DIR)
            tf.extractall(dest, filter="data")
    shutil.rmtree(dest / "__MACOSX", ignore_errors=True)


def _expand_nested(folder: Path, depth: int = 3) -> None:
    for _ in range(depth):
        inner = [p for p in folder.rglob("*") if p.is_file() and _is_archive(p)]
        if not inner:
            return
        for z in inner:
            safe_extract(z, z.with_name(z.name.split(".")[0]))
            z.unlink()


def _has_data(folder: Path) -> bool:
    return folder.is_dir() and any(_is_data(str(p)) for p in folder.rglob("*"))


# --- driver --------------------------------------------------------------------


def _fingerprint(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


def run(roots: list[Path] | None = None) -> list[dict[str, Any]]:
    """Process every archive / loose data file found. Returns the report rows."""
    roots = roots or [PROJECT_ROOT, DATASETS_DIR]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    prev = {r["source"]: r for r in _load_report()}
    rows: list[dict[str, Any]] = []

    archives: list[Path] = []
    loose: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        # project root: top level only (never walk .venv / data); datasets/: recursive
        it = root.rglob("*") if root == DATASETS_DIR else root.iterdir()
        for p in it:
            if p.is_file() and _is_archive(p):
                archives.append(p)
            elif p.is_file() and root == DATASETS_DIR and _is_data(p.name):
                loose.append(p)

    for arc in sorted(set(archives)):
        row: dict[str, Any] = {"source": str(arc), "fingerprint": _fingerprint(arc)}
        try:
            ds = KNOWN_ARCHIVES.get(arc.name.lower())
            if ds is None and arc.name.lower().endswith(".zip"):
                ds = peek_zip(arc)
            row["dataset"] = ds
            if ds is None:
                row["status"] = "skipped: not a recognised dataset"
                rows.append(row)
                continue
            dest = RAW_DIR / ds
            old = prev.get(str(arc), {})
            if old.get("fingerprint") == row["fingerprint"] and _has_data(dest):
                row["status"] = "ok (already unpacked)"
                rows.append(row)
                continue
            # manually placed data is left alone; data from another archive of the
            # same dataset is not "manual" and gets this archive merged next to it
            ours = any(r.get("dataset") == ds for r in [*prev.values(), *rows])
            if _has_data(dest) and not old and not ours:
                row["status"] = "ok (data already present)"
                rows.append(row)
                continue
            staging = RAW_DIR / "_staging" / ds
            shutil.rmtree(staging, ignore_errors=True)
            _log.info("unpacking %s -> %s", arc.name, dest)
            safe_extract(arc, staging)
            _expand_nested(staging)
            if not _has_data(staging):
                raise ValueError("archive contains no CSV/Parquet data")
            # one sub-folder per archive: a dataset shipped as several archives
            # keeps all of them; re-unpacking an archive replaces only its own files
            target = dest / arc.name.split(".")[0]
            shutil.rmtree(target, ignore_errors=True)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging), str(target))
            row["status"] = "ok (unpacked)"
        except (zipfile.BadZipFile, tarfile.TarError, OSError, ValueError) as exc:
            row["status"] = f"FAILED: {exc}"
            _log.error("intake %s failed: %s", arc.name, exc)
        rows.append(row)

    for f in sorted(loose):
        ds = identify_file(f)
        row = {"source": str(f), "dataset": ds, "fingerprint": _fingerprint(f)}
        try:
            if ds is None:
                row["status"] = "skipped: not a recognised dataset"
            else:
                (RAW_DIR / ds).mkdir(parents=True, exist_ok=True)
                target = RAW_DIR / ds / f.name
                if not target.exists() or _fingerprint(target) != row["fingerprint"]:
                    _check_space(f.stat().st_size, RAW_DIR)
                    shutil.copy2(f, target)
                row["status"] = "ok (copied)"
        except OSError as exc:
            row["status"] = f"FAILED: {exc}"
        rows.append(row)

    shutil.rmtree(RAW_DIR / "_staging", ignore_errors=True)
    REPORT.write_text(json.dumps(rows, indent=2), "utf-8")
    return rows


def _load_report() -> list[dict[str, Any]]:
    try:
        return json.loads(REPORT.read_text("utf-8"))
    except (OSError, ValueError):
        return []


def main() -> None:
    configure_logging()
    rows = run()
    for r in rows:
        print(
            f"[intake] {Path(r['source']).name:40s} -> {r.get('dataset') or '-':24s} {r['status']}"
        )
    have = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir() and _has_data(p))
    print(f"[intake] datasets ready: {', '.join(have) or 'none'}")
    if any(r["status"].startswith("FAILED") for r in rows):
        print("[intake] some archives failed (see above); the rest continue.")


if __name__ == "__main__":
    main()
