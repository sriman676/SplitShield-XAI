"""EMBER (feature version 2) raw features from a Windows PE file on disk, using LIEF
(owner 2026-10-15: files seen in logs are also scanned by the malware model).

Produces the same raw-feature record layout EMBER publishes (histogram,
byteentropy, strings, general, header, section, imports, exports,
datadirectories), so ``src/mlareas/malware_features.vectorize`` can read it.

Caveat (stated wherever results are shown): EMBER's features were computed with
LIEF 0.9; this uses LIEF 0.16, whose enum names and a few counts can differ, so
scores on real files are NOT verified against EMBER. Only the false-alarm rate is
measured, on this PC's Microsoft-signed system files
(``scripts/validate_pescan.py``). Scan hits are therefore low-confidence.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np

MAX_BYTES = 64 * 2**20  # larger files are skipped (time + memory)
_PRINTABLE = re.compile(rb"[\x20-\x7f]{5,}")
_PATHS = re.compile(rb"c:\\", re.I)
_URLS = re.compile(rb"https?://", re.I)
_REGISTRY = re.compile(rb"HKEY_")
_MZ = re.compile(rb"MZ")


def available() -> bool:
    try:
        import lief  # noqa: F401
    except ImportError:
        return False
    return True


def _name(x: Any) -> str:
    """'MACHINE_TYPES.AMD64' / enum -> 'AMD64' (EMBER's short form)."""
    return str(x).split(".")[-1]


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def byte_entropy(b: np.ndarray, window: int = 2048, step: int = 1024) -> list[int]:
    """EMBER's 16x16 byte-entropy histogram."""
    out = np.zeros((16, 16), dtype=np.int64)
    if len(b) < window:
        blocks = [b]
    else:
        shape = ((len(b) - window) // step + 1, window)
        strides = (b.strides[0] * step, b.strides[0])
        blocks = list(np.lib.stride_tricks.as_strided(b, shape=shape, strides=strides))
    for block in blocks:
        coarse = np.bincount(block >> 4, minlength=16)
        # exactly as EMBER: probabilities over the WINDOW size, entropy x2 (16 coarse
        # bins hold 4 of the 8 bits), 16 rows of 0.5 bit
        p = coarse[coarse > 0].astype(np.float64) / window
        h = float(-(p * np.log2(p)).sum()) * 2
        row = min(int(h * 2), 15)
        out[row] += coarse
    return [int(v) for v in out.flatten()]


def strings(data: bytes) -> dict[str, Any]:
    found = _PRINTABLE.findall(data)
    if found:
        lengths = [len(s) for s in found]
        joined = np.frombuffer(b"".join(found), dtype=np.uint8) - 0x20
        dist = np.bincount(joined, minlength=96)[:96]
        printables = int(dist.sum())
    else:
        lengths, dist, printables = [], np.zeros(96, dtype=np.int64), 0
    return {
        "numstrings": len(found),
        "avlength": float(np.mean(lengths)) if lengths else 0.0,
        "printabledist": [int(v) for v in dist],
        "printables": printables,
        "entropy": _entropy(dist),
        "paths": len(_PATHS.findall(data)),
        "urls": len(_URLS.findall(data)),
        "registry": len(_REGISTRY.findall(data)),
        "MZ": len(_MZ.findall(data)),
    }


def _pe_parts(binary: Any, data: bytes) -> dict[str, Any]:
    h, opt = binary.header, binary.optional_header
    out: dict[str, Any] = {}
    out["general"] = {
        "size": len(data),
        "vsize": int(getattr(binary, "virtual_size", 0)),
        "has_debug": int(bool(binary.has_debug)),
        "exports": len(binary.exported_functions),
        "imports": len(binary.imported_functions),
        "has_relocations": int(bool(binary.has_relocations)),
        "has_resources": int(bool(binary.has_resources)),
        "has_signature": int(bool(binary.has_signatures)),
        "has_tls": int(bool(binary.has_tls)),
        "symbols": len(binary.symbols),
    }
    out["header"] = {
        "coff": {
            "timestamp": int(h.time_date_stamps),
            "machine": _name(h.machine),
            "characteristics": [_name(c) for c in h.characteristics_list],
        },
        "optional": {
            "subsystem": _name(opt.subsystem),
            "dll_characteristics": [_name(c) for c in opt.dll_characteristics_lists],
            "magic": _name(opt.magic),
            "major_image_version": opt.major_image_version,
            "minor_image_version": opt.minor_image_version,
            "major_linker_version": opt.major_linker_version,
            "minor_linker_version": opt.minor_linker_version,
            "major_operating_system_version": opt.major_operating_system_version,
            "minor_operating_system_version": opt.minor_operating_system_version,
            "major_subsystem_version": opt.major_subsystem_version,
            "minor_subsystem_version": opt.minor_subsystem_version,
            "sizeof_code": opt.sizeof_code,
            "sizeof_headers": opt.sizeof_headers,
            "sizeof_heap_commit": opt.sizeof_heap_commit,
        },
    }
    try:
        entry = binary.section_from_offset(binary.va_to_offset(binary.entrypoint)).name
    except Exception:  # noqa: BLE001 - entry point outside every section
        entry = ""
    out["section"] = {
        "entry": entry,
        "sections": [
            {"name": s.name, "size": s.size, "entropy": float(s.entropy),
             "vsize": s.virtual_size, "props": [_name(c) for c in s.characteristics_lists]}
            for s in binary.sections
        ],
    }  # fmt: skip
    imports: dict[str, list[str]] = {}
    for lib in binary.imports:
        imports.setdefault(lib.name, []).extend(
            e.name if e.name else f"ordinal{e.ordinal}" for e in lib.entries
        )
    out["imports"] = imports
    out["exports"] = [e.name for e in binary.exported_functions]
    out["datadirectories"] = [
        {"name": _name(d.type), "size": d.size, "virtual_address": d.rva}
        for d in binary.data_directories
    ]
    return out


def extract(path: Path) -> dict[str, Any]:
    """Raw EMBER-style features of one PE file. Raises ValueError for non-PE files."""
    import lief

    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"{path.name}: larger than {MAX_BYTES // 2**20} MB")
    data = path.read_bytes()
    if not data.startswith(b"MZ"):
        raise ValueError(f"{path.name}: not a PE file")
    b = np.frombuffer(data, dtype=np.uint8)
    rec: dict[str, Any] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "histogram": [int(v) for v in np.bincount(b, minlength=256)],
        "byteentropy": byte_entropy(b),
        "strings": strings(data),
    }
    lief.logging.disable()
    binary = lief.PE.parse(str(path))
    if binary is None:
        raise ValueError(f"{path.name}: LIEF could not parse it")
    rec.update(_pe_parts(binary, data))
    return rec
