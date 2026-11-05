"""False-alarm rate of the LIEF file scan on known-good files (owner 2026-10-15).

Known-good = .exe / .dll files from C:\\Windows\\System32 whose Authenticode
signature Windows reports as Valid with a Microsoft signer (embedded or catalog).
Detection on real malware is NOT measured: live malware is never downloaded (owner
rule), so scan hits stay low-confidence. Results: results/extra/pescan_validation.json.

    python -m scripts.validate_pescan [--n 1500]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

from utils.paths import RESULTS_DIR

SYS32 = Path(r"C:\Windows\System32")


def microsoft_signed(paths: list[Path]) -> list[Path]:
    """Windows' own verdict (Get-AuthenticodeSignature handles catalog signing)."""
    keep: list[Path] = []
    for i in range(0, len(paths), 200):
        chunk = paths[i : i + 200]
        listing = ";".join(f"'{str(p).replace(chr(39), chr(39) * 2)}'" for p in chunk)
        ps = (f"@({listing}) | ForEach-Object {{ $s = Get-AuthenticodeSignature -LiteralPath $_; "
              "if ($s.Status -eq 'Valid' -and $s.SignerCertificate.Subject -match 'O=Microsoft') "
              "{ $_ } }")  # fmt: skip
        r = subprocess.run([shutil.which("pwsh") or "powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=600)  # fmt: skip
        keep += [Path(x.strip()) for x in r.stdout.splitlines() if x.strip()]
    return keep


def main() -> None:
    from src.mlareas import malware
    from src.soc import pefeatures

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()
    files = sorted(
        p
        for p in SYS32.iterdir()
        if p.suffix.lower() in (".exe", ".dll") and p.is_file()
    )
    random.Random(0).shuffle(files)
    signed = microsoft_signed(files[: a.n])
    counts = {"benign": 0, "suspicious": 0, "malicious": 0}
    flagged, errors = [], 0
    t0 = time.time()
    for p in signed:
        try:
            res = malware.check_file_features(pefeatures.extract(p))
        except Exception:  # noqa: BLE001 - unparsable / locked files are counted
            errors += 1
            continue
        counts[res["verdict"]] += 1
        if res["verdict"] != "benign":
            flagged.append({"file": p.name, "verdict": res["verdict"],
                            "p": round(res["malware_probability"], 4)})  # fmt: skip
    n = sum(counts.values())
    out = {
        "what": "LIEF file scan on Microsoft-signed System32 files (known good)",
        "sampled": a.n, "microsoft_signed": len(signed), "scanned": n, "errors": errors,
        "verdicts": counts,
        "false_alarm_rate_malicious": counts["malicious"] / max(1, n),
        "false_alarm_rate_any": (counts["malicious"] + counts["suspicious"]) / max(1, n),
        "files_per_s": round(n / max(1e-9, time.time() - t0), 1),
        "flagged": flagged[:50],
        "not_measured": "detection on real malware (never downloaded); hits stay low-confidence",
    }  # fmt: skip
    (RESULTS_DIR / "extra").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "extra" / "pescan_validation.json").write_text(
        json.dumps(out, indent=1), "utf-8"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "flagged"}, indent=1))


if __name__ == "__main__":
    main()
