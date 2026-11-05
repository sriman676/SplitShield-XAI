"""Compile LOLBAS (Windows) and GTFOBins (Linux) into one matcher file (owner 2026-10-24).

Both projects document how ordinary system programs are misused (certutil downloading a
file, find spawning a shell). Each documented command becomes a pattern: the program
name plus the command's distinctive switches (and, for GTFOBins shells, the shell it
starts). Commands without a distinctive switch are left out: matching a bare program
name (every mshta, every tar) would flag normal use.

Sources (GPL-3.0, credited; kept as a separate data file, third_party/lolbins/):
  LOLBAS-Project/LOLBAS (lolbas-project.github.io/api/lolbas.json)
  GTFOBins/GTFOBins.github.io (_gtfobins/*, _data/functions.yml for the ATT&CK ids)

    python -m scripts.build_lolbins --lolbas PATH --gtfobins ZIP
Writes third_party/lolbins/lolbins.json (+ LICENSE note).
"""

from __future__ import annotations

import argparse
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import yaml

from utils.paths import PROJECT_ROOT

OUT = PROJECT_ROOT / "third_party" / "lolbins"
GENERIC = {"/c", "/k", "-c", "-e", "/q", "-q", "-y", "/y", "-v", "-f", "/f", "-o", "-i", "-a", "-r", "-p",
           "-n", "-s", "-x", "-t", "-l", "-d", "-h", "--", "-", "/?", "-?"}  # fmt: skip
SHELLS = ("/bin/sh", "/bin/bash", "/bin/dash", "/bin/zsh", "sh -c", "bash -c")


def _switches(cmd: str) -> list[str]:
    toks = []
    for raw in re.split(r"\s+", cmd.strip()):
        t = raw.strip("\"'`,;()").lower()
        if not t or "{" in t or t.startswith("$"):
            continue
        if t[0] in "-/" and len(t) >= 3 and not t.startswith(("//", "/path", "/dev/", "/tmp", "/bin", "/usr", "/etc")):
            t = t.split("=", 1)[0].split(":", 1)[0] if t.startswith("-") else t.split(":", 1)[0]
            if t not in GENERIC and len(t) >= 3:
                toks.append(t)
    return sorted(set(toks))


def lolbas(path: Path) -> list[dict[str, Any]]:
    out = []
    for e in json.loads(path.read_text("utf-8")):
        name = str(e.get("Name", "")).lower()
        for c in e.get("Commands") or []:
            tech = str(c.get("MitreID") or "")
            sw = _switches(str(c.get("Command", "")))
            if not tech.startswith("T") or not sw:
                continue
            out.append({"os": "windows", "binary": name, "switches": sw, "technique": tech,
                        "what": str(c.get("Category") or ""), "source": "LOLBAS"})  # fmt: skip
    return out


def gtfobins(zpath: Path) -> list[dict[str, Any]]:
    z = zipfile.ZipFile(zpath)
    root = next(n for n in z.namelist() if n.endswith("_data/functions.yml")).rsplit("_data/", 1)[0]
    fmap = yaml.safe_load(z.read(root + "_data/functions.yml"))
    out = []
    for n in z.namelist():
        if not n.startswith(root + "_gtfobins/") or n.endswith("/"):
            continue
        name = n.rsplit("/", 1)[-1].lower()
        try:
            doc = next(iter(yaml.safe_load_all(z.read(n).decode("utf-8"))))
        except (yaml.YAMLError, StopIteration):
            continue
        for fn, examples in (doc or {}).get("functions", {}).items():
            techs = [t for t in (fmap.get(fn) or {}).get("mitre") or [] if str(t).startswith("T")]
            if not techs or fn in ("file-read", "file-write"):
                continue  # reading / writing a file is normal use far too often
            for ex in examples or []:
                code = str(ex.get("code", ""))
                line = next((ln for ln in code.splitlines() if re.search(rf"(^|[\s/]){re.escape(name)}\b", ln)), "")
                sw = _switches(line)
                shell = next((s for s in SHELLS if s in line), None)
                if not sw and not shell:
                    continue
                for t in techs:
                    out.append({"os": "linux", "binary": name, "switches": sw, "shell": shell, "technique": t,
                                "what": fn, "source": "GTFOBins"})  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lolbas", type=Path, default=Path("D:/SplitShield-data/new/lolbas/lolbas.json"))
    ap.add_argument("--gtfobins", type=Path, default=Path("D:/SplitShield-data/new/gtfobins/gtfobins.zip"))
    a = ap.parse_args()
    pats = lolbas(a.lolbas) + gtfobins(a.gtfobins)
    seen, uniq = set(), []
    for p in pats:
        k = (p["os"], p["binary"], tuple(p["switches"]), p.get("shell"), p["technique"])
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lolbins.json").write_text(json.dumps({"built": time.strftime("%Y-%m-%d"), "patterns": uniq}, indent=0), "utf-8")
    (OUT / "LICENSE.md").write_text(
        "lolbins.json is compiled from LOLBAS-Project/LOLBAS and GTFOBins/GTFOBins.github.io,\n"
        "both GNU GPL-3.0 (https://www.gnu.org/licenses/gpl-3.0.html). It is distributed as a\n"
        "separate data file under that licence, with credit to both projects; the SplitShield\n"
        "code that reads it is MIT (see the project LICENSE).\n", "utf-8")  # fmt: skip
    print(len(uniq), "patterns:", sum(p["os"] == "windows" for p in uniq), "Windows,",
          sum(p["os"] == "linux" for p in uniq), "Linux")


if __name__ == "__main__":
    main()
