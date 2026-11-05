"""Copy the current deployed detector aside before a redeploy (never deletes).

artifacts/deploy/ plus its reports (results/deploy_report.json,
results/soc_selftest.json) are copied to artifacts/deploy_backup/<dataset>/. If a
backup for that dataset already exists it is kept as is (idempotent: a retried
phase never overwrites the first, good copy). Restore by copying the folder back.

    python -m scripts.backup_deploy
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "artifacts" / "deploy"
BACKUPS = ROOT / "artifacts" / "deploy_backup"
REPORTS = ("deploy_report.json", "soc_selftest.json")


def backup(deploy: Path = DEPLOY, backups: Path = BACKUPS) -> Path | None:
    """Return the backup folder, or None when there is nothing deployed."""
    cols = deploy / "columns.json"
    if not cols.is_file():
        print("[backup-deploy] nothing deployed; no backup needed")
        return None
    dataset = json.loads(cols.read_text("utf-8")).get("dataset", "unknown")
    dest = backups / dataset
    if dest.exists():
        print(f"[backup-deploy] backup already present: {dest}")
        return dest
    shutil.copytree(deploy, dest / "deploy")
    for name in REPORTS:
        src = deploy.parents[1] / "results" / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    print(f"[backup-deploy] {dataset} detector copied to {dest}")
    return dest


def main() -> int:
    backup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
