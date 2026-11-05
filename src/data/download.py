"""Dataset download with loud failure and no silent substitution (04_DATA §1).

Sources in the registry are UNVERIFIED (11_LIMITATIONS §3). This tool takes the
slug/url explicitly, verifies files landed and are plausibly sized, and reports
what actually worked so the human can confirm it against 04_DATA §2. It never
substitutes a different dataset.

    python -m src.data.download --source cicids2017-improved --slug <kaggle/slug>
    python -m src.data.download --source lycos-ids2017 --url <https://...>

Kaggle credentials come from ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
(08_ENGINEERING_STANDARDS §3). They are never logged or passed on the CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.sources import get_spec
from utils.logging_config import configure_logging, get_logger
from utils.paths import RAW_DIR, ensure_within

_log = get_logger(__name__)
MIN_PLAUSIBLE_BYTES = 10_000


def _verify(out_dir: Path) -> list[Path]:
    """Return CSVs found under ``out_dir``, failing loud if none/too small."""
    csvs = sorted(out_dir.glob("**/*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSVs under {out_dir} after download. The source may have moved; "
            "verify the slug/url against 04_DATA §2 and retry. Do NOT substitute "
            "a different dataset."
        )
    small = [p for p in csvs if p.stat().st_size < MIN_PLAUSIBLE_BYTES]
    if small:
        _log.warning(
            "Suspiciously small files (<%d bytes): %s",
            MIN_PLAUSIBLE_BYTES,
            [p.name for p in small],
        )
    return csvs


def _kaggle_download(slug: str, out_dir: Path) -> None:
    """Download and unzip a Kaggle dataset. Guarded import; loud on missing creds."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # guarded
    except ImportError as exc:
        raise RuntimeError(
            "kaggle not installed; pip install -r requirements.txt"
        ) from exc
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as exc:  # missing/invalid credentials
        raise RuntimeError(
            "Kaggle authentication failed. Put credentials in ~/.kaggle/kaggle.json "
            "or set KAGGLE_USERNAME/KAGGLE_KEY, then retry."
        ) from exc
    api.dataset_download_files(slug, path=str(out_dir), unzip=True, quiet=False)


def _url_download(url: str, out_dir: Path) -> None:
    """Download a single file by URL (no execution, just bytes)."""
    import urllib.request

    dest = out_dir / Path(url).name
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed public dataset URL


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download a registered dataset")
    ap.add_argument("--source", required=True, help="registry id")
    ap.add_argument("--slug", help="kaggle dataset slug")
    ap.add_argument("--url", help="direct download URL")
    ap.add_argument("--out-dir", help="override output dir")
    args = ap.parse_args(argv)
    configure_logging()

    spec = get_spec(args.source)
    out_dir = ensure_within(Path(args.out_dir) if args.out_dir else RAW_DIR / spec.id)
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = args.slug or spec.kaggle_slug
    url = args.url or spec.url
    if not slug and not url:
        _log.error(
            "No source for %s. Its registry slug/url is unverified; supply one "
            "with --slug or --url (04_DATA §2).",
            spec.id,
        )
        return 2

    if slug:
        _kaggle_download(slug, out_dir)
    else:
        _url_download(url, out_dir)  # type: ignore[arg-type]

    found = _verify(out_dir)
    _log.info("Downloaded %d CSV(s) for %s under %s", len(found), spec.id, out_dir)
    print(
        f"[download] {spec.id}: {len(found)} CSV(s) in {out_dir}. "
        "Confirm this matches 04_DATA §2 before relying on it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
