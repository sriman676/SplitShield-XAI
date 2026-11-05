"""Insider-threat user-day features from CMU CERT logs (owner 2026-10-15).

Purpose: turn the raw CERT activity logs (logon, device, file, email, http,
psychometric, LDAP) into ONE row per user-day that a model can score at the end
of that day. Everything is streamed in chunks and aggregated per user-day while
reading, so the 14 GB ``http.csv`` never sits in memory (peak RAM stays well under
the 6 GB budget).

No-leakage rules:
  * per-user baselines (mean / z-score vs the previous 30 days) use a closed-left
    time window, i.e. strictly earlier days only;
  * peer-group deviations compare a user with their role peers on the SAME day,
    which an analyst also has at the end of that day;
  * ``hr_leaving_soon`` marks the user's last LDAP month when the user is absent
    from the next monthly LDAP snapshot. It models an HR notice period (resignation
    known about a month ahead); it is the only forward-looking field and the
    training script reports an ablation without it.

Labels: a user-day is malicious when the answers file of a release insider lists
an event of that insider on that day (rows of other users in the answers files,
e.g. the victim's replies in scenario 3, are not counted).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATE_FMT = "%m/%d/%Y %H:%M:%S"
CHUNK_ROWS = 1_000_000
WORK_START, WORK_END = 7, 19  # after hours = before 07:00 or from 19:00
COMPANY_DOMAIN = "dtaa.com"
URL_RULES_VERSION = "v2"  # bump when URL_PATTERNS change (new http cache file)

# URL categories by keyword (generic, public category terms). Job-search and
# cloud rules look at the HOST only: words like "job" appear in random page paths
# of the synthetic web log (e.g. paypal.com/.../jobs...), which made them noisy.
_HOST = r"^[a-z]+://[^/]*"
URL_PATTERNS: dict[str, str] = {
    "http_job": _HOST + r"(?:job|career|hiring|recruit|monster\.com|indeed\.com|"
    r"linkedin\.com|simplyhired|careerbuilder|craigslist|glassdoor)",
    "http_wikileaks": r"wikileaks",
    "http_keylog": r"keylog|"
    + _HOST
    + r"(?:(?<![a-z])spy|spectorsoft|refog|softactivity|"
    r"webwatcher|relytec|actionalert)",
    "http_cloud": _HOST
    + r"(?:dropbox|drive\.google|onedrive|(?<![a-z])box\.com|mega\.nz|"
    r"mediafire|4shared|wetransfer|sendspace|rapidshare|megaupload)",
}
FILE_TYPES: dict[str, tuple[str, ...]] = {
    "file_doc": (".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".pdf"),
    "file_exe": (".exe", ".dll", ".bat", ".msi"),
    "file_zip": (".zip", ".rar", ".7z", ".gz"),
    "file_img": (".jpg", ".jpeg", ".png", ".gif", ".bmp"),
}

LOGON_COLS = ["logon_n", "logoff_n", "logon_ah", "pc_n", "first_hour", "last_hour"]
DEVICE_COLS = ["usb_conn_n", "usb_conn_ah", "usb_disc_n"]
FILE_COLS = ["file_n", "file_ah", *FILE_TYPES]
EMAIL_COLS = [
    "email_n",
    "email_ext_n",
    "email_recip_n",
    "email_bytes",
    "email_attach_n",
    "email_ext_attach_n",
    "email_ah",
]
HTTP_COLS = ["http_n", "http_ah", *URL_PATTERNS]
COUNT_COLS = LOGON_COLS + DEVICE_COLS + FILE_COLS + EMAIL_COLS + HTTP_COLS
OCEAN = ["O", "C", "E", "A", "N"]
BASELINE_COLS = [
    "logon_ah",
    "usb_conn_n",
    "usb_conn_ah",
    "file_n",
    "file_ah",
    "email_ext_n",
    "email_ext_attach_n",
    "email_bytes",
    "email_recip_n",
    "http_n",
    "http_job",
    "http_wikileaks",
    "http_keylog",
    "http_cloud",
    "pc_n",
    "first_hour",
    "last_hour",
]
PEER_COLS = ["usb_conn_n", "file_n", "email_ext_n", "http_job", "logon_ah", "pc_n"]
WINDOW7_COLS = {
    "job_days7": "http_job",
    "usb_days7": "usb_conn_n",
    "ah_days7": "logon_ah",
}
STATIC_COLS = [*OCEAN, "role_code", "dept_code", "func_code", "hr_leaving_soon"]
CALENDAR_COLS = ["weekday", "is_weekend"]


def feature_columns() -> list[str]:
    """Model input columns in a fixed order."""
    base = list(COUNT_COLS)
    base += [f"{c}_mean30" for c in BASELINE_COLS]
    base += [f"{c}_z30" for c in BASELINE_COLS]
    base += [f"{c}_peerz" for c in PEER_COLS]
    base += list(WINDOW7_COLS)
    return base + STATIC_COLS + CALENDAR_COLS


# ------------------------------------------------------------------ reading


def iter_csv(
    path: Path, usecols: Sequence[str], chunksize: int = CHUNK_ROWS
) -> Iterator[pd.DataFrame]:
    """Stream selected columns of a CERT CSV as string chunks."""
    yield from pd.read_csv(
        path,
        usecols=list(usecols),
        dtype=str,
        chunksize=chunksize,
        keep_default_na=False,
        on_bad_lines="skip",
    )


def _stamp(chunk: pd.DataFrame) -> pd.DataFrame:
    """Add ``day`` (midnight timestamp) and ``ah`` (after-hours / weekend flag)."""
    ts = pd.to_datetime(chunk["date"], format=DATE_FMT, errors="coerce")
    out = chunk.assign(ts=ts).dropna(subset=["ts"])
    hour = out["ts"].dt.hour
    weekend = out["ts"].dt.dayofweek >= 5
    after = (hour < WORK_START) | (hour >= WORK_END) | weekend
    return out.assign(
        day=out["ts"].dt.normalize(), hour=hour, ah=after.astype(np.int32)
    )


def _combine(parts: list[pd.DataFrame], how: dict[str, str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=list(how)).rename_axis(["user", "day"])
    return pd.concat(parts).groupby(level=["user", "day"]).agg(how)


def agg_logon(path: Path, chunksize: int = CHUNK_ROWS) -> pd.DataFrame:
    """Logons, logoffs, after-hours logons, distinct PCs, first/last hour."""
    parts, pcs = [], []
    for raw in iter_csv(path, ["date", "user", "pc", "activity"], chunksize):
        c = _stamp(raw)
        on = (c["activity"] == "Logon").astype(np.int32)
        c = c.assign(
            logon_n=on,
            logoff_n=1 - on,
            logon_ah=on * c["ah"],
            first_hour=c["hour"].where(on == 1),
            last_hour=c["hour"],
        )
        g = c.groupby(["user", "day"])
        parts.append(
            g.agg(
                logon_n=("logon_n", "sum"),
                logoff_n=("logoff_n", "sum"),
                logon_ah=("logon_ah", "sum"),
                first_hour=("first_hour", "min"),
                last_hour=("last_hour", "max"),
            )
        )
        pcs.append(c[["user", "day", "pc"]].drop_duplicates())
    how = {
        "logon_n": "sum",
        "logoff_n": "sum",
        "logon_ah": "sum",
        "first_hour": "min",
        "last_hour": "max",
    }
    out = _combine(parts, how)
    if pcs:
        pc_n = pd.concat(pcs).drop_duplicates().groupby(["user", "day"]).size()
        out["pc_n"] = pc_n.reindex(out.index).fillna(0)
    else:
        out["pc_n"] = 0
    return out


def agg_device(path: Path, chunksize: int = CHUNK_ROWS) -> pd.DataFrame:
    """USB connect / disconnect counts (connects after hours separately)."""
    parts = []
    for raw in iter_csv(path, ["date", "user", "activity"], chunksize):
        c = _stamp(raw)
        conn = (c["activity"] == "Connect").astype(np.int32)
        c = c.assign(usb_conn_n=conn, usb_conn_ah=conn * c["ah"], usb_disc_n=1 - conn)
        parts.append(c.groupby(["user", "day"])[DEVICE_COLS].sum())
    return _combine(parts, dict.fromkeys(DEVICE_COLS, "sum"))


def agg_file(path: Path, chunksize: int = CHUNK_ROWS) -> pd.DataFrame:
    """File copies to removable media (CERT r4.x ``file.csv``) and file types."""
    parts = []
    for raw in iter_csv(path, ["date", "user", "filename"], chunksize):
        c = _stamp(raw)
        name = c["filename"].str.lower()
        cols = {"file_n": np.ones(len(c), dtype=np.int32), "file_ah": c["ah"]}
        for col, exts in FILE_TYPES.items():
            cols[col] = name.str.endswith(exts).astype(np.int32)
        c = c.assign(**cols)
        parts.append(c.groupby(["user", "day"])[FILE_COLS].sum())
    return _combine(parts, dict.fromkeys(FILE_COLS, "sum"))


def _n_addresses(s: pd.Series) -> pd.Series:
    return s.str.count(";") + (s.str.len() > 0).astype(np.int32)


def _n_external(s: pd.Series) -> pd.Series:
    return s.str.count("@") - s.str.count(re.escape("@" + COMPANY_DOMAIN))


def agg_email(path: Path, chunksize: int = CHUNK_ROWS) -> pd.DataFrame:
    """Emails, external recipients, recipient count, size and attachments."""
    parts = []
    use = ["date", "user", "to", "cc", "bcc", "size", "attachments"]
    for raw in iter_csv(path, use, chunksize):
        c = _stamp(raw)
        rcpt = c["to"] + ";" + c["cc"] + ";" + c["bcc"]
        n_rcpt = _n_addresses(c["to"]) + _n_addresses(c["cc"]) + _n_addresses(c["bcc"])
        ext = (_n_external(rcpt) > 0).astype(np.int32)
        attach = pd.to_numeric(c["attachments"], errors="coerce").fillna(0)
        c = c.assign(
            email_n=1,
            email_ext_n=ext,
            email_recip_n=n_rcpt,
            email_bytes=pd.to_numeric(c["size"], errors="coerce").fillna(0),
            email_attach_n=attach,
            email_ext_attach_n=attach * ext,
            email_ah=c["ah"],
        )
        parts.append(c.groupby(["user", "day"])[EMAIL_COLS].sum())
    return _combine(parts, dict.fromkeys(EMAIL_COLS, "sum"))


def agg_http(path: Path, chunksize: int = CHUNK_ROWS) -> pd.DataFrame:
    """Web visits by category (job search, wikileaks, keylogger, cloud upload)."""
    parts = []
    for raw in iter_csv(path, ["date", "user", "url"], chunksize):
        c = _stamp(raw)
        url = c["url"].str.lower()
        cols: dict[str, object] = {"http_n": 1, "http_ah": c["ah"]}
        for col, pat in URL_PATTERNS.items():
            cols[col] = url.str.contains(pat, regex=True).astype(np.int32)
        c = c.assign(**cols)
        parts.append(c.groupby(["user", "day"])[HTTP_COLS].sum())
        del raw, c, url
    return _combine(parts, dict.fromkeys(HTTP_COLS, "sum"))


# ------------------------------------------------------------- static tables


def load_psychometric(path: Path) -> pd.DataFrame:
    """OCEAN scores per user (static)."""
    df = pd.read_csv(path)
    return df.set_index("user_id")[OCEAN].astype(np.float32)


def _codes(values: pd.Series) -> pd.Series:
    cats = sorted(values.dropna().astype(str).unique())
    mapping = {c: i for i, c in enumerate(cats)}
    return values.astype(str).map(mapping).fillna(-1).astype(np.int32)


def load_ldap(ldap_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (static per-user org table, per-user last-month table).

    The org table holds role / department / functional unit codes and the role
    name (for peer groups). ``last_month`` is the user's last LDAP month when
    the user disappears before the final snapshot (a leaver), else NaT.
    """
    files = sorted(ldap_dir.glob("*.csv"))
    rows = []
    for f in files:
        month = pd.Timestamp(f.stem + "-01")
        m = pd.read_csv(f, dtype=str, keep_default_na=False)
        m["month"] = month
        rows.append(m)
    ldap = pd.concat(rows, ignore_index=True)
    last = ldap.groupby("user_id")["month"].max()
    final = ldap["month"].max()
    leaver_month = last.where(last < final)
    org = ldap.sort_values("month").groupby("user_id").first()
    dept = org.get("department", pd.Series("", index=org.index))
    func = org.get("functional_unit", pd.Series("", index=org.index))
    static = pd.DataFrame(
        {
            "role": org["role"],
            "role_code": _codes(org["role"]),
            "dept_code": _codes(dept),
            "func_code": _codes(func),
        }
    )
    return static, leaver_month.rename("leaver_month").to_frame()


# ------------------------------------------------------------- feature table


@dataclass
class ReleasePaths:
    """File locations of one CERT release directory."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def file(self, name: str) -> Path:
        return self.root / name

    @property
    def ldap(self) -> Path:
        return self.root / "LDAP"


def merge_daily(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join per-source user-day aggregates; missing activity -> 0."""
    df = pd.concat(list(parts), axis=1)
    for col in COUNT_COLS:
        if col not in df.columns:
            df[col] = 0.0
    hours = df[["first_hour", "last_hour"]]
    df[COUNT_COLS] = df[COUNT_COLS].fillna(0)
    df[["first_hour", "last_hour"]] = hours.fillna(-1)
    return df.reset_index().astype({c: np.float32 for c in COUNT_COLS})


def add_baselines(
    df: pd.DataFrame, cols: Sequence[str] = BASELINE_COLS
) -> pd.DataFrame:
    """Mean and z-score of each col vs the user's PREVIOUS 30 days (closed-left)."""
    df = df.sort_values(["user", "day"]).reset_index(drop=True)
    roll = (
        df.set_index("day")
        .groupby("user")[list(cols)]
        .rolling("30D", closed="left", min_periods=1)
    )
    mean = roll.mean().reset_index(drop=True)
    std = roll.std(ddof=0).reset_index(drop=True)
    for c in cols:
        m = mean[c].fillna(0).to_numpy(np.float32)
        s = std[c].fillna(0).to_numpy(np.float32)
        df[f"{c}_mean30"] = m
        df[f"{c}_z30"] = (df[c].to_numpy(np.float32) - m) / (s + 1.0)
    return df


def add_window7(df: pd.DataFrame) -> pd.DataFrame:
    """Number of the last 7 calendar days (incl. today) with the given activity."""
    df = df.sort_values(["user", "day"]).reset_index(drop=True)
    flags = pd.DataFrame(
        {k: (df[v] > 0).astype(np.float32) for k, v in WINDOW7_COLS.items()}
    )
    flags["user"], flags["day"] = df["user"], df["day"]
    roll = (
        flags.set_index("day").groupby("user")[list(WINDOW7_COLS)].rolling("7D").sum()
    )
    vals = roll.reset_index(drop=True)
    for k in WINDOW7_COLS:
        df[k] = vals[k].to_numpy(np.float32)
    return df


def add_peer(df: pd.DataFrame, cols: Sequence[str] = PEER_COLS) -> pd.DataFrame:
    """z-score vs the user's role peers on the same day."""
    g = df.groupby(["role", "day"])[list(cols)]
    mean, std = g.transform("mean"), g.transform("std").fillna(0)
    for c in cols:
        df[f"{c}_peerz"] = ((df[c] - mean[c]) / (std[c] + 1.0)).astype(np.float32)
    return df


def add_static(
    df: pd.DataFrame,
    psych: pd.DataFrame,
    org: pd.DataFrame,
    leavers: pd.DataFrame,
) -> pd.DataFrame:
    """Attach OCEAN, org codes, HR notice flag and calendar fields."""
    df = df.join(psych, on="user").join(org, on="user").join(leavers, on="user")
    for c in OCEAN:
        df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)
    df["role"] = df["role"].fillna("unknown")
    for c in ("role_code", "dept_code", "func_code"):
        df[c] = df[c].fillna(-1)
    month = df["day"].dt.to_period("M").dt.to_timestamp()
    df["hr_leaving_soon"] = (month == df["leaver_month"]).astype(np.float32)
    df["weekday"] = df["day"].dt.dayofweek.astype(np.float32)
    df["is_weekend"] = (df["weekday"] >= 5).astype(np.float32)
    return df.drop(columns=["leaver_month"])


def finish_features(
    daily: pd.DataFrame, psych: pd.DataFrame, org: pd.DataFrame, leavers: pd.DataFrame
) -> pd.DataFrame:
    """All derived features on top of the merged per-source counts."""
    df = add_static(daily, psych, org, leavers)
    df = add_baselines(df)
    df = add_window7(df)
    df = add_peer(df)
    cols = feature_columns()
    df[cols] = df[cols].astype(np.float32)
    return df.sort_values(["day", "user"]).reset_index(drop=True)


def build_features(
    release_dir: Path, cache_dir: Path | None = None, log: object = print
) -> pd.DataFrame:
    """Stream every source of one release into the user-day feature table.

    Per-source aggregates are cached as parquet in ``cache_dir`` so a crashed or
    interrupted run resumes without re-reading the 14 GB http log.
    """
    paths = ReleasePaths(release_dir)
    readers = {
        "logon": agg_logon,
        "device": agg_device,
        "file": agg_file,
        "email": agg_email,
        "http": agg_http,
    }
    parts = []
    for name, fn in readers.items():
        tag = f"_{URL_RULES_VERSION}" if name == "http" else ""
        cached = cache_dir / f"agg_{name}{tag}.parquet" if cache_dir else None
        if cached is not None and cached.exists():
            parts.append(pd.read_parquet(cached))
            continue
        if callable(log):
            log(f"aggregating {name}.csv ...")
        part = fn(paths.file(f"{name}.csv"))
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            part.to_parquet(cached)
        parts.append(part)
    daily = merge_daily(parts)
    psych = load_psychometric(paths.file("psychometric.csv"))
    org, leavers = load_ldap(paths.ldap)
    return finish_features(daily, psych, org, leavers)


# ------------------------------------------------------------------- labels


def load_insiders(answers_dir: Path, release: str = "4.2") -> pd.DataFrame:
    """Insider list of one release: user, scenario, details file, start, end."""
    ins = pd.read_csv(Path(answers_dir) / "insiders.csv", dtype=str)
    ins = ins[ins["dataset"].str.strip() == release].copy()
    ins["scenario"] = ins["scenario"].astype(int)
    ins["start"] = pd.to_datetime(ins["start"], format=DATE_FMT)
    ins["end"] = pd.to_datetime(ins["end"], format=DATE_FMT)
    return ins.reset_index(drop=True)


def _detail_path(answers_dir: Path, release: str, row: pd.Series) -> Path:
    sub = Path(answers_dir) / f"r{release}-{row['scenario']}" / str(row["details"])
    return sub if sub.exists() else Path(answers_dir) / str(row["details"])


def malicious_days(answers_dir: Path, release: str = "4.2") -> pd.DataFrame:
    """(user, day) pairs that contain at least one answers event of that insider."""
    ins = load_insiders(answers_dir, release)
    out = []
    for _, row in ins.iterrows():
        path = _detail_path(answers_dir, release, row)
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                f = line.split(",", 5)
                if len(f) < 4 or f[3].strip() != row["user"]:
                    continue
                ts = pd.to_datetime(f[2], format=DATE_FMT, errors="coerce")
                if pd.notna(ts):
                    out.append((row["user"], ts.normalize(), int(row["scenario"])))
    days = pd.DataFrame(out, columns=["user", "day", "scenario"]).drop_duplicates()
    return days.reset_index(drop=True)


def attach_labels(df: pd.DataFrame, mal: pd.DataFrame) -> pd.DataFrame:
    """Add ``label`` (1 = malicious user-day) and ``insider`` (user is an insider)."""
    key = mal[["user", "day"]].drop_duplicates().assign(label=1)
    df = df.drop(columns=["label"], errors="ignore").merge(
        key, on=["user", "day"], how="left"
    )
    df["label"] = df["label"].fillna(0).astype(np.int8)
    df["insider"] = df["user"].isin(set(mal["user"])).astype(np.int8)
    return df
