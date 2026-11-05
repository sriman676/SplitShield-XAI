"""Fast tests for the insider-threat area on tiny synthetic CERT-like CSVs (owner 2026-10-15)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.mlareas import insider as ins
from src.mlareas import insider_features as feats

USERS = ["AAA0001", "BBB0002", "CCC0003", "DDD0004"]
DAYS = pd.date_range("2010-01-04", periods=40, freq="D")


def _ts(day: pd.Timestamp, hour: int) -> str:
    return (day + pd.Timedelta(hours=hour)).strftime(feats.DATE_FMT)


def make_release(root: Path, spike_day: int = 35) -> Path:
    """Four users, 40 days; user AAA0001 copies files to USB after hours on spike_day."""
    rel = root / "r4.2"
    (rel / "LDAP").mkdir(parents=True)
    logon, device, file_, email, http = [], [], [], [], []
    k = 0
    for d_i, day in enumerate(DAYS):
        for u_i, u in enumerate(USERS):
            pc = f"PC-{u_i}"
            k += 1
            logon.append((f"{{L{k}}}", _ts(day, 8), u, pc, "Logon"))
            logon.append((f"{{O{k}}}", _ts(day, 17), u, pc, "Logoff"))
            email.append(
                (
                    f"{{E{k}}}",
                    _ts(day, 10),
                    u,
                    pc,
                    "x@dtaa.com",
                    "",
                    "",
                    f"{u}@dtaa.com",
                    "1000",
                    "0",
                    "hi",
                )
            )
            http.append((f"{{H{k}}}", _ts(day, 11), u, pc, "http://news.com/a", "text"))
            if u == "AAA0001" and d_i == spike_day:
                logon.append((f"{{LA{k}}}", _ts(day, 23), u, pc, "Logon"))
                device.append((f"{{D{k}}}", _ts(day, 23), u, pc, "Connect"))
                device.append((f"{{DD{k}}}", _ts(day, 23), u, pc, "Disconnect"))
                for j in range(5):
                    file_.append(
                        (f"{{F{k}-{j}}}", _ts(day, 23), u, pc, f"R:\\x{j}.docx", "c")
                    )
                email.append(
                    (
                        f"{{EX{k}}}",
                        _ts(day, 12),
                        u,
                        pc,
                        "a@gmail.com;b@dtaa.com",
                        "",
                        "",
                        f"{u}@dtaa.com",
                        "5000",
                        "2",
                        "x",
                    )
                )
                http.append(
                    (
                        f"{{HJ{k}}}",
                        _ts(day, 12),
                        u,
                        pc,
                        "http://www.monster.com/jobs",
                        "t",
                    )
                )
                http.append(
                    (f"{{HW{k}}}", _ts(day, 23), u, pc, "http://wikileaks.org/up", "t")
                )
    cols = {
        "logon.csv": (["id", "date", "user", "pc", "activity"], logon),
        "device.csv": (["id", "date", "user", "pc", "activity"], device),
        "file.csv": (["id", "date", "user", "pc", "filename", "content"], file_),
        "email.csv": (
            [
                "id",
                "date",
                "user",
                "pc",
                "to",
                "cc",
                "bcc",
                "from",
                "size",
                "attachments",
                "content",
            ],
            email,
        ),
        "http.csv": (["id", "date", "user", "pc", "url", "content"], http),
    }
    for name, (header, rows) in cols.items():
        pd.DataFrame(rows, columns=header).to_csv(rel / name, index=False)
    psych = pd.DataFrame(
        {
            "employee_name": USERS,
            "user_id": USERS,
            "O": 1,
            "C": 2,
            "E": 3,
            "A": 4,
            "N": 5,
        }
    )
    psych.to_csv(rel / "psychometric.csv", index=False)
    org = {"employee_name": USERS, "user_id": USERS, "email": "", "role": ["Eng", "Eng", "Sales", "Sales"],
           "business_unit": "1", "functional_unit": "F", "department": "D", "team": "T", "supervisor": ""}  # fmt: skip
    pd.DataFrame(org).to_csv(rel / "LDAP" / "2010-01.csv", index=False)
    feb = pd.DataFrame(org)
    feb[feb["user_id"] != "AAA0001"].to_csv(
        rel / "LDAP" / "2010-02.csv", index=False
    )  # AAA leaves
    return rel


def make_answers(root: Path, spike_day: int = 35) -> Path:
    ans = root / "answers"
    (ans / "r4.2-1").mkdir(parents=True)
    day = DAYS[spike_day]
    pd.DataFrame(
        [("4.2", 1, "r4.2-1-AAA0001.csv", "AAA0001", _ts(day, 23), _ts(day, 23)),
         ("4.1", 1, "r4.1-1.csv", "ZZZ0009", _ts(day, 1), _ts(day, 2))],
        columns=["dataset", "scenario", "details", "user", "start", "end"],
    ).to_csv(ans / "insiders.csv", index=False)  # fmt: skip
    lines = [
        f"device,{{D1}},{_ts(day, 23)},AAA0001,PC-0,Connect",
        f"email,{{E9}},{_ts(day, 9)},BBB0002,PC-1,a@dtaa.com,,,b@dtaa.com,10,0,reply",
    ]
    (ans / "r4.2-1" / "r4.2-1-AAA0001.csv").write_text("\n".join(lines) + "\n")
    return ans


@pytest.fixture()
def release(tmp_path: Path) -> tuple[Path, Path]:
    return make_release(tmp_path), make_answers(tmp_path)


def test_feature_aggregation(release: tuple[Path, Path]) -> None:
    rel, ans = release
    df = feats.build_features(rel)
    assert set(feats.feature_columns()) <= set(df.columns)
    assert len(df) == len(USERS) * len(DAYS)
    row = df[(df["user"] == "AAA0001") & (df["day"] == DAYS[35])].iloc[0]
    assert row["usb_conn_n"] == 1 and row["usb_conn_ah"] == 1
    assert row["file_n"] == 5 and row["file_ah"] == 5 and row["file_doc"] == 5
    assert row["logon_n"] == 2 and row["logon_ah"] == 1
    assert row["http_job"] == 1 and row["http_wikileaks"] == 1
    assert row["email_ext_n"] == 1 and row["email_ext_attach_n"] == 2
    assert row["usb_conn_n_mean30"] == 0  # never used USB before
    assert row["file_n_z30"] == pytest.approx(5.0)
    # AAA0001 is absent from the Feb LDAP snapshot -> last month flagged
    jan = df[(df["user"] == "AAA0001") & (df["day"].dt.month == 1)]
    assert jan["hr_leaving_soon"].eq(1).all()
    assert df.loc[df["user"] == "BBB0002", "hr_leaving_soon"].eq(0).all()
    assert df["O"].eq(1).all()


def test_baselines_use_only_past(release: tuple[Path, Path]) -> None:
    rel, _ = release
    df = feats.build_features(rel)
    a = df[df["user"] == "AAA0001"].set_index("day")
    # the spike day itself is not in its own baseline, but is in the next day's
    assert a.loc[DAYS[35], "file_n_mean30"] == 0
    assert a.loc[DAYS[36], "file_n_mean30"] == pytest.approx(5 / 30)
    # changing the future must not change any earlier baseline feature
    rel2 = make_release(rel.parent.parent / "other", spike_day=39)
    b = feats.build_features(rel2)
    b = b[b["user"] == "AAA0001"].set_index("day")
    past_cols = [c for c in df.columns if c.endswith(("_mean30", "_z30"))]
    pd.testing.assert_frame_equal(
        a.loc[DAYS[:35], past_cols], b.loc[DAYS[:35], past_cols]
    )


def test_labels_only_insider_own_events(release: tuple[Path, Path]) -> None:
    rel, ans = release
    mal = feats.malicious_days(ans, "4.2")
    assert list(mal["user"]) == ["AAA0001"]  # BBB0002's reply row and r4.1 ignored
    df = feats.attach_labels(feats.build_features(rel), mal)
    assert df["label"].sum() == 1
    assert df.loc[df["label"] == 1, "day"].iloc[0] == DAYS[35]


def test_splits() -> None:
    days = pd.Series(np.repeat(DAYS, 2))
    sp = ins.time_split(days)
    assert (
        days[sp == "train"].max()
        < days[sp == "val"].min()
        <= days[sp == "val"].max()
        < days[sp == "test"].min()
    )
    users = pd.Series([f"U{i}" for i in range(20)] * 2)
    us = ins.user_split(users, {"U0", "U1", "U2"}, seed=0)
    per_user = pd.DataFrame({"u": users, "s": us}).drop_duplicates()
    assert per_user["u"].is_unique  # a user sits in exactly one split
    assert set(us) == {"train", "val", "test"}


def _synthetic_table(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = feats.feature_columns()
    df = pd.DataFrame(
        rng.poisson(1.0, size=(n, len(cols))).astype(np.float32), columns=cols
    )
    y = (rng.random(n) < 0.08).astype(int)
    df.loc[y == 1, "usb_conn_n"] += 6
    df.loc[y == 1, "file_n"] += 20
    df["label"] = y
    return df


@pytest.fixture()
def store(tmp_path: Path) -> Path:
    df = _synthetic_table()
    cols = feats.feature_columns()
    tr, va = df.iloc[:400], df.iloc[400:]
    model = ins.make_model("logreg", 0, {"C": 0.1})
    model.fit(tr[cols].to_numpy(np.float32), tr["label"])
    b = ins.make_bundle(
        "logreg",
        model,
        {"C": 0.1},
        cols,
        va[cols].to_numpy(np.float32),
        va["label"].to_numpy(),
        tr[cols].to_numpy(np.float32),
    )
    ins.save_bundle(b, tmp_path)
    ins.save_store_data(tr, va, tmp_path)
    return tmp_path


def test_score_user_day_shape(store: Path) -> None:
    feats_in = {
        "usb_conn_n": 9,
        "usb_conn_ah": 9,
        "usb_conn_n_mean30": 0.3,
        "file_n": 214,
        "file_ah": 200,
        "file_n_mean30": 7,
    }
    out = ins.score_user_day(feats_in, root=store)
    assert set(out) >= {"risk", "verdict", "why"}
    assert out["verdict"] in {"high", "medium", "low"}
    assert all(line.startswith(f"{i}. ") for i, line in enumerate(out["why"], 1))
    text = " ".join(out["why"])
    assert "Verdict:" in text and "What to do:" in text
    assert (
        "WikiLeaks" not in text and "job-search" not in text
    )  # unsupported claims absent
    assert out["verdict"] == "high" and "USB" in text
    low = ins.score_user_day({"usb_conn_n": 0, "file_n": 0}, root=store)
    assert low["risk"] < out["risk"]


def test_improve_keep_and_rollback(store: Path) -> None:
    good = _synthetic_table(400, seed=1)
    res = ins.improve(good, root=store)
    assert res["promoted"] and (store / "insider" / "prev" / "model.joblib").exists()
    assert res["challenger_rows_per_s"] > 0 and res["champion_rows_per_s"] > 0
    before = ins.load_bundle(store)["trained_at"], ins.area_dir(store)
    bad = _synthetic_table(4000, seed=2)
    bad["label"] = 1 - bad["label"]  # poisoned labels -> worse on validation
    res2 = ins.improve(bad, root=store)
    assert not res2["promoted"] and res2["decision"] == "keep champion"
    assert ins.load_bundle(store)["trained_at"] == before[0]
