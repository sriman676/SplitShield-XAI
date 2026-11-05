"""Dashboard login: salted PBKDF2 hash only, never the password (src/soc/auth.py)."""

from __future__ import annotations

from pathlib import Path

from src.soc import auth


def test_hash_verify_and_salting() -> None:
    h = auth.make_hash("correct horse battery", iterations=1000)
    assert auth.verify("correct horse battery", h)
    assert not auth.verify("wrong password!!", h)
    assert "correct horse" not in h
    assert h != auth.make_hash("correct horse battery", iterations=1000)  # salted
    assert not auth.verify("x", "garbage")


def test_save_keeps_other_env_lines(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    env = tmp_path / ".env"
    env.write_text("NUM_WORKERS=4\nDASHBOARD_PASSWORD_HASH=old\n", "utf-8")
    monkeypatch.setattr(auth, "ENV", env)
    monkeypatch.delenv(auth.KEY, raising=False)
    auth.save_hash("pbkdf2_sha256$1$aa$bb")
    lines = env.read_text("utf-8").splitlines()
    assert lines == ["NUM_WORKERS=4", "DASHBOARD_PASSWORD_HASH=pbkdf2_sha256$1$aa$bb"]
    assert auth.stored_hash() == "pbkdf2_sha256$1$aa$bb"
