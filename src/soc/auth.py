"""Dashboard login (owner decision 2026-10-14).

Only a salted PBKDF2-SHA256 hash of the password is stored, in ``.env`` as
``DASHBOARD_PASSWORD_HASH`` (set with ``python -m scripts.set_dashboard_password``;
the password itself is typed hidden and never written anywhere). The dashboard
also listens on localhost only, so the login protects against other users of
this PC and the network option.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets

from utils.paths import PROJECT_ROOT

ENV = PROJECT_ROOT / ".env"
KEY = "DASHBOARD_PASSWORD_HASH"
ITERATIONS = 300_000
MIN_LENGTH = 10


def make_hash(password: str, iterations: int = ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify(password: str, stored: str) -> bool:
    try:
        algo, it, salt, digest = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(it))
    return hmac.compare_digest(dk.hex(), digest)


def stored_hash() -> str | None:
    """The configured hash (environment first, then .env), or None."""
    if os.environ.get(KEY):
        return os.environ[KEY]
    try:
        for line in ENV.read_text("utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == KEY and v.strip():
                return v.strip()
    except OSError:
        pass
    return None


def save_hash(h: str) -> None:
    """Write/replace the hash line in .env, keeping every other line."""
    lines = ENV.read_text("utf-8").splitlines() if ENV.is_file() else []
    lines = [ln for ln in lines if ln.partition("=")[0].strip() != KEY]
    lines.append(f"{KEY}={h}")
    ENV.write_text("\n".join(lines) + "\n", "utf-8")
