"""Users, roles and permissions (owner 2026-10-16, enterprise).

Every dashboard user has one role; every role is a set of permissions. Only users
whose role has the ``users`` permission - the admin role - may create or remove
users, change usernames and passwords (their own included), assign roles and edit
what each role may do. The admin role always keeps every permission and the last
admin cannot be removed or demoted, so the SOC can never be locked out.
Passwords are stored as salted PBKDF2 hashes (src/soc/auth.py). Every change is
written to the audit log (src/soc/audit.py).

The first admin: the existing dashboard password (if one was set) becomes the user
``admin``; otherwise the dashboard asks for an admin name and password on first open.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

PERMISSIONS: dict[str, str] = {
    "view": "See alerts, incidents, search, reports and results",
    "review": "Mark alerts Real / False alarm and rate answers (teaches the models)",
    "investigate": "Analyse uploaded logs, check URLs / files, test datasets",
    "rules": "Own detection rules, allow-list and sensitivity",
    "models": "Add datasets, retrain, evolve / freeze models, approve code speed-ups",
    "settings": "SIEM connection, performance, places & backup, updates, retention",
    "collection": "Agent keys and API tokens (central collection, SIEM / SOAR access)",
    "reports": "Edit the global report template and report schedules (anyone who can view makes reports)",
    "users": "Users, roles, permissions and the audit log (admin only)",
}
DEFAULT_ROLES: dict[str, list[str]] = {
    "admin": list(PERMISSIONS),
    "analyst": ["view", "review", "investigate", "rules"],
    "viewer": ["view"],
}
_NAME = re.compile(r"^[A-Za-z0-9._-]{3,32}$")


def path() -> Path:
    from src.soc import engine

    return Path(engine.EVENTS).with_name("users.json")


def load() -> dict[str, Any]:
    """{"users": {name: {...}}, "roles": {role: [permissions]}} (first admin migrated
    from the old single dashboard password when there is one)."""
    try:
        d: dict[str, Any] = json.loads(path().read_text("utf-8"))
    except (OSError, ValueError):
        d = {"users": {}, "roles": {k: list(v) for k, v in DEFAULT_ROLES.items()}}
        from src.soc import auth

        old = auth.stored_hash()
        if old:
            d["users"]["admin"] = {"hash": old, "role": "admin", "created": time.strftime("%Y-%m-%d %H:%M"),
                                   "note": "from the earlier dashboard password"}  # fmt: skip
            _write(d)
    d.setdefault("roles", {k: list(v) for k, v in DEFAULT_ROLES.items()})
    d["roles"]["admin"] = list(PERMISSIONS)  # never reduced
    return d


def _write(d: dict[str, Any]) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1), "utf-8")
    tmp.replace(p)


def needs_first_admin() -> bool:
    return not load()["users"]


def permissions_of(name: str) -> set[str]:
    d = load()
    u = d["users"].get(name)
    if not u or u.get("disabled"):
        return set()
    return set(d["roles"].get(u["role"], []))


def can(name: str, permission: str) -> bool:
    return permission in permissions_of(name)


LOCK_FAILS, LOCK_WINDOW_S, LOCK_S = 5, 300.0, 900.0


def locked(name: str, now: float | None = None) -> bool:
    """5 failed sign-ins for this username within 5 minutes lock it for 15 minutes
    (pentest 2026-10-17: the dashboard's back-off was per browser session only)."""
    from src.soc import audit

    now = time.time() if now is None else now
    fails = [float(r.get("ts") or 0) for r in audit.read(last=500)
             if r.get("action") == "sign_in" and r.get("who") == name and not r.get("ok")
             and now - float(r.get("ts") or 0) < LOCK_WINDOW_S + LOCK_S]  # fmt: skip
    for k in range(len(fails) - LOCK_FAILS + 1):
        window = fails[k : k + LOCK_FAILS]
        if window[-1] - window[0] <= LOCK_WINDOW_S and now - window[-1] < LOCK_S:
            return True
    return False


def authenticate(name: str, password: str) -> bool:
    from src.soc import audit, auth

    if locked(name):
        audit.record(name or "?", "sign_in_locked", ok=False)
        return False
    u = load()["users"].get(name)
    ok = bool(u) and not u.get("disabled") and auth.verify(password, u["hash"])
    audit.record(name or "?", "sign_in", ok=ok)
    return ok


def _require_admin(by: str) -> None:
    if not can(by, "users"):
        raise PermissionError("only an admin may change users, roles or permissions")


def _admins(d: dict[str, Any]) -> list[str]:
    return [
        n
        for n, u in d["users"].items()
        if u["role"] == "admin" and not u.get("disabled")
    ]


def _check_password(pw: str) -> None:
    from src.soc import auth

    if len(pw) < auth.MIN_LENGTH:
        raise ValueError(f"password too short - at least {auth.MIN_LENGTH} characters")


def create_first_admin(name: str, password: str) -> None:
    """Only when no user exists yet (first open of a new install)."""
    from src.soc import audit, auth

    d = load()
    if d["users"]:
        raise PermissionError("an admin exists already")
    if not _NAME.match(name):
        raise ValueError("username: 3-32 letters, digits, dot, dash or underscore")
    _check_password(password)
    d["users"][name] = {
        "hash": auth.make_hash(password),
        "role": "admin",
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    _write(d)
    audit.record(name, "first_admin_created")


def add_user(by: str, name: str, password: str, role: str) -> None:
    from src.soc import audit, auth

    _require_admin(by)
    d = load()
    if not _NAME.match(name):
        raise ValueError("username: 3-32 letters, digits, dot, dash or underscore")
    if name in d["users"]:
        raise ValueError(f"user {name!r} exists already")
    if role not in d["roles"]:
        raise ValueError(f"unknown role {role!r}")
    _check_password(password)
    d["users"][name] = {
        "hash": auth.make_hash(password),
        "role": role,
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    _write(d)
    audit.record(by, "user_added", {"user": name, "role": role})


def set_password(by: str, name: str, password: str) -> None:
    from src.soc import audit, auth

    _require_admin(by)
    d = load()
    if name not in d["users"]:
        raise ValueError(f"no user {name!r}")
    _check_password(password)
    d["users"][name]["hash"] = auth.make_hash(password)
    _write(d)
    audit.record(by, "password_changed", {"user": name})


def rename(by: str, old: str, new: str) -> None:
    from src.soc import audit

    _require_admin(by)
    d = load()
    if old not in d["users"]:
        raise ValueError(f"no user {old!r}")
    if not _NAME.match(new) or new in d["users"]:
        raise ValueError(
            "new username taken or not allowed (3-32 letters, digits, . - _)"
        )
    d["users"][new] = d["users"].pop(old)
    _write(d)
    audit.record(by, "user_renamed", {"from": old, "to": new})


def set_role(by: str, name: str, role: str) -> None:
    from src.soc import audit

    _require_admin(by)
    d = load()
    if name not in d["users"] or role not in d["roles"]:
        raise ValueError("unknown user or role")
    if d["users"][name]["role"] == "admin" and role != "admin" and _admins(d) == [name]:
        raise ValueError("the last admin cannot be demoted")
    d["users"][name]["role"] = role
    _write(d)
    audit.record(by, "role_changed", {"user": name, "role": role})


def set_disabled(by: str, name: str, disabled: bool) -> None:
    from src.soc import audit

    _require_admin(by)
    d = load()
    if name not in d["users"]:
        raise ValueError(f"no user {name!r}")
    if disabled and _admins(d) == [name]:
        raise ValueError("the last admin cannot be disabled")
    d["users"][name]["disabled"] = bool(disabled)
    _write(d)
    audit.record(by, "user_disabled" if disabled else "user_enabled", {"user": name})


def remove_user(by: str, name: str) -> None:
    from src.soc import audit

    _require_admin(by)
    d = load()
    if name not in d["users"]:
        raise ValueError(f"no user {name!r}")
    if _admins(d) == [name]:
        raise ValueError("the last admin cannot be removed")
    d["users"].pop(name)
    _write(d)
    audit.record(by, "user_removed", {"user": name})


def set_role_permissions(by: str, role: str, perms: list[str]) -> None:
    """Create a role or change what it may do (the admin role is fixed)."""
    from src.soc import audit

    _require_admin(by)
    if role == "admin":
        raise ValueError("the admin role always has every permission")
    if not _NAME.match(role):
        raise ValueError("role name: 3-32 letters, digits, dot, dash or underscore")
    if bad := [p for p in perms if p not in PERMISSIONS]:
        raise ValueError(f"unknown permission(s): {', '.join(bad)}")
    if "users" in perms:
        raise ValueError("'users' (managing users) belongs to the admin role only")
    d = load()
    d["roles"][role] = sorted(set(perms), key=list(PERMISSIONS).index)
    _write(d)
    audit.record(
        by, "role_permissions", {"role": role, "permissions": d["roles"][role]}
    )


def delete_role(by: str, role: str) -> None:
    from src.soc import audit

    _require_admin(by)
    d = load()
    if role in ("admin",) or role not in d["roles"]:
        raise ValueError("cannot delete that role")
    if any(u["role"] == role for u in d["users"].values()):
        raise ValueError("users still have this role - give them another role first")
    d["roles"].pop(role)
    _write(d)
    audit.record(by, "role_deleted", {"role": role})
