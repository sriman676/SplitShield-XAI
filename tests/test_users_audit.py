"""Users / roles / permissions (admin-only changes) and the tamper-evident audit log."""

from __future__ import annotations

import json

import pytest

from src.soc import audit, users

PW = "correct-horse-1"


@pytest.fixture()
def admin() -> str:
    users.create_first_admin("boss", PW)
    return "boss"


def test_first_admin_then_only_admin_manages(admin: str) -> None:
    assert users.authenticate("boss", PW) and not users.authenticate(
        "boss", "wrong-password"
    )
    users.add_user(admin, "ana", PW, "analyst")
    users.add_user(admin, "vic", PW, "viewer")
    assert users.can("ana", "rules") and not users.can("ana", "users")
    assert users.can("vic", "view") and not users.can("vic", "review")
    for act in (lambda: users.add_user("ana", "eve", PW, "admin"),
                lambda: users.set_password("ana", "boss", "hacked-password-1"),
                lambda: users.set_role("vic", "vic", "admin"),
                lambda: users.set_role_permissions("ana", "analyst", ["view", "users"])):  # fmt: skip
        with pytest.raises(PermissionError):
            act()
    with pytest.raises(PermissionError):
        users.create_first_admin("other", PW)  # only on an empty install


def test_admin_can_change_everything_but_never_lock_out(admin: str) -> None:
    users.rename(admin, "boss", "chief")
    assert users.authenticate("chief", PW)  # same password after the rename
    users.set_password("chief", "chief", "new-password-12")
    assert users.authenticate("chief", "new-password-12")
    for act in (lambda: users.set_role("chief", "chief", "viewer"),
                lambda: users.remove_user("chief", "chief"),
                lambda: users.set_disabled("chief", "chief", True)):  # fmt: skip
        with pytest.raises(ValueError):
            act()
    with pytest.raises(ValueError):
        users.set_role_permissions("chief", "admin", ["view"])  # admin role is fixed


def test_custom_role_and_permission_edit(admin: str) -> None:
    users.set_role_permissions(admin, "responder", ["view", "review", "investigate"])
    users.add_user(admin, "resp", PW, "responder")
    assert users.can("resp", "investigate") and not users.can("resp", "rules")
    users.set_role_permissions(admin, "responder", ["view"])
    assert not users.can("resp", "investigate")
    with pytest.raises(ValueError):
        users.set_role_permissions(admin, "responder", ["users"])  # admin only
    with pytest.raises(ValueError):
        users.delete_role(admin, "responder")  # still assigned
    users.set_disabled(admin, "resp", True)
    assert users.permissions_of("resp") == set() and not users.authenticate("resp", PW)


def test_audit_chain_detects_tampering(admin: str) -> None:
    users.add_user(admin, "ana", PW, "analyst")
    users.set_role(admin, "ana", "viewer")
    assert audit.verify()["ok"] and audit.verify()["entries"] >= 3
    p = audit.path()
    lines = p.read_text("utf-8").splitlines()
    e = json.loads(lines[1])
    e["details"]["role"] = "admin"  # someone edits history
    lines[1] = json.dumps(e)
    p.write_text("\n".join(lines) + "\n", "utf-8")
    res = audit.verify()
    assert not res["ok"] and res["broken_at"] == 2


def test_username_lockout_after_repeated_failures(admin: str) -> None:
    """Pentest 2026-10-17: guessing from new browser sessions must still be slowed."""
    for _ in range(users.LOCK_FAILS):
        assert not users.authenticate("boss", "wrong-password")
    assert users.locked("boss")
    assert not users.authenticate("boss", PW)  # even the right password, while locked
    assert any(r["action"] == "sign_in_locked" for r in audit.read())
    import time

    assert not users.locked("boss", now=time.time() + users.LOCK_S + 1)
