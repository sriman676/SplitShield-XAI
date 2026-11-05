"""Dashboard smoke: every page renders without an exception (headless AppTest).

2026-10-16 redesign: pages live in a grouped st.navigation; each one is opened through
its ``?page=<address>`` link (AppTest cannot click a callable page). Since the
enterprise users / roles step, every page is opened signed in as an admin of the
temporary user store (tests/conftest.py), and sign-in itself is tested here."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
DASH = Path(__file__).resolve().parent.parent / "dashboard.py"
# url paths from the NAV table (read from the source: importing runs the app)
PATHS = re.findall(r'\(\s*"[^"]+",\s*"[^"]+",\s*page_\w+,\s*"[^"]+",\s*"([^"]+)",?\s*\)',
                   DASH.read_text("utf-8"))  # fmt: skip
PW = "test-password-1"


def app(path: str = "", user: str | None = "admin", timeout: int = 120) -> Any:
    """An AppTest on ``?page=path``, signed in as ``user`` (admin made on first use)."""
    from src.soc import users

    if users.needs_first_admin():
        users.create_first_admin("admin", PW)
    at = st_testing.AppTest.from_file(str(DASH), default_timeout=timeout)
    if path:
        at.query_params["page"] = path
    if user:
        at.session_state["user"] = user
    return at


def test_navigation_lists_every_page() -> None:
    assert len(PATHS) == 25 and len(set(PATHS)) == 25


@pytest.mark.parametrize("path", PATHS)
def test_every_page_renders(path: str) -> None:
    at = app(path)
    at.run()
    assert not at.exception, f"{path}: {at.exception}"


def test_first_open_asks_for_the_admin() -> None:
    at = st_testing.AppTest.from_file(str(DASH), default_timeout=120)
    at.run()
    assert not at.exception and any("admin" in b.label.lower() for b in at.button)
    at.text_input[0].set_value("boss")
    at.text_input[1].set_value(PW)
    at.text_input[2].set_value(PW)
    at.button[0].click().run()
    from src.soc import users

    assert users.can("boss", "users") and at.session_state["user"] == "boss"


def test_sign_in_and_wrong_password(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    monkeypatch.setattr(time, "sleep", lambda _s: None)  # skip the back-off wait
    at = app(user=None)
    at.run()
    at.text_input[0].set_value("admin")
    at.text_input[1].set_value("wrong-password")
    at.button[0].click().run()
    assert any("Wrong username or password" in e.value for e in at.error)
    at.text_input[0].set_value("admin")
    at.text_input[1].set_value(PW)
    at.button[0].click().run()
    assert at.session_state["user"] == "admin" and not at.exception


def test_viewer_sees_only_view_pages() -> None:
    from src.soc import users

    app()  # makes the admin
    users.add_user("admin", "vic", PW, "viewer")
    at = app("users", user="vic")  # asks for an admin page...
    at.run()
    assert not at.exception
    assert not any("Users & roles" in h.value for h in at.header)  # ...gets Home instead
    users.set_disabled("admin", "vic", True)
    at = app("home", user="vic")  # disabled while signed in -> back to sign-in
    at.run()
    assert any(b.label == "Sign in" for b in at.button)
