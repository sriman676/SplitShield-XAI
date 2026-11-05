"""Phishing page model (owner 2026-10-24): uploaded / attached HTML only, never fetched."""

from __future__ import annotations

from email.message import EmailMessage

import pytest

from src.mlareas import phishing_page

LOGIN = ("<html><head><title>Account verification</title></head><body><form action='http://203.0.113.9/p.php'>"
         "<input type='email' name='user'><input type='password' name='pass'></form>"
         "<a href='#'>help</a><a href='#'>terms</a></body></html>")  # fmt: skip


def test_email_html_parts_are_found() -> None:
    m = EmailMessage()
    m["Subject"] = "hello"
    m.set_content("plain text")
    m.add_alternative("<html><body><p>hi</p></body></html>", subtype="html")
    m.add_attachment(LOGIN.encode(), maintype="text", subtype="html", filename="invoice.html")
    parts = phishing_page.html_from_email(m.as_bytes())
    assert len(parts) == 2 and any(n == "invoice.html" for n, _ in parts)


def test_facts_name_what_the_page_asks_for() -> None:
    facts = phishing_page.page_facts(LOGIN, "http://pages.example.net/x")
    assert "asks for a password" in facts and "a form posts to a script or an e-mail address" in facts


def test_deployed_model_scores_an_upload() -> None:
    if not phishing_page.BUNDLE.is_file():
        pytest.skip("page model not deployed")
    r = phishing_page.check_page(LOGIN, "http://pages.example.net/x")
    assert 0.0 <= r["probability"] <= 1.0 and r["verdict"] in ("phishing", "legitimate")


def test_well_known_sites_lose_only_the_suspicious_warning() -> None:
    from src.mlareas import phishing

    assert "google.com" in phishing.top_domains()
    if not (phishing.common.area_dir("phishing") / "current" / "model.joblib").is_file():
        pytest.skip("URL model not deployed")
    r = phishing.check_urls(["https://www.google.com/"])[0]
    assert r["verdict"] in ("legitimate", "phishing")  # never "suspicious" for a top-10k domain
