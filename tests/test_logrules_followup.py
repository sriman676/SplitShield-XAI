"""Linux / identity rules from the non-blind attack_data follow-up (2026-10-15):
each rule fires on the behaviour and stays quiet on the look-alike admin activity."""

from __future__ import annotations

import json

import pytest

from src.soc import logformats, logrules
from src.soc.parsers import parse_line

LINUX = ("linux", "process_creation", None)


def _hit(fields: dict, source: tuple = LINUX) -> list[str]:
    h = logrules.check(fields, source)
    return h[1] if h else []


# --- parser: auditd interpreted format + nested msg -----------------------------------


def test_interpreted_auditd_proctitle_is_parsed_and_checked() -> None:
    line = ("type=PROCTITLE msg=audit(02/20/2025 13:34:54.661:154635) : "
            "proctitle=sudo setcap cap_setuid+ep ./priv_esc")  # fmt: skip
    ev = logformats.parse_auditd(line)
    assert ev is not None
    assert ev["kind"] == "capability_added" and "T1548" in ev["techniques"]
    assert ev["ts"] > 1.7e9


def test_auditd_nested_msg_unit_is_a_field() -> None:
    line = ("type=SERVICE_STOP msg=audit(1722965459.146:15207): pid=1 uid=0 auid=4294967295 "
            "ses=4294967295 subj=unconfined msg='unit=sysmon comm=\"systemd\" "
            "exe=\"/usr/lib/systemd/systemd\" hostname=? addr=? terminal=? res=success'")  # fmt: skip
    ev = logformats.parse_auditd(line)
    assert ev is not None and ev["techniques"] == ["T1562"]
    benign = line.replace("unit=sysmon", "unit=atd")
    assert logformats.parse_auditd(benign)["techniques"] == []  # type: ignore[index]


def test_raw_auditd_still_parses() -> None:
    line = 'type=EXECVE msg=audit(1700000000.123:42): argc=2 a0="cat" a1="/etc/shadow"'
    ev = logformats.parse_auditd(line)
    assert ev is not None and "T1003" in ev["techniques"]


# --- Linux process rules ----------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "gdb -nx -ex !sh -ex quit",
        'node -e child_process.spawn("/bin/sh", {stdio: [0, 1, 2]})',
        "php -r system('/bin/sh');",
        'ruby -e exec "/bin/sh"',
        "openvpn --dev null --script-security 2 --up /bin/sh -c sh",
        "find . -exec /bin/sh \\; -quit",
        "python3 -c 'import pty;pty.spawn(\"/bin/bash\")'",
    ],
)
def test_shell_escapes(cmd: str) -> None:
    assert "T1548" in _hit({"CommandLine": cmd})


@pytest.mark.parametrize(
    "cmd",
    [
        "python3 -c 'import subprocess; print(1)'",  # interpreter, no shell
        "php -r 'echo phpversion();'",
        "find /var/log -name '*.gz' -mtime +7",
        "make -j8 install",
        "crontab -l",
        "systemctl restart auditd.service",  # restart reloads rules: normal admin
        "chmod 1777 /srv/shared",  # sticky bit, not setuid
        "apt install gdb",
    ],
)
def test_admin_look_alikes_stay_quiet(cmd: str) -> None:
    assert _hit({"CommandLine": cmd}) == []


@pytest.mark.parametrize(
    ("cmd", "tech"),
    [
        ("setcap cap_net_raw+ep ./bin", "T1548"),
        ("chmod u+s /tmp/x", "T1548"),
        ("chmod 4755 ./vuln", "T1548"),
        ("cat /etc/shadow", "T1003"),
        ("bash -c export HISTFILE=/dev/null; id", "T1070"),
        ("systemctl stop sysmon", "T1562"),
        ("gcc -shared -fPIC -Wl,-init,x -o lib.so x.c", "T1129"),
        ("./gdrive --refresh-token abc upload secret.zip", "T1567"),
        ("rclone copy /data remote:bucket", "T1567"),
        ("sudo insmod rootkit.ko", "T1547"),
        ("at now + 1 minute", "T1053"),
        ("/dev/shm/payload --init", "T1059"),
    ],
)
def test_linux_behaviours(cmd: str, tech: str) -> None:
    assert tech in _hit({"CommandLine": cmd, "Image": cmd.split(" ", 1)[0]})


def test_persistence_writes_skip_package_managers() -> None:
    f = {"TargetFilename": "/etc/update-motd.d/99-x", "Image": "/usr/bin/bash"}
    assert _hit(f, ("linux", "file_event", None)) == ["T1037"]
    f["Image"] = "/usr/bin/dpkg"
    assert _hit(f, ("linux", "file_event", None)) == []
    unit = {
        "TargetFilename": "/etc/systemd/system/evil.service",
        "Image": "/usr/bin/vim.basic",
    }
    assert _hit(unit, ("linux", "file_event", None)) == ["T1543"]


def test_document_app_spawning_shell() -> None:
    f = {
        "Image": "/usr/bin/dash",
        "ParentImage": "/usr/local/bin/gs",
        "CommandLine": "sh -c x",
    }
    assert _hit(f) == ["T1203"]


# --- syslog (linux_secure) ------------------------------------------------------------------


def test_sudo_root_shell_in_auth_log() -> None:
    line = ("Oct 15 10:00:00 host sudo:   alice : TTY=pts/0 ; PWD=/home/alice ; USER=root ; "
            "COMMAND=/bin/bash")  # fmt: skip
    ev = parse_line(line)
    assert ev is not None and "T1548" in ev["techniques"]
    normal = line.replace("COMMAND=/bin/bash", "COMMAND=/usr/bin/apt update")
    ev2 = parse_line(normal)
    assert ev2 is not None and "T1548" not in ev2["techniques"]


# --- identity -------------------------------------------------------------------------------


def _json(rec: dict) -> list[str]:
    evs = logformats.parse_json_text(json.dumps(rec))
    assert evs
    return evs[0]["techniques"]


def test_azure_audit_operations() -> None:
    base = {"operationName": "Disable Strong Authentication", "category": "UserManagement",
            "resourceId": "/tenants/x", "time": "2025-01-01T00:00:00Z"}  # fmt: skip
    assert "T1556" in _json(base)
    assert "T1528" in _json(base | {"operationName": "Consent to application"})
    assert _json(base | {"operationName": "Update user"}) == []  # nothing risky changed


def test_azure_signin_failures_vs_mfa_prompts() -> None:
    rec = {"operationName": "Sign-in activity", "category": "SignInLogs", "resourceId": "x",
           "resultType": "50126", "time": "2025-01-01T00:00:00Z"}  # fmt: skip
    assert _json(rec) == ["T1110"]
    assert _json(rec | {"resultType": "50074"}) == []  # MFA required: not a failure


def test_m365_mail_rules_and_forwarding() -> None:
    rule = {"Workload": "Exchange", "Operation": "New-InboxRule", "CreationTime": "2025-01-01T00:00:00",
            "Parameters": [{"Name": "ForwardTo", "Value": "x@example.com"}, {"Name": "Name", "Value": "r"}]}  # fmt: skip
    assert "T1114" in _json(rule)
    hide = rule | {
        "Parameters": [
            {"Name": "MoveToFolder", "Value": "RSS Feeds"},
            {"Name": "MarkAsRead", "Value": "True"},
        ]
    }
    assert _json(hide) == ["T1564"]
    fwd = {"Workload": "Exchange", "Operation": "Set-Mailbox", "CreationTime": "2025-01-01T00:00:00",
           "Parameters": [{"Name": "ForwardingSmtpAddress", "Value": "smtp:x@example.com"}]}  # fmt: skip
    assert "T1114" in _json(fwd)
    cleared = fwd | {"Parameters": [{"Name": "ForwardingSmtpAddress", "Value": ""}]}
    assert _json(cleared) == []


def test_m365_login_failures() -> None:
    rec = {"Workload": "AzureActiveDirectory", "Operation": "UserLoginFailed",
           "CreationTime": "2025-01-01T00:00:00", "LogonError": "InvalidUserNameOrPassword"}  # fmt: skip
    assert _json(rec) == ["T1110"]


def test_okta_public_event_types() -> None:
    rec = {"eventType": "user.mfa.factor.deactivate", "displayMessage": "x",
           "published": "2025-01-01T00:00:00Z", "outcome": {"result": "SUCCESS"}}  # fmt: skip
    assert "T1556" in _json(rec)
    fail = rec | {"eventType": "user.session.start", "outcome": {"result": "FAILURE"}}
    assert _json(fail) == ["T1110"]
    ok = rec | {"eventType": "user.session.start"}
    assert _json(ok) == []
