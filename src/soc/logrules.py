"""Our own detection rules for non-Windows logs (owner 2026-10-15).

Like ``winrules.py``: generic behaviours, never test-file names. AWS and Kubernetes
rules are written from the public ATT&CK cloud/container matrices before looking at
the labelled attack events; web, Linux, DNS and mail rules from DEVELOPMENT splits
only (scripts/extra_logs.py scores them on held-out data).

``check(fields, source)`` returns ``(kind, [techniques])`` or None. ``fields`` uses
the field names of the log source (CloudTrail / Kubernetes JSON flattened to
lower-case dotted keys; web logs in Sigma's webserver names).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

Source = tuple[str | None, str | None, str | None]
Hit = tuple[str, list[str]]
F = dict[str, Any]


def _g(f: F, *keys: str) -> str:
    for k in keys:
        v = f.get(k)
        if v not in (None, ""):
            return str(v)
    return ""


def _has(f: F, key: str, pat: str) -> bool:
    return re.search(pat, _g(f, key), re.I) is not None


# --- AWS CloudTrail -----------------------------------------------------------------

_AWS_EVASION = {
    "StopLogging", "DeleteTrail", "UpdateTrail", "PutEventSelectors", "DeleteFlowLogs",
    "DeleteDetector", "UpdateDetector", "DisableSecurityHub", "DeleteConfigRule",
    "StopConfigurationRecorder", "DeleteConfigurationRecorder", "PutBucketLogging",
    "DisableEbsEncryptionByDefault", "DeleteLogGroup", "DeleteLogStream",
}  # fmt: skip
_AWS_PERSIST = {
    "CreateAccessKey", "CreateLoginProfile", "UpdateLoginProfile", "UpdateAssumeRolePolicy",
    "CreateVirtualMFADevice", "DeactivateMFADevice", "AddUserToGroup", "CreatePolicyVersion",
}  # fmt: skip
_AWS_POLICY = {"AttachUserPolicy", "AttachRolePolicy", "AttachGroupPolicy", "PutUserPolicy",
               "PutRolePolicy", "PutGroupPolicy"}  # fmt: skip
_AWS_IMPACT = {"DeleteBucket", "ScheduleKeyDeletion", "DisableKey", "DeleteDBInstance",
               "DeleteDBCluster", "PutBucketLifecycleConfiguration"}  # fmt: skip


def _aws(f: F) -> Hit | None:
    name = _g(f, "eventname")
    params = " ".join(
        str(v) for k, v in f.items() if k.startswith("requestparameters.")
    )
    if _g(f, "errorcode"):
        if name == "ConsoleLogin" or "AccessDenied" in _g(f, "errorcode"):
            return None  # failures are counted by the correlator, not single alerts
    if name in _AWS_EVASION:
        return "cloud_logging_disabled", ["T1562"]
    if name in _AWS_POLICY and re.search(
        r"AdministratorAccess|\"\*\"|:\*\b|'\*'", params
    ):
        return "cloud_admin_policy_attached", ["T1098"]
    if name in _AWS_PERSIST:
        return "cloud_credential_persistence", ["T1098"]
    if name == "CreateUser":
        return "cloud_account_created", ["T1136"]
    if name in ("GetSecretValue", "BatchGetSecretValue"):
        return "cloud_secret_read", ["T1555"]
    if name == "GetPasswordData":
        return "ec2_password_read", ["T1552"]
    if name in ("GetParameter", "GetParameters", "GetParametersByPath") and re.search(
        r"withdecryption\W*true", params, re.I
    ):
        return "ssm_secret_decrypted", ["T1552"]
    if name in ("PutBucketPolicy", "PutBucketAcl", "PutObjectAcl") and re.search(
        r"AllUsers|AuthenticatedUsers|\"Principal\":\s*\"\*\"|principal\W+\*",
        params,
        re.I,
    ):
        return "bucket_made_public", ["T1530"]
    if name in (
        "ModifySnapshotAttribute",
        "ModifyDBSnapshotAttribute",
        "ModifyImageAttribute",
    ) and re.search(r"add|restore|all", params, re.I):
        return "snapshot_shared", ["T1537"]
    if name in _AWS_IMPACT:
        return "cloud_destruction", ["T1485"]
    if name in ("SendCommand", "StartSession"):
        return "cloud_remote_command", ["T1651"]
    if name == "ModifyInstanceAttribute" and "userdata" in params.lower():
        return "user_data_modified", ["T1059"]
    if name == "ConsoleLogin" and _has(f, "additionaleventdata.mfaused", "^no$"):
        return "console_login_without_mfa", ["T1078"]
    return None


# --- Kubernetes audit -----------------------------------------------------------------


def _k8s(f: F) -> Hit | None:
    verb, res = _g(f, "verb"), _g(f, "objectref.resource")
    sub, user = _g(f, "objectref.subresource"), _g(f, "user.username")
    body = " ".join(str(v) for k, v in f.items() if k.startswith("requestobject."))
    if user in ("system:anonymous",) and verb not in ("get",):
        return "k8s_anonymous_access", ["T1078"]
    if res == "pods" and sub in ("exec", "attach") and verb in ("create", "get"):
        return "k8s_exec_into_pod", ["T1609"]
    if (
        res == "pods"
        and verb == "create"
        and re.search(
            r"privileged\W+true|hostpid\W+true|hostnetwork\W+true|hostpath", body, re.I
        )
    ):
        return "k8s_privileged_pod", ["T1611"]
    if (
        res in ("clusterrolebindings", "rolebindings")
        and verb == "create"
        and "cluster-admin" in body
    ):
        return "k8s_admin_binding", ["T1098"]
    if res == "secrets" and verb in ("list", "get") and not user.startswith("system:"):
        return "k8s_secret_access", ["T1552"]
    if res in ("cronjobs",) and verb == "create":
        return "k8s_cronjob_created", ["T1053"]
    if res == "serviceaccounts" and sub == "token" and verb == "create":
        return "k8s_token_created", ["T1528"]
    return None


# --- web server access logs (from the AIT-LDS development hosts cup + insect) -----------

_SCANNER_UA = re.compile(
    r"nikto|sqlmap|nmap|masscan|zgrab|wpscan|dirbuster|gobuster|feroxbuster|ffuf|wfuzz|"
    r"acunetix|nessus|openvas|nuclei|whatweb|joomscan|w3af|arachni|skipfish",
    re.I,
)
_BRUTE_UA = re.compile(r"hydra|medusa|ncrack|patator", re.I)
_SHELL_CMD = (
    r"(netcat|\bnc\b|ncat|/bin/(ba)?sh|bash\s+-i|/dev/tcp|whoami|uname|\bid\b|"
    r"wget\s|curl\s|chmod\s|python[23]?\s+-c|perl\s+-e|powershell|cat\s+/etc/)"
)
_WEBSHELL = re.compile(
    r"[?&](cmd|exec|command|execute|shell|run|c)=[^&]*" + _SHELL_CMD, re.I
)


_UPLOAD_SCRIPT = re.compile(r"/(wp-content/)?uploads?/[^?\s]*\.(php\d?|phtml|phar|jspx?|aspx?|ashx|cgi)(\?|$)", re.I)
_B64 = re.compile(r"^[A-Za-z0-9+/_-]{12,}={0,2}$")
_SHELL_WORD = re.compile(r"^\s*\[?\s*[\"']?(whoami|uname|id|hostname|ls|cat|pwd|wget|curl|tar|chmod|nc|ncat|bash|sh|"
                         r"python[23]?|perl|php|ps|netstat|ifconfig|ip|sudo|su|find|echo)\b", re.I)  # fmt: skip


def _decoded_commands(query: str) -> str | None:
    """A base64 parameter value that decodes to a shell command (or a JSON list of
    command words, as web shells send them): the decoded text, else None."""
    import base64
    import binascii
    from urllib.parse import parse_qsl

    for _k, v in parse_qsl(query, keep_blank_values=False):
        if not _B64.match(v):
            continue
        try:
            raw = base64.b64decode(v + "=" * (-len(v) % 4), altchars=b"-_" if ("-" in v or "_" in v) else None)
            txt = raw.decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            continue
        if txt.isprintable() and _SHELL_WORD.search(txt):
            return txt[:120]
    return None


def _web(f: F) -> Hit | None:
    from urllib.parse import unquote_plus

    ua = _g(f, "cs-user-agent", "c-useragent")
    raw_target = _g(f, "cs-uri-query", "c-uri")
    target = unquote_plus(raw_target)
    if _WEBSHELL.search(target):
        return "webshell_command", ["T1505"]
    query = raw_target.partition("?")[2]
    if query and _decoded_commands(query):  # 2026-10-24 (AIT-LDS v2 russellmitchell)
        return "webshell_encoded_command", ["T1505.003", "T1059.004"]
    if _UPLOAD_SCRIPT.search(target.partition("?")[0] + ("?" if "?" in target else "")):
        return "script_in_upload_folder", ["T1505.003"]
    if _BRUTE_UA.search(ua):
        return "web_bruteforce_tool", ["T1110"]
    if _SCANNER_UA.search(ua):
        return "web_scanner", ["T1595"]
    return None


# --- Linux auditd (decoded EXECVE / PROCTITLE, PATH records) -------------------------------

_REVERSE_SHELL = re.compile(
    r"\b(nc|ncat|netcat)\b.*\s-[ec]\s|/dev/tcp/|bash\s+-i\s*>|socat\b.*exec|"
    r"python[23]?\s+-c\s.*socket|perl\s+-e\s.*socket",
    re.I,
)
_TMP_DOWNLOAD = re.compile(
    r"\b(wget|curl)\b.*\s(-o\s*|-O\s*)?/(tmp|dev/shm|var/tmp)/", re.I
)


# Follow-up 2026-10-15: behaviours below come from the DEV half of non-blind Splunk
# attack_data (Sysmon for Linux, auditd; scripts/attackdata_followup.py) - generic
# patterns only, never dataset host / user names.
_SH = r"(/bin/|/usr/bin/)?(ba|da|z)?sh\b"
_SHELL_ESCAPE = re.compile(  # GTFOBins-style: a utility or interpreter opens a shell
    r"\bgdb\b.*\s-ex\s+['\"]?!|"
    r"\b(g|m|n)?awk\b.*\bsystem\s*\(|"
    r"\bnode(js)?\b.*\s-e\s.*child_process.*" + _SH + "|"
    r"\bphp[\d.]*\s+-r\s.*\b(system|exec|passthru|shell_exec|popen|proc_open)\s*\(.*" + _SH + "|"
    r"\bruby[\d.]*\s+-e\s.*\b(exec|system|spawn)\b.*" + _SH + "|"
    r"\bperl\s+-e\s.*\b(exec|system)\b.*" + _SH + "|"
    r"\bpython[\d.]*\s+-c\s.*(pty\.spawn|os\.system\s*\(\s*['\"]" + _SH + ")|"
    r"--script-security\s+2|\s--up\s+['\"]?" + _SH + "|"
    r"\bpuppet\s+apply\s+-e\s.*\bexec\b|"
    r"\bmake\b.*\s--eval\s*=|"
    r"\bfind\b.*\s-exec\s+" + _SH + "|"
    r"\bvim?\b.*\s-c\s+['\"]?:?!|"
    r"\btar\b.*--checkpoint-action\s*=\s*exec|"
    r"\b(env|nice|stdbuf|ionice|setarch|taskset)\s+" + _SH,
    re.I,
)  # fmt: skip
_CRED_FILES = re.compile(
    r"\b(cat|cp|less|more|head|tail|vim?|nano|grep|base64|xxd|dd|tar|zip|scp|rsync|"
    r"unshadow|john|strings)\b[^|;&]*\s/etc/(g?shadow|security/opasswd)\b",
    re.I,
)
_HISTORY = re.compile(
    r"HISTFILE=/dev/null|unset\s+HISTFILE|history\s+-c\b|HISTSIZE=0\b|"
    r"rm\s+(-\S+\s+)*\S*\.(bash|zsh)_history|ln\s+-sf?\s+/dev/null\s+\S*history",
    re.I,
)
_SECURITY_UNITS = (
    r"(auditd|sysmon\S*|rsyslog|syslog(-ng)?|apparmor|firewalld|ufw|falco\S*|osqueryd|"
    r"clamav\S*|wazuh\S*|ossec\S*|auditbeat|filebeat|falcon-sensor|splunk\S*)"
)
_SVC_STOP = re.compile(
    r"\bsystemctl\s+(stop|disable|mask|kill)\s+" + _SECURITY_UNITS + r"|"
    r"\bservice\s+" + _SECURITY_UNITS + r"\s+stop\b",
    re.I,
)
_PERSIST_PATHS = (  # (path pattern, kind, technique)
    (re.compile(r"^/etc/(update-motd\.d|profile\.d|init\.d|rc\d?\.d)/|^/etc/rc\.local$"),
     "logon_script_written", "T1037"),
    (re.compile(r"^/etc/cron(tab$|\.d/|\.(hourly|daily|weekly|monthly)/)|^/var/spool/cron/"),
     "cron_file_written", "T1053"),
    (re.compile(r"^(/etc|/usr/lib|/lib)/systemd/system/|/\.config/systemd/user/"),
     "systemd_unit_written", "T1543"),
    (re.compile(r"^/etc/ld\.so\.preload$"), "ld_preload_written", "T1574"),
    (re.compile(r"^/etc/sudoers(\.d/|$)"), "sudoers_written", "T1548"),
    (re.compile(r"/\.ssh/authorized_keys2?$"), "ssh_key_added", "T1098"),
    (re.compile(r"^/(root|home/[^/]+)/\.(bashrc|bash_profile|profile|zshrc|bash_login)$"),
     "shell_profile_written", "T1546"),
)  # fmt: skip
_PKG_TOOLS = re.compile(
    r"/(dpkg|apt(-get)?|rpm|yum|dnf|snapd?|unattended-upgrade\S*|packagekitd|cloud-init|"
    r"systemd\S*|NetworkManager|apk|zypper|pacman)$",
    re.I,
)
_DOC_APPS = re.compile(
    r"/(gs|gsc|ghostscript|soffice(\.bin)?|libreoffice\S*|evince|okular|convert|magick)$"
)
_EXFIL_TOOLS = re.compile(
    r"(^|/)(gdrive|rclone|megacmd|mega-put|megaput|dropbox_uploader(\.sh)?|gupload)\b|"
    r"\brclone\s+(copy|sync|move)\b",
    re.I,
)


def _linux(f: F) -> Hit | None:
    text = " ".join(
        str(f.get(k, "")) for k in ("proctitle", "commandline", "exe", "comm")
    )
    if _REVERSE_SHELL.search(text):
        return "reverse_shell", ["T1059"]
    if _TMP_DOWNLOAD.search(text):
        return "download_to_tmp", ["T1105"]
    if re.search(r"\bchmod\b\s+(\+x|[0-7]*7[0-7]{2})\s+/(tmp|dev/shm|var/tmp)/", text):
        return "tmp_made_executable", ["T1222"]
    name, uid = _g(f, "name"), _g(f, "uid")
    if name == "/etc/shadow" and uid not in ("", "0", "root"):
        return "shadow_read", ["T1003"]
    cmd = _g(f, "commandline", "proctitle")
    image = _g(f, "image", "exe")
    if cmd:
        if _SHELL_ESCAPE.search(cmd):
            return "shell_escape_via_utility", ["T1548", "T1059"]
        if re.search(r"\bsetcap\s+(-\S+\s+)*cap_\w+([,+=]\w*)*[+=]e?p", cmd, re.I):
            return "capability_added", ["T1548"]
        if re.search(r"\bchmod\s+(-\S+\s+)*([ug]*\+s|[246][0-7]{3})\s", cmd + " "):
            return "setuid_bit_set", ["T1548"]
        if _CRED_FILES.search(cmd):
            return "credential_file_access", ["T1003"]
        if _HISTORY.search(cmd):
            return "history_cleared", ["T1070"]
        if _SVC_STOP.search(cmd):
            return "security_service_stopped", ["T1562"]
        if re.search(r"\bgcc\b.*-shared.*-Wl,-init|-Wl,-init.*-shared", cmd):
            return "init_shared_object_built", ["T1129"]
        if _EXFIL_TOOLS.search(cmd.split(" ", 1)[0]) or _EXFIL_TOOLS.search(cmd):
            return "cloud_exfil_tool", ["T1567"]
        if re.match(r"(sudo\s+)?(insmod|rmmod)\s", cmd):
            return "kernel_module_change", ["T1547"]
        if re.match(r"(sudo\s+)?(at|batch)\s+\S|(sudo\s+)?crontab\s+(-e|-r|-u\s+\S+\s+\S+|[^-\s]\S*)",
                    cmd):  # fmt: skip
            return "scheduled_job_created", ["T1053"]
    if image.startswith(("/dev/shm/", "/run/shm/")) or cmd.startswith(
        ("/dev/shm/", "/run/shm/")
    ):
        return "exec_from_shared_memory", ["T1059"]
    parent = _g(f, "parentimage")
    if parent and _DOC_APPS.search(parent) and re.search(r"/(ba|da|z)?sh$", image):
        return "document_app_spawned_shell", ["T1203"]
    target = _g(f, "targetfilename")
    if target and not _PKG_TOOLS.search(image):
        for pat, kind, tech in _PERSIST_PATHS:
            if pat.search(target):
                return kind, [tech]
        if re.search(r"^/(tmp|var/tmp|dev/shm)/.+/etc/(passwd|group|shadow|sudoers|nsswitch\.conf)$",
                     target):  # fmt: skip
            return "fake_system_files_staged", ["T1036"]
    rtype = _g(f, "type")
    comm = _g(f, "comm").lower()
    if rtype == "SYSCALL" and _g(f, "tty").startswith("pts"):
        if comm in ("insmod", "rmmod", "modprobe"):
            return "kernel_module_change", ["T1547"]
        if comm in ("at", "crontab"):
            return "scheduled_job_created", ["T1053"]
    if rtype == "SERVICE_STOP" and re.fullmatch(_SECURITY_UNITS, _g(f, "unit"), re.I):
        return "security_service_stopped", ["T1562"]
    if rtype == "PATH" and _g(f, "nametype") == "CREATE":
        if re.search(r"(^|/)(sshd?_config|authorized_keys2?)~?$", name):
            return "ssh_config_changed", ["T1098"]
        if re.search(r"^/etc/cron|^/var/spool/cron/", name):
            return "cron_file_written", ["T1053"]
    return None


# --- identity: Azure AD / Entra ID, Microsoft 365, Okta --------------------------------

# Entra ID / Azure AD and the AzureActiveDirectory workload of the M365 audit log share
# operation names (M365 adds a trailing "."). From the dev half of attack_data.
_IDP_OPS: dict[str, Hit] = {
    "disable strong authentication": ("mfa_disabled", ["T1556"]),
    "consent to application": ("app_consent_granted", ["T1528"]),
    "add delegated permission grant": ("app_permission_granted", ["T1098"]),
    "add app role assignment to service principal": ("app_permission_granted", ["T1098"]),
    "add app role assignment grant to user": ("app_permission_granted", ["T1098"]),
    "add member to role": ("role_assigned", ["T1098"]),
    "add eligible member to role": ("role_assigned", ["T1098"]),
    "add scoped member to role": ("role_assigned", ["T1098"]),
    "add member to role completed (pim activation)": ("role_assigned", ["T1098"]),
    "add member to role in pim completed (permanent)": ("role_assigned", ["T1098"]),
    "user registered security info": ("mfa_method_registered", ["T1098"]),
    "user registered all required security info": ("mfa_method_registered", ["T1098"]),
    "admin registered security info": ("mfa_method_registered", ["T1098"]),
    "update authorization policy": ("identity_policy_weakened", ["T1562"]),
    "update company settings": ("identity_policy_weakened", ["T1562"]),
    "update conditional access policy": ("identity_policy_weakened", ["T1562"]),
    "delete conditional access policy": ("identity_policy_weakened", ["T1562"]),
    "set domain authentication": ("federation_changed", ["T1484"]),
    "set federation settings on domain": ("federation_changed", ["T1484"]),
    "invite external user": ("account_created", ["T1136"]),
    "add user": ("account_created", ["T1136"]),
    "add service principal": ("service_principal_created", ["T1136"]),
    "add application": ("service_principal_created", ["T1136"]),
    "add service principal credentials": ("app_credential_added", ["T1098"]),
    "update application – certificates and secrets management": ("app_credential_added", ["T1098"]),
}  # fmt: skip
# credential failures only (MFA interrupts, consent prompts, expired passwords are not)
_AAD_FAIL = {"50126", "50053", "50034", "50057", "50056", "50064"}
_M365_FAIL = {"InvalidUserNameOrPassword", "IdsLocked", "UserAccountNotFound"}
_SENSITIVE_SEARCH = re.compile(
    r"passw|credential|secret|confidential|\bvpn\b|private key|api[ _-]?key|\btoken\b|"
    r"\bssn\b|salary|payroll|admin",
    re.I,
)
_MAIL_DEFENCE = re.compile(
    r"^(disable|remove)-(safelinks|safeattachment|antiphish|malwarefilter|hostedcontentfilter|"
    r"dlpcompliance|atpprotectionpolicy)\w*$|^set-adminauditlogconfig$",
    re.I,
)


def _op(f: F, *keys: str) -> str:
    return _g(f, *keys).strip().rstrip(".").lower()


def _params(f: F) -> dict[str, str]:
    """M365 cmdlet ``Parameters`` (flattened as parameters.N.name / .value)."""
    out = {}
    for k, v in f.items():
        m = re.fullmatch(r"parameters\.(\d+)\.name", k)
        if m:
            out[str(v).lower()] = str(f.get(f"parameters.{m.group(1)}.value", ""))
    return out


def _modified(f: F) -> str:
    """Names of modified properties (Entra ID targetResources / M365 ModifiedProperties)."""
    return " ".join(
        str(v)
        for k, v in f.items()
        if "modifiedproperties" in k and k.endswith((".displayname", ".name"))
    )


def _update_user(f: F) -> Hit | None:
    props = _modified(f)
    if re.search(r"\bUserType\b", props):
        changes = " ".join(str(v) for k, v in f.items() if "modifiedproperties" in k)
        if re.search(r"member", changes, re.I) and re.search(r"guest", changes, re.I):
            return "guest_promoted_to_member", ["T1098"]
    if re.search(r"StrongAuthentication(Requirement|Method)", props):
        return "mfa_settings_changed", ["T1556"]
    return None


def _azure(f: F) -> Hit | None:
    op = _op(f, "operationname", "properties.activitydisplayname")
    if op == "sign-in activity":
        if _g(f, "resulttype", "properties.status.errorcode") in _AAD_FAIL:
            return "failed_logon", ["T1110"]
        return None
    if op == "update user":
        return _update_user(f)
    return _IDP_OPS.get(op)


def _m365(f: F) -> Hit | None:
    op = _op(f, "operation")
    if op == "userloginfailed":
        err = _g(f, "logonerror")
        if err in _M365_FAIL:
            return "failed_logon", ["T1110"]
        if err.startswith("SsoArtifact"):
            return "sso_token_error", ["T1550"]
        return None
    if op == "update user":
        return _update_user(f)
    if op in _IDP_OPS:
        return _IDP_OPS[op]
    p = _params(f)
    if op in ("new-inboxrule", "set-inboxrule", "updateinboxrules"):
        techs = []
        if {"forwardto", "forwardasattachmentto", "redirectto"} & set(p):
            techs.append("T1114")
        if {"deletemessage", "movetofolder", "markasread", "softdeletemessage"} & set(
            p
        ):
            techs.append("T1564")
        return (
            ("mailbox_rule_created", techs or ["T1114"])
            if p or op == "updateinboxrules"
            else None
        )
    if op == "set-mailbox" and any(
        p.get(k) for k in ("forwardingsmtpaddress", "forwardingaddress")
    ):
        return "mailbox_forwarding_set", ["T1114"]
    if op in ("new-transportrule", "set-transportrule") and {
        "blindcopyto", "redirectmessageto", "copyto", "addtorecipients"} & set(p):  # fmt: skip
        return "mail_flow_forwarding", ["T1114"]
    if op in ("add-mailboxpermission", "add-recipientpermission"):
        return "mailbox_permission_granted", ["T1098"]
    if op in ("modifyfolderpermissions", "addfolderpermissions", "add-mailboxfolderpermission",
              "set-mailboxfolderpermission"):  # fmt: skip
        return "mailbox_folder_shared", ["T1114"]
    if op in ("new-compliancesearch", "start-compliancesearch", "new-compliancesearchaction",
              "searchstarted", "searchexported", "new-mailboxexportrequest"):  # fmt: skip
        return "mailbox_search_or_export", ["T1114"]
    if op in ("add-federateddomain", "new-federateddomain", "set-domainauthentication",
              "set-msoldomainauthentication"):  # fmt: skip
        return "federation_changed", ["T1484"]
    if _MAIL_DEFENCE.match(op):
        return "mail_security_disabled", ["T1562"]
    if op.startswith("searchquery") and _SENSITIVE_SEARCH.search(
        _g(f, "searchquerytext", "querytext")
    ):
        return "sensitive_search", ["T1213"]
    if op == "filemalwaredetected":
        return "malware_file_detected", ["T1204"]
    if op in ("atpdetection", "timaildata"):
        return "malicious_mail_detected", ["T1566"]
    return None


# Okta System Log: no non-blind Okta data exists (the blind sample used every Okta
# dataset), so these come from Okta's public event-type catalogue, not from test data.
_OKTA: dict[str, Hit] = {
    "user.mfa.factor.deactivate": ("mfa_disabled", ["T1556"]),
    "user.mfa.factor.reset_all": ("mfa_disabled", ["T1556"]),
    "policy.lifecycle.deactivate": ("identity_policy_weakened", ["T1562"]),
    "policy.lifecycle.delete": ("identity_policy_weakened", ["T1562"]),
    "policy.rule.deactivate": ("identity_policy_weakened", ["T1562"]),
    "network_zone.rule.disabled": ("identity_policy_weakened", ["T1562"]),
    "zone.deactivate": ("identity_policy_weakened", ["T1562"]),
    "zone.delete": ("identity_policy_weakened", ["T1562"]),
    "user.account.privilege.grant": ("role_assigned", ["T1098"]),
    "group.privilege.grant": ("role_assigned", ["T1098"]),
    "system.api_token.create": ("api_token_created", ["T1098"]),
    "app.oauth2.as.consent.grant": ("app_consent_granted", ["T1528"]),
    "user.session.impersonation.initiate": ("session_impersonation", ["T1078"]),
    "user.lifecycle.create": ("account_created", ["T1136"]),
    "system.idp.lifecycle.create": ("federation_changed", ["T1484"]),
}  # fmt: skip


def _okta(f: F) -> Hit | None:
    et = _g(f, "eventtype").lower()
    if et in ("user.session.start", "user.authentication.sso", "user.authentication.auth_via_mfa") \
            and _g(f, "outcome.result").upper() == "FAILURE":  # fmt: skip
        return "failed_logon", ["T1110"]
    if et == "security.threat.detected":
        return "threat_detected", ["T1110"]
    return _OKTA.get(et)


# --- registry ---------------------------------------------------------------------------

PREDICATES: dict[str, list[Callable[[F], Hit | None]]] = {
    "aws": [_aws],
    "kubernetes": [_k8s],
    "web": [_web],
    "linux": [_linux],
    "azure": [_azure],
    "m365": [_m365],
    "okta": [_okta],
}
RULES: list[tuple[str, re.Pattern[str], str, list[str]]] = [
    # PHP running shell functions from an uploaded script (Apache error log)
    ("linux", re.compile(r"PHP (Warning|Notice).*\b(system|exec|passthru|shell_exec|popen|"
                         r"proc_open)\(\)|Undefined index: (cmd|exec|command)\b", re.I),
     "webshell_php_error", ["T1505"]),
    # SMTP user enumeration (Exim / Postfix)
    ("linux", re.compile(r"(?<!250-)(?<!250 )\b(VRFY|EXPN)\s+[<\w]", re.I), "smtp_user_enumeration", ["T1087"]),
    # repeated IMAP/POP authentication failures in one session (Dovecot)
    ("linux", re.compile(r"auth failed, [2-9]\d* attempts|auth failed, \d{2,} attempts", re.I),
     "mail_bruteforce", ["T1110"]),
    # Linux auth log (linux_secure). No non-blind attack_data exists for it (the blind
    # sample used every dataset), so these follow sudo / shadow-utils message formats;
    # checked for false alarms on the benign Loghub Linux log only.
    ("linux", re.compile(r"sudo: (?!.*BECOME-SUCCESS-).*COMMAND=(/usr)?/bin/((ba|da|z)?sh|su)(\s|$)", re.I),
     "sudo_root_shell", ["T1548"]),
    ("linux", re.compile(r"sudo: .*user NOT in sudoers|sudo: .*NOT allowed to run", re.I),
     "sudo_denied", ["T1548"]),
    ("linux", re.compile(r"usermod.*\s-a?G\s*\S*\b(sudo|wheel|admin|root)\b|"
                         r"add '[^']+' to (shadow )?group '(sudo|wheel|admin|root)'", re.I),
     "added_to_admin_group", ["T1098"]),
    ("linux", re.compile(r"password changed for root\b|passwd\[\d+\]: .*changed password for root",
                         re.I), "root_password_changed", ["T1098"]),
    # 2026-10-24, written from AIT-LDS v2 russellmitchell only (santos is the check):
    # a web / service account switching to a person's account (a web shell escalating)
    ("linux", re.compile(r"su(\[\d+\])?: (Successful su for \S+ by|\+ \S+) (www-data|apache|nginx|httpd|nobody|"
                         r"daemon|tomcat\S*|jenkins|wildfly)\b|"
                         r"uid=(33|48) .*exe=\"/(usr/)?bin/su\" .*res=success", re.I),
     "service_account_su", ["T1078.003"]),
    # credential files read through sudo / audit (cat /etc/shadow, decoded USER_CMD)
    ("linux", re.compile(r"\b(cat|cp|less|more|head|tail|vim?|nano|strings|base64|xxd|tar|zip)\b[^|;&]*\s"
                         r"/etc/(g?shadow|security/opasswd)\b", re.I), "credential_file_read", ["T1003.008"]),
    ("linux", re.compile(r"sudo: .*COMMAND=list\b", re.I), "sudo_privileges_listed", ["T1548.003"]),
]  # fmt: skip
"""(source key, pattern over the joined field values, kind, techniques) for text-like
logs (web, linux, dns, mail, zeek, macos); filled from development splits."""


def _key(source: Source) -> str:
    product, category, _service = source
    if category in ("webserver", "proxy"):
        return "web"
    if category == "dns":
        return "dns"
    return str(product or category or "*")


def check(fields: dict[str, Any], source: Source) -> Hit | None:
    key = _key(source)
    f = {str(k).lower(): v for k, v in fields.items()}
    for pred in PREDICATES.get(key, []):
        if hit := pred(f):
            return hit
    if not RULES:
        return None
    text = " ".join(str(v) for v in fields.values())
    for k, pat, kind, techs in RULES:
        if k in (key, "*") and pat.search(text):
            return kind, list(techs)
    return None
