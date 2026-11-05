"""Extra Windows detection rules (owner 2026-10-14: log -> ATT&CK mapping).

Written from a fixed DEVELOPMENT half of EVTX-ATTACK-SAMPLES only and scored on the
held-out half plus OTRF Security-Datasets (scripts/extra_datasets.py), so the
reported accuracy is not inflated. The rules are generic behaviours (the kind public
Sigma rules use), never sample file names, and they ignore Sysmon's ``RuleName``
field, which carries the sample author's own labels.

Each rule returns ``(kind, [technique ids])`` or ``None``.
"""

from __future__ import annotations

import re

Hit = tuple[str, list[str]]

_I = re.I | re.M


def _f(msg: str, *names: str) -> str:
    """First value of a ``Name: value`` field (Sysmon / Security message layout)."""
    for n in names:
        m = re.search(rf"^\s*{re.escape(n)}:\s*(.*)$", msg, _I)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return ""


def strip_rulename(msg: str) -> str:
    """Drop Sysmon's RuleName line: it holds the config author's labels, not evidence."""
    return re.sub(r"^\s*RuleName:.*$", "", msg, flags=re.M)


_SHELLS = (
    r"\\(cmd|powershell|pwsh|whoami|net1?|rundll32|mshta|cscript|wscript|mmc)\.exe$"
)
_AUTO_ELEVATE = (
    r"\\(fodhelper|computerdefaults|eventvwr|sdclt|slui|changepk|wsreset|perfmon)\.exe$"
)
_ACCESSIBILITY = r"\\(utilman|sethc|osk|magnify|narrator|displayswitch|atbroker)\.exe$"
_SERVERS = r"\\(w3wp|sqlservr|httpd|tomcat\d*|php-cgi|nginx|java)\.exe$"

# (command-line pattern, kind, technique) -- most specific first
_CMD: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"ntdsutil.*\b(ifm|create full)\b|vssadmin(\.exe)?\s+create\s+shadow|"
                r"reg(\.exe)?\s+save\s+hklm\\(sam|system|security)", re.I),
     "credential_store_copy", "T1003"),
    (re.compile(r"cmd(\.exe)?[\"']?\s+/q\s+/c\s.*\\\\[\w.-]+\\(admin|c)\$", re.I),
     "remote_exec_output_to_share", "T1021"),
    (re.compile(r">\s*\\\\\.\\pipe\\", re.I), "named_pipe_impersonation", "T1134"),
    (re.compile(r"/namespace:.*subscription|__eventfilter|commandlineeventconsumer|"
                r"activescripteventconsumer", re.I), "wmi_subscription", "T1546"),
    (re.compile(r"wmic(\.exe)?[\"']?\s.*process\s+call\s+create", re.I),
     "wmi_process_create", "T1047"),
    (re.compile(r"wmic(\.exe)?[\"']?\s.*/format:\s*[\"']?\S*(https?:|\.xsl)|"
                r"\bmsxsl(\.exe)?[\"']?\s", re.I), "xsl_script", "T1220"),
    (re.compile(r"/i:\s*[\"']?(https?:|\S+\.sct)", re.I), "scriptlet_proxy_exec", "T1218"),
    (re.compile(r"rundll32(\.exe)?[\"']?\s+.*(advpack\.dll,\s*(registerocx|launchinfsection)|"
                r"pcwutl\.dll,\s*launchapplication|zipfldr\.dll,\s*routethecall|"
                r"url\.dll,\s*(openurl|fileprotocolhandler)|shell32\.dll\"?,\s*#44|"
                r"control_rundll.*\.cpl|ieadvpack|syssetup\.dll)", re.I),
     "lolbin_proxy_exec", "T1218"),
    (re.compile(r"mshta(\.exe)?[\"']?\s+[\"']?[^\"'\s]*\.hta|vshadow(\.exe)?[\"']?\s.*-exec=|"
                r"control(\.exe)?[\"']?\s+[\"']?[^\"']*\.cpl", re.I),
     "lolbin_proxy_exec", "T1218"),
    (re.compile(r"\b[cw]script(\.exe)?[\"']?\s.*(//?e:|\.(vbs|vbe|js|jse|wsf)\b)", re.I),
     "script_host", "T1059"),
    (re.compile(r"schtasks(\.exe)?[\"']?\s.*/create\b", re.I), "scheduled_task", "T1053"),
    (re.compile(r"bitsadmin.*/(setnotifycmdline|addfile|create)\b", re.I), "bits_job", "T1197"),
    (re.compile(r"\battrib(\.exe)?[\"']?\s+.*\+h\b", re.I), "hidden_file", "T1564"),
    (re.compile(r"\b(icacls|cacls)(\.exe)?[\"']?\s+\S*\\windows\\system32\\\S+\s+/grant|"
                r"takeown(\.exe)?[\"']?\s+/f\s+\S*\\windows\\system32", re.I),
     "system_file_permissions", "T1222"),
    (re.compile(r"\bsdbinst(\.exe)?[\"']?\s", re.I), "application_shim", "T1546"),
    (re.compile(r"\bnet1?(\.exe)?[\"']?\s+(localgroup|group)\b", re.I),
     "group_discovery", "T1069"),
    (re.compile(r"\bnet1?(\.exe)?[\"']?\s+(user|accounts)\b", re.I),
     "account_discovery", "T1087"),
    (re.compile(r"\b(nltest|dsquery|adfind)(\.exe)?\b|\bnet1?(\.exe)?[\"']?\s+view\b", re.I),
     "remote_system_discovery", "T1018"),
    (re.compile(r"\bwhoami(\.exe)?\b", re.I), "user_discovery", "T1033"),
    (re.compile(r"\b(ipconfig|netstat)(\.exe)?\b|\barp(\.exe)?\s+-a\b|\broute\s+print\b", re.I),
     "network_config_discovery", "T1016"),
    (re.compile(r"\bsysteminfo(\.exe)?\b", re.I), "system_info_discovery", "T1082"),
]  # fmt: skip


def process(msg: str) -> Hit | None:
    """Process creation (Sysmon 1 / Security 4688): parent-child pairs, then the
    command line."""
    image = _f(msg, "Image", "New Process Name")
    parent = _f(msg, "ParentImage", "Creator Process Name")
    cmd = _f(msg, "CommandLine", "Process Command Line") or image
    if re.search(
        r"^[a-z]:\\windows \\", image, re.I
    ):  # "C:\Windows \System32" mock dir
        return "masquerading_trusted_dir", ["T1548", "T1036"]
    if re.search(r"\\harddiskvolumeshadowcopy\d+\\", image + cmd, re.I):
        return "exec_from_shadow_copy", ["T1036"]
    if parent:
        if re.search(_AUTO_ELEVATE, parent, re.I) and re.search(_SHELLS, image, re.I):
            return "uac_bypass", ["T1548"]
        if re.search(_ACCESSIBILITY, parent, re.I) and re.search(_SHELLS, image, re.I):
            return "accessibility_backdoor", ["T1546"]
        if re.search(_SERVERS, parent, re.I) and re.search(_SHELLS, image, re.I):
            return "server_spawned_shell", ["T1505"]
        if re.search(r"\\wmiprvse\.exe$", parent, re.I) and re.search(
            _SHELLS, image, re.I
        ):
            return "wmi_exec", ["T1047"]
        if re.search(r"\\(winrshost|wsmprovhost)\.exe$", parent, re.I):
            return "winrm_exec", ["T1021"]
        if re.search(r"\\hh\.exe$", parent, re.I) and re.search(_SHELLS, image, re.I):
            return "chm_proxy_exec", ["T1218"]
    for pat, kind, tech in _CMD:
        if pat.search(cmd):
            return kind, [tech]
    return None


_REG: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\\(ms-settings|mscfile|exefile|folder|launcher\.systemsettings|appxfile)"
                r"\\shell\\(open|runas)\\command|\\environment\\windir\b|"
                r"volatile environment\\systemroot", re.I), "uac_bypass_registry", "T1548"),
    (re.compile(r"image file execution options\\[^\\]+\\(debugger|globalflag)|"
                r"silentprocessexit\\[^\\]+\\monitorprocess", re.I),
     "image_hijack_registry", "T1546"),
    (re.compile(r"user shell folders\\(common )?startup", re.I), "autostart_added", "T1547"),
    (re.compile(r"\\(scriptblocklogging|transcription|modulelogging)\\|__pslockdownpolicy|"
                r"\\winevt\\channels\\.*\\enabled|windows defender\\.*(disable|exclusions)|"
                r"\\shellids\\microsoft\.powershell\\executionpolicy", re.I),
     "defense_tampering", "T1562"),
    (re.compile(r"\\terminal server\\(winstations\\rdp-tcp\\portnumber|fdenytsconnections)",
                re.I), "rdp_settings_changed", "T1112"),
    (re.compile(r"\\sam\\sam\\domains\\account\\users\\names\\[^\\]*\$", re.I),
     "hidden_account", "T1564"),
    (re.compile(r"\\sam\\sam\\domains\\account\\users\\names\\", re.I),
     "account_created", "T1136"),
    # 2026-10-16: LSA secrets (machine-account password, service passwords) written
    (re.compile(r"\\security\\policy\\secrets\\", re.I), "lsa_secret_written", "T1003"),
    # a new SMB share (staging for lateral movement)
    (re.compile(r"\\services\\lanmanserver\\shares\\", re.I), "share_added", "T1021"),
]  # fmt: skip


def registry(msg: str) -> Hit | None:
    """Sysmon 12 / 13 / 14 registry events."""
    target = _f(msg, "TargetObject")
    for pat, kind, tech in _REG:
        if pat.search(target):
            return kind, [tech]
    return None


_PRIVATE = re.compile(
    r"^(10\.|127\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|0:0:|::1|fe80)"
)


def sysmon_other(eid: int, msg: str) -> Hit | None:
    """Sysmon events other than process creation / registry."""
    if eid == 10 and "UNKNOWN(" in _f(msg, "CallTrace"):  # call from unbacked memory
        return "injected_code_access", ["T1055"]
    if eid == 11:
        name = _f(msg, "TargetFilename")
        if re.search(r"mimilsa\.log$|lsass[^\\]*\.dmp$", name, re.I):
            return "credential_dump_file", ["T1003"]
    if eid in (17, 18):
        pipe = _f(msg, "PipeName")
        if re.search(r"^\\(psexesvc|remcom_communic|paexec)", pipe, re.I):
            return "remote_service_pipe", ["T1569", "T1021"]
        if re.search(r"^\\[^\\]+\\pipe\\", pipe, re.I):  # \RoguePotato\pipe\epmapper
            return "named_pipe_impersonation", ["T1134"]
    if eid in (19, 20, 21):
        return "wmi_subscription", ["T1546"]
    if eid == 25:
        return "process_tampering", ["T1055"]
    if eid == 3:
        image = _f(msg, "Image")
        dst, port = _f(msg, "DestinationIp"), _f(msg, "DestinationPort")
        if (
            re.search(_SERVERS, image, re.I)
            and dst.startswith(("127.", "::1"))
            and port in ("3389", "445", "22")
        ):
            return "tunnel_from_server_process", ["T1572"]
        if (
            re.search(
                r"\\(mshta|regsvr32|rundll32|wscript|cscript|wmic|msxsl)\.exe$",
                image,
                re.I,
            )
            and dst
            and not _PRIVATE.match(dst)
        ):
            return "lolbin_network", ["T1218"]
    if eid == 7:
        image, dll = _f(msg, "Image"), _f(msg, "ImageLoaded")
        if re.search(r"^[a-z]:\\windows \\", image + dll, re.I) or (
            re.search(r"^[a-z]:\\windows\\", image, re.I)
            and re.search(
                r"\\(appdata\\local\\temp|programdata|users\\public)\\[^\\]+\.dll$",
                dll,
                re.I,
            )
        ):
            return "dll_hijack", ["T1574"]
    return None


_REPL = re.compile(
    r"1131f6a[ad]-9c07-11d1-f79f-00c04fc2dcd2|89e95b76-444d-4c62-991a-0facbeda640c",
    re.I,
)
_BROWSER_STORE = re.compile(r"\\(login data|logins\.json|key[34]\.db|web data)$", re.I)
_BROWSERS = re.compile(r"\\(chrome|firefox|msedge|opera|brave|iexplore)\.exe$", re.I)


def security(eid: int, msg: str, provider: str) -> Hit | None:
    """Security / System / Application events (non-Sysmon)."""
    if eid == 104 and re.search(r"log file was cleared", msg, re.I):
        return "log_cleared", ["T1070"]
    if eid == 4703 and re.search(r"SeDebugPrivilege", msg):
        return "debug_privilege_enabled", ["T1134"]
    if eid in (4765, 4766):
        return "sid_history_added", ["T1134"]
    if eid == 4794:
        return "dsrm_password_set", ["T1098"]
    if eid == 4769 and re.search(r"Ticket Encryption Type:\s*0x17\b", msg):
        return "kerberoasting_rc4_ticket", ["T1558"]
    if (
        eid == 4768
        and re.search(r"Ticket Encryption Type:\s*0x17\b", msg)
        and re.search(r"Pre-Authentication Type:\s*0\b", msg)
    ):
        return "asrep_roasting", ["T1558"]
    if eid == 4662 and _REPL.search(msg):
        return "directory_replication", ["T1003"]
    if eid in (4656, 4663):
        obj, proc = _f(msg, "Object Name"), _f(msg, "Process Name")
        if re.search(r"\\lsass\.exe$", obj, re.I):
            return "lsass_access", ["T1003"]
        if _BROWSER_STORE.search(obj) and not _BROWSERS.search(proc):
            return "browser_credential_access", ["T1555"]
    if eid == 4624:
        if re.search(r"Logon Type:\s*9\b", msg):
            return "new_credentials_logon", ["T1550"]
        if re.search(r"Logon Type:\s*10\b", msg) and re.search(
            r"Source Network Address:\s*(127\.0\.0\.1|::1)\b", msg
        ):
            return "rdp_over_tunnel", ["T1572"]
    if eid == 5145:
        target = _f(msg, "Relative Target Name")
        if re.search(
            r"psexesvc|remcom|paexec|-(stdin|stdout|stderr)$|^(svcctl|atsvc)$",
            target,
            re.I,
        ):
            return "remote_service_share", ["T1021"]
    if eid in (7045, 4697):
        svc = _f(msg, "Service File Name")
        techs = ["T1543"]
        if re.search(r"\\\\\.\\pipe\\", svc):
            techs.append("T1134")
        if re.search(r"%comspec%|cmd(\.exe)?\s+/c|powershell", svc, re.I):
            techs.append("T1569")
        # the service binary or its output goes through an admin share: remote service
        # execution (PsExec / Impacket-style), 2026-10-25 from the EVTX development half
        if re.search(r"\\\\[^\\]+\\admin\$|\\\\127\.0\.0\.1\\", svc, re.I):
            techs += ["T1021.002", "T1569.002"]
        return "service_installed", techs
    if eid == 3001 and re.search(
        r"lsass\.exe' was terminated by the process", msg, re.I
    ):
        return "lsass_silent_exit_dump", ["T1003"]
    if eid in (325, 326, 327) and re.search(r"NTDS.*\$snap_", msg, re.I):
        return "ntds_snapshot_copy", ["T1003"]
    if eid == 500 and re.search(r"Compatibility fix applied to", msg, re.I):
        return "application_shim", ["T1546"]
    # inbound sessions only (169 user authenticated, 193 virtual account); client-side
    # WinRM events are routine management traffic
    if ("winrm" in provider.lower() and eid in (169, 193)) or re.search(
        r"WinRM virtual account", msg
    ):
        return "winrm_session", ["T1021"]
    if eid in (1040, 1042) and re.search(
        r"Installer transaction:\s*https?://", msg, re.I
    ):
        return "remote_msi_install", ["T1218"]
    # --- 2026-10-16 additions (generic behaviours, from development samples only) ---
    # EFS remote protocol calls: PetitPotam-style coerced authentication
    if eid in (5, 6) and "rpc" in provider.lower() and _EFSRPC.search(msg):
        return "efsrpc_coerced_auth", ["T1187"]
    # boot configuration loaded with kernel debugging / test signing / no integrity checks
    if eid == 4826 and re.search(
        r"(Kernel Debugging|Test Signing|Disable Integrity Checks):\s*Yes", msg, re.I
    ):
        return "code_signing_weakened", ["T1553"]
    # transacted file changes by a program outside Windows: process doppelganging
    if eid == 4985:
        proc = _f(msg, "Process Name")
        if proc and not re.search(r"^[a-z]:\\windows\\", proc, re.I):
            return "transaction_by_user_program", ["T1055"]
    # anonymous network logon from this machine itself: potato-style token theft
    if (
        eid == 4624
        and re.search(r"Logon Type:\s*3\b", msg)
        and re.search(r"Account Name:\s*ANONYMOUS LOGON", msg, re.I)
        and re.search(
            r"Source Network Address:\s*(127\.0\.0\.1|::1)\s*$", msg, re.I | re.M
        )
    ):
        return "anonymous_local_logon", ["T1134"]
    # BITS downloading a payload-type file from outside the usual software vendors
    # (updaters such as Windows, Office, Chrome or Firefox use BITS every day)
    if eid in (59, 60) and "bits" in provider.lower():
        url = re.search(r"https?://([^/\s]+)(/\S*)?", msg)
        if (
            url
            and re.search(_PAYLOAD, (url.group(2) or "").split("?")[0], re.I)
            and not re.search(_VENDORS, url.group(1), re.I)
        ):
            return "bits_external_download", ["T1197"]
    return None


_PAYLOAD = r"\.(exe|dll|ps1|bat|cmd|vbs|js|hta|scr|msi|7z|zip|rar|iso|bin)$"
_VENDORS = (
    r"(^|\.)(microsoft|windowsupdate|windows|msftconnecttest|office|bing|live|google|"
    r"googleapis|gvt1|mozilla|adobe|apple|dell|hp|lenovo|intel|nvidia|amd)\.(com|net)$"
)
_EFSRPC = re.compile(
    r"c681d488-d850-11d0-8c52-00c04fd90f7e|df1941c5-fe89-4e79-bf10-463657acf44d", re.I
)
