# Run ONCE in an Administrator PowerShell (owner decision 2026-10-14):
#   powershell -ExecutionPolicy Bypass -File "scripts\install_sysmon.ps1"
#
# Installs Microsoft Sysinternals Sysmon with the SwiftOnSecurity community config
# (both downloaded to tools\sysmon\). Sysmon logs process creation, network
# connections, LSASS access, registry autostarts, remote threads... to
# "Microsoft-Windows-Sysmon/Operational"; the SOC reads it ("Analyze logs" ->
# "Pull Windows logs"). Running this accepts the Sysmon EULA (tools\sysmon\bin\Eula.txt).
# Uninstall:  tools\sysmon\bin\Sysmon64.exe -u

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "tools\sysmon\bin\Sysmon64.exe"
$cfg = Join-Path $root "tools\sysmon\sysmonconfig-export.xml"

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { throw "Please run this in an Administrator PowerShell." }

# never install a binary that is not Microsoft-signed
$sig = Get-AuthenticodeSignature $exe
if ($sig.Status -ne "Valid" -or $sig.SignerCertificate.Subject -notmatch "O=Microsoft Corporation") {
    throw "Sysmon64.exe signature check failed ($($sig.Status)); not installing."
}
[xml](Get-Content $cfg) | Out-Null  # config must be well-formed XML

if (Get-Service -Name Sysmon64 -ErrorAction SilentlyContinue) {
    & $exe -c $cfg            # already installed: just apply the config
} else {
    & $exe -accepteula -i $cfg
}
Get-Service Sysmon64 | Select-Object Name, Status
Write-Host "Sysmon is logging to Microsoft-Windows-Sysmon/Operational."
