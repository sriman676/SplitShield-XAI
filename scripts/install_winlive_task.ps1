# Live Windows-log watching for this PC (owner 2026-10-16). No administrator needed:
#   powershell -ExecutionPolicy Bypass -File "scripts\install_winlive_task.ps1"
#
# Registers "SplitShield-XAI live logs": at logon it starts (hidden)
#   pythonw -m src.soc.winlive
# which reads the new Security / Sysmon / PowerShell records every 10 s and stores
# them through the SOC (detection, incidents, SIEM forwarding). Read-only: it never
# blocks or changes anything. It runs as you, with normal rights.
#
# Security and Sysmon additionally need ONE administrator step (then sign out and in):
#   net localgroup "Event Log Readers" %USERNAME% /add
# Until then only the PowerShell log is read; the dashboard's Getting started page
# shows the status of each log.
# Remove with:  Unregister-ScheduledTask -TaskName "SplitShield-XAI live logs"

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pyw = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pyw)) { throw "No Python environment found: run INSTALL.cmd first." }

$action = New-ScheduledTaskAction -Execute $pyw -Argument "-m src.soc.winlive" `
    -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
# restart if it ever stops; never two at once; no time limit (it is a watcher)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
$who = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName "SplitShield-XAI live logs" -Action $action `
    -Trigger $trigger -Settings $settings -Principal $who -Force | Out-Null
Start-ScheduledTask -TaskName "SplitShield-XAI live logs"

Write-Host "Registered and started 'SplitShield-XAI live logs' (runs at every logon)."
Write-Host 'For Security + Sysmon, once as administrator: net localgroup "Event Log Readers" %USERNAME% /add  (then sign out and in)'
