# Run ONCE in an Administrator PowerShell, only if the owner wants daily updates:
#   powershell -ExecutionPolicy Bypass -File "scripts\install_update_task.ps1"
#
# Registers "SplitShield-XAI update agent (daily)": every day at 04:00 it runs
#   python -m scripts.update_agent
# which does NOTHING unless the agent is switched on (dashboard toggle, or
#   python -m scripts.update_agent --enable). See docs/UPDATE_AGENT.md.
# Updates only Sigma rules, MITRE ATT&CK and free threat-intel feeds; every update is
# gated (validation, false-alarm check, tests) and kept for rollback.
# Remove with:  Unregister-ScheduledTask -TaskName "SplitShield-XAI update agent (daily)"

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { throw "Please run this in an Administrator PowerShell." }

$action = New-ScheduledTaskAction -Execute $py -Argument "-m scripts.update_agent" `
    -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At 04:00
# missed (PC off at 04:00) -> run when next available; no network -> the agent logs
# it and tries again the next day
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew
$who = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName "SplitShield-XAI update agent (daily)" -Action $action `
    -Trigger $trigger -Settings $settings -Principal $who -Force | Out-Null

Write-Host "Registered 'SplitShield-XAI update agent (daily)' (04:00, runs only if enabled)."
