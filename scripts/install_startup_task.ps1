# Run ONCE in an Administrator PowerShell (owner decision 2026-10-14):
#   powershell -ExecutionPolicy Bypass -File "scripts\install_startup_task.ps1"
#
# Registers "SplitShield-XAI watchdog (startup)": starts the run supervisor when the
# PC boots, before anyone logs in (after a power cut the run continues on its own).
# Runs as SYSTEM, so no password is stored. The existing logon task stays; the
# supervisor's lock file (rule R1) makes sure only one copy ever runs.
# Remove with:  Unregister-ScheduledTask -TaskName "SplitShield-XAI watchdog (startup)"

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$sup = Join-Path $root "scripts\supervisor.py"

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { throw "Please run this in an Administrator PowerShell." }

$action = New-ScheduledTaskAction -Execute $py -Argument "`"$sup`"" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT2M"  # let networking and disks come up first
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)
$who = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName "SplitShield-XAI watchdog (startup)" -Action $action `
    -Trigger $trigger -Settings $settings -Principal $who -Force | Out-Null

Write-Host "Registered 'SplitShield-XAI watchdog (startup)' (runs at boot as SYSTEM)."
Write-Host "Tip: in BIOS/UEFI enable 'Restore on AC Power Loss' = Power On, so the PC"
Write-Host "     switches itself back on after a power cut."
