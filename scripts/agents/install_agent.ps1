<#
Install the SplitShield-XAI Windows agent as a scheduled task that starts with Windows
and runs as SYSTEM (so it can read the Security log). Run as Administrator:

  powershell -ExecutionPolicy Bypass -File install_agent.ps1            # install + setup
  powershell -ExecutionPolicy Bypass -File install_agent.ps1 -Remove    # uninstall

The agent files are copied to C:\ProgramData\SplitShield-Agent (agent.json there holds
the key and is readable only by SYSTEM and Administrators).
#>
param([switch]$Remove)
$ErrorActionPreference = 'Stop'
$Task = 'SplitShield-XAI agent'
$Dest = Join-Path $env:ProgramData 'SplitShield-Agent'

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { Write-Output 'Please run this in a PowerShell opened as Administrator.'; exit 1 }

if ($Remove) {
    Unregister-ScheduledTask -TaskName $Task -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "Task removed. The folder $Dest (config, spool) was kept; delete it yourself if you want."
    exit 0
}

New-Item -ItemType Directory -Force $Dest | Out-Null
Copy-Item -Force (Join-Path $PSScriptRoot 'splitshield_agent.ps1') $Dest
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Dest 'splitshield_agent.ps1') -Setup -AgentHome $Dest
if ($LASTEXITCODE -ne 0) { Write-Output 'Setup did not connect - fix the address / key / fingerprint and run again.'; exit 1 }
icacls $Dest /inheritance:r /grant:r 'SYSTEM:(OI)(CI)F' 'Administrators:(OI)(CI)F' | Out-Null

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument (
    "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Dest\splitshield_agent.ps1`" -AgentHome `"$Dest`"")
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName $Task -Action $action -Trigger $trigger -Settings $settings `
    -User 'SYSTEM' -RunLevel Highest -Force | Out-Null
Start-ScheduledTask -TaskName $Task
Write-Output "Installed and started: task '$Task' (runs at every start, restarts if it stops)."
