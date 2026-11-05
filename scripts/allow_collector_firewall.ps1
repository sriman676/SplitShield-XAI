<#
Allow the SplitShield-XAI log receiver through Windows Firewall - local network only.
Run ONCE as Administrator (the receiver itself runs with normal rights):

  powershell -ExecutionPolicy Bypass -File scripts\allow_collector_firewall.ps1 [-Port 8600]
  powershell -ExecutionPolicy Bypass -File scripts\allow_collector_firewall.ps1 -Remove

The rule admits TCP on the receiver port only from the local subnet and only on
Private / Domain networks (never on Public Wi-Fi). The receiver also refuses
non-private addresses itself.
#>
param([int]$Port = 8600, [switch]$Remove)
$ErrorActionPreference = 'Stop'
$Name = 'SplitShield-XAI log receiver'
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { Write-Output 'Please run this in a PowerShell opened as Administrator.'; exit 1 }
Remove-NetFirewallRule -DisplayName $Name -ErrorAction SilentlyContinue
if ($Remove) { Write-Output 'Rule removed.'; exit 0 }
New-NetFirewallRule -DisplayName $Name -Direction Inbound -Action Allow -Protocol TCP `
    -LocalPort $Port -RemoteAddress LocalSubnet -Profile Private, Domain | Out-Null
Write-Output "Allowed TCP $Port from the local subnet (Private / Domain networks only)."
