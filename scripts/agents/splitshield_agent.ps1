<#
SplitShield-XAI log agent for Windows (PowerShell 5.1+, nothing to install).

Reads new records from the Security, Sysmon, PowerShell and System event logs and
sends them in batches over HTTPS to the SplitShield-XAI log receiver (port 8600).
The receiver's certificate is pinned by its SHA-256 fingerprint. Where each log
stopped is saved only after the batch was sent or spooled; batches that cannot be
sent wait in .\spool and are re-sent in order, so nothing is lost.

  powershell -ExecutionPolicy Bypass -File splitshield_agent.ps1 -Setup
  powershell -ExecutionPolicy Bypass -File splitshield_agent.ps1          # runs until stopped
  powershell -ExecutionPolicy Bypass -File splitshield_agent.ps1 -Once

Reading the Security log needs Administrator or membership of "Event Log Readers".
Install as a scheduled task with install_agent.ps1.
#>
param([switch]$Setup, [switch]$Once, [int]$Every = 2, [string]$AgentHome = $PSScriptRoot)

$ErrorActionPreference = 'Stop'
$Version = '1.0'
$Batch = 500
$Logs = [ordered]@{
    security   = 'Security'
    sysmon     = 'Microsoft-Windows-Sysmon/Operational'
    powershell = 'Microsoft-Windows-PowerShell/Operational'
    system     = 'System'
}
$CfgPath = Join-Path $AgentHome 'agent.json'
$StatePath = Join-Path $AgentHome 'agent_state.json'
$Spool = Join-Path $AgentHome 'spool'

function Write-Log([string]$m) { Write-Output ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m) }

function Get-Pinned([string]$Server, [string]$Fingerprint) {
    # TLS 1.2+, and the server certificate must have the pinned SHA-256. No fingerprint:
    # Windows' normal certificate checks (the organisation's own CA in the Windows store).
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    if (-not $Fingerprint) { [Net.ServicePointManager]::ServerCertificateValidationCallback = $null; return }
    $want = $Fingerprint.ToLower().Replace(':', '')
    [Net.ServicePointManager]::ServerCertificateValidationCallback = {
        param($s, $cert, $chain, $errors)
        $sha = [Security.Cryptography.SHA256]::Create()
        $got = -join ($sha.ComputeHash($cert.GetRawCertData()) | ForEach-Object { $_.ToString('x2') })
        return $got -eq $want
    }.GetNewClosure()
}

function Send-Raw([byte[]]$Body, $Cfg) {
    try {
        $req = [Net.HttpWebRequest]::Create(($Cfg.server.TrimEnd('/')) + '/ingest')
        $req.Method = 'POST'
        $req.ContentType = 'application/json'
        $req.Headers.Add('Authorization', 'Bearer ' + $Cfg.key)
        $req.Headers.Add('Content-Encoding', 'gzip')
        $req.UserAgent = "splitshield-agent-ps/$Version"
        $req.Timeout = 30000
        $req.ContentLength = $Body.Length
        $s = $req.GetRequestStream(); $s.Write($Body, 0, $Body.Length); $s.Close()
        $resp = $req.GetResponse(); $code = [int]$resp.StatusCode; $resp.Close()
        return $code -eq 200
    } catch {
        Write-Log "receiver unreachable or refused: $($_.Exception.Message)"
        return $false
    }
}

function ConvertTo-Gzip([string]$Text) {
    $bytes = [Text.Encoding]::UTF8.GetBytes($Text)
    $ms = New-Object IO.MemoryStream
    $gz = New-Object IO.Compression.GZipStream($ms, [IO.Compression.CompressionMode]::Compress)
    $gz.Write($bytes, 0, $bytes.Length); $gz.Close()
    return $ms.ToArray()
}

function Send-Spool($Cfg) {
    if (-not (Test-Path $Spool)) { return $true }
    foreach ($f in (Get-ChildItem $Spool -Filter *.json.gz | Sort-Object Name)) {
        if (-not (Send-Raw ([IO.File]::ReadAllBytes($f.FullName)) $Cfg)) { return $false }
        Remove-Item $f.FullName
    }
    return $true
}

function Send-Batch($Doc, $Cfg) {
    $body = ConvertTo-Gzip ($Doc | ConvertTo-Json -Depth 6 -Compress)
    if (-not (Send-Spool $Cfg) -or -not (Send-Raw $body $Cfg)) {
        New-Item -ItemType Directory -Force $Spool | Out-Null
        $name = '{0:D20}-{1}.json.gz' -f [DateTime]::UtcNow.Ticks, $PID
        [IO.File]::WriteAllBytes((Join-Path $Spool $name), $body)
    }
}

function Save-State($State) {
    $tmp = "$StatePath.tmp"
    ($State | ConvertTo-Json -Depth 4) | Set-Content -Encoding UTF8 $tmp
    Move-Item -Force $tmp $StatePath
}

function Read-Log([string]$Name, [string]$LogName, $State, $Cfg) {
    $key = "win:$LogName"
    $mark = $State.$key
    if ($null -eq $mark) {   # first run: start at the newest record
        $top = Get-WinEvent -LogName $LogName -MaxEvents 1 -ErrorAction SilentlyContinue
        $State | Add-Member -Force -NotePropertyName $key -NotePropertyValue ([int64]($top.RecordId))
        Save-State $State
        return 0
    }
    $xpath = "*[System[(EventRecordID>$([int64]$mark))]]"
    $evs = @(Get-WinEvent -LogName $LogName -FilterXPath $xpath -MaxEvents $Batch -Oldest -ErrorAction SilentlyContinue)
    if ($evs.Count -eq 0) { return 0 }
    $rows = foreach ($e in $evs) {
        $data = [ordered]@{}
        foreach ($d in ([xml]$e.ToXml()).Event.EventData.Data) { if ($d.Name) { $data[$d.Name] = [string]$d.'#text' } }
        $msg = $e.Message
        if (-not $msg) { $msg = ($data.GetEnumerator() | ForEach-Object { "$($_.Key): $($_.Value)" }) -join "`n" }
        [ordered]@{ Id = $e.Id; TimeCreated = $e.TimeCreated.ToUniversalTime().ToString('o')
                    ProviderName = $e.ProviderName; Channel = $e.LogName; Computer = $e.MachineName
                    Message = $msg; EventData = $data }
    }
    $sent = [double]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()) / 1000   # clock-skew check
    $bid = [guid]::NewGuid().ToString('N')   # the receiver drops a batch it already stored
    Send-Batch ([ordered]@{ source = $Name; host = $env:COMPUTERNAME; agent_version = $Version; sent = $sent; batch_id = $bid; events = @($rows) }) $Cfg
    $State | Add-Member -Force -NotePropertyName $key -NotePropertyValue ([int64]($evs | Measure-Object RecordId -Maximum).Maximum)
    Save-State $State
    return $evs.Count
}

if ($Setup) {
    $server = Read-Host 'Receiver address (e.g. https://192.168.1.10:8600)'
    $key = Read-Host 'Agent key (starts with ssa_)' -AsSecureString
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($key))
    $fp = Read-Host 'Certificate fingerprint (SHA-256)'
    $cfg = [ordered]@{ server = $server; key = $plain; fingerprint = $fp.ToLower().Replace(':', ''); logs = @($Logs.Keys) }
    New-Item -ItemType Directory -Force $AgentHome | Out-Null
    ($cfg | ConvertTo-Json) | Set-Content -Encoding UTF8 $CfgPath
    icacls $CfgPath /inheritance:r /grant:r "$($env:USERNAME):F" "SYSTEM:F" "Administrators:F" | Out-Null  # key is a secret
    Get-Pinned $cfg.server $cfg.fingerprint
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri ($server.TrimEnd('/') + '/agent/ping') -Headers @{ Authorization = "Bearer $plain" }
        Write-Output "Connected: $($r.Content)"
    } catch { Write-Output "Saved, but the receiver refused or could not be reached: $($_.Exception.Message)"; exit 1 }
    exit 0
}

if (-not (Test-Path $CfgPath)) { Write-Output 'No agent.json yet - run with -Setup first.'; exit 1 }
$cfg = Get-Content -Raw $CfgPath | ConvertFrom-Json
Get-Pinned $cfg.server $cfg.fingerprint
$state = if (Test-Path $StatePath) { Get-Content -Raw $StatePath | ConvertFrom-Json } else { New-Object PSObject }

while ($true) {
    $n = 0
    foreach ($name in $cfg.logs) {
        if (-not $Logs.Contains($name)) { continue }
        try { $n += Read-Log $name $Logs[$name] $state $cfg } catch { Write-Log "$name : $($_.Exception.Message)" }
    }
    [void](Send-Spool $cfg)
    if ($n) { Write-Log "sent $n records" }
    if ($Once) { break }
    Start-Sleep -Seconds $Every
}
