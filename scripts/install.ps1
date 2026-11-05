# SplitShield-XAI one-step installer (called by INSTALL.cmd, owner 2026-10-15).
#   1. Python 3.12 (winget) if missing
#   2. .venv + runtime libraries (requirements-run.txt, CPU PyTorch), with retries
#   3. checks that the detectors, Sigma rules and assistant load
#   4. background jobs (src/soc/services.py) + desktop shortcut "SplitShield SOC" -> START.cmd
#   5. opens the dashboard
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($msg) { Write-Host "`n== $msg ==" -ForegroundColor Cyan }
function Retry([scriptblock]$cmd, [string]$what, [int]$n = 3) {
    for ($i = 1; $i -le $n; $i++) {
        & $cmd
        if ($LASTEXITCODE -eq 0) { return $true }
        Write-Host "  $what failed (attempt $i/$n), retrying in 10 s ..." -ForegroundColor Yellow
        Start-Sleep 10
    }
    return $false
}

Step "1/5 Python 3.12"
$py = $null
foreach ($c in @('py -3.12', 'python3.12', 'python')) {
    $parts = $c.Split(' ')
    try {
        $v = & $parts[0] @($parts[1..9] | Where-Object { $_ }) -c "import sys;print(sys.version_info[:2]==(3,12))" 2>$null
        if ($v -eq 'True') { $py = $parts; break }
    } catch {}
}
if (-not $py) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "Python 3.12 (64-bit) is needed: install it from https://www.python.org/downloads/ (tick 'Add to PATH'), then run INSTALL.cmd again."
    }
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements --silent
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
    $py = @('py', '-3.12')
}
Write-Host "  using: $($py -join ' ')"

Step "2/5 Virtual environment + libraries (about 10 minutes)"
$vpy = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $vpy)) { & $py[0] @($py[1..9] | Where-Object { $_ }) -m venv (Join-Path $root '.venv') }
[void](Retry { & $vpy -m pip install --upgrade pip --quiet } 'pip upgrade')
$ok = Retry { & $vpy -m pip install -r requirements-run.txt --extra-index-url https://download.pytorch.org/whl/cpu --quiet } 'library install'
if (-not $ok) { throw "Could not install the libraries - check the internet connection and run INSTALL.cmd again (it resumes)." }

Step "3/5 Checking that everything loads"
& $vpy -c @"
import sys
from src.soc import engine, sigma
from src.knowledge.explain import kb
det = engine.load_detector()
print('  flow detector :', 'OK' if det is not None else 'MISSING (artifacts/deploy)')
print('  Sigma rules   :', sigma.load()['stats']['loaded'], 'rules')
print('  ATT&CK        :', len(kb()), 'techniques')
from src.nlp.assistant import ShieldAssistant
print('  assistant     :', ShieldAssistant().answer('help')['intent'])
sys.exit(0 if det is not None else 1)
"@
if ($LASTEXITCODE -ne 0) { Write-Host "  Some models are missing; the dashboard will still open." -ForegroundColor Yellow }

Step "4/5 Background jobs (autopilot, backup, updates, live logs) + desktop shortcut"
& $vpy -m src.soc.services install
if ($LASTEXITCODE -ne 0) { Write-Host "  Some background jobs could not be registered - see above." -ForegroundColor Yellow }
try {
    $ws = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'SplitShield SOC.lnk'))
    $lnk.TargetPath = Join-Path $root 'START.cmd'
    $lnk.WorkingDirectory = $root
    $lnk.Save()
    Write-Host "  'SplitShield SOC' added to the desktop"
} catch { Write-Host "  (shortcut skipped: $($_.Exception.Message))" }

Step "5/5 Opening the dashboard at http://localhost:8501"
& (Join-Path $root 'START.cmd')
