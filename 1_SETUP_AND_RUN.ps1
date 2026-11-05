# SplitShield-XAI - ONE-CLICK SETUP + RUN (Windows). Does everything, in order:
#   1. restarts itself as Administrator (firewall blocking, Security-log reading and
#      the lid/sleep settings need it; UAC prompt once)
#   2. keeps the PC awake and un-pausable until the training run has finished
#   3. installs Python 3.12 if missing (winget), creates .venv, installs every library
#      (with retries; CUDA PyTorch automatically when an NVIDIA GPU is present)
#   4. runs 2_AI_ENGINEER.ps1: unzips + identifies every dataset, then training,
#      testing, validation and results\FINAL_REPORT.md
#   5. opens the SOC dashboard at http://localhost:8501
#
#   powershell -ExecutionPolicy Bypass -File 1_SETUP_AND_RUN.ps1          # full
#   powershell -ExecutionPolicy Bypass -File 1_SETUP_AND_RUN.ps1 -Fast    # quick check
# Everything is logged to logs\setup_<time>.log.
param([switch]$Fast, [switch]$Fresh, [switch]$NoDashboard, [switch]$NoPause,
      [switch]$NoElevate)  # -NoElevate: run without asking for Administrator

# Native tools (pip, python) write warnings to stderr; under "Stop" Windows
# PowerShell 5.1 turns that into a fatal error, so exit codes are checked explicitly.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
. "$PSScriptRoot\scripts\power.ps1"

# --- 1. administrator ----------------------------------------------------------
if (-not (Test-Admin) -and -not $NoElevate) {
    $shell =if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh" } else { "powershell" }
    $argv = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"")
    foreach ($s in "Fast", "Fresh", "NoDashboard", "NoPause") {
        if ((Get-Variable $s).Value) { $argv += "-$s" }
    }
    try {
        Start-Process $shell -Verb RunAs -ArgumentList $argv -WorkingDirectory $PSScriptRoot
        Write-Host "Continuing in the Administrator window that just opened."
        exit 0
    } catch {
        Write-Host "Administrator rights were refused - continuing WITHOUT them." -ForegroundColor Yellow
        Write-Host "(Training works; firewall blocking, Security-log reading and the lid setting will not.)"
    }
}

New-Item -ItemType Directory -Force -Path "$PSScriptRoot\logs" | Out-Null
$log = "$PSScriptRoot\logs\setup_$(Get-Date -Format yyyyMMdd_HHmmss).log"
try { Start-Transcript -Path $log -Append | Out-Null } catch { }
Disable-QuickEdit
$Host.UI.RawUI.WindowTitle = "SplitShield-XAI - setup and training (do not close)"
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
# pip and Python unpack into %TEMP% (usually C:). A nearly full C: drive breaks the
# install even when the project drive has room, so keep both on the project drive.
foreach ($d in ".tmp", ".pip-cache") { New-Item -ItemType Directory -Force -Path "$PSScriptRoot\$d" | Out-Null }
$env:TEMP = "$PSScriptRoot\.tmp"; $env:TMP = $env:TEMP; $env:PIP_CACHE_DIR = "$PSScriptRoot\.pip-cache"

function Invoke-Retry([scriptblock]$Do, [string]$What, [int]$Tries = 3) {
    for ($i = 1; $i -le $Tries; $i++) {
        & $Do | Out-Host  # keep pip output out of the function's return value
        if ($LASTEXITCODE -eq 0) { return $true }
        Write-Host "  $What failed (attempt $i/$Tries)" -ForegroundColor Yellow
        Start-Sleep -Seconds (5 * $i)
    }
    return $false
}

function Find-Python312 {
    $cands = @()
    if (Get-Command py -ErrorAction SilentlyContinue) { $cands += ,@("py", "-3.12") }
    if (Get-Command python -ErrorAction SilentlyContinue) { $cands += ,@("python") }
    foreach ($c in $cands) {
        $rest = @($c | Select-Object -Skip 1)
        $ok = & $c[0] @rest -c "import sys,struct;print(int(sys.version_info[:2]>=(3,12) and struct.calcsize('P')==8))" 2>$null
        if ("$ok".Trim() -eq "1") { return ,$c }
    }
    return $null
}

$rc = 1
Enter-NoSleep
try {
    # --- 2. long paths (dataset folders nest deeply) -------------------------------
    if (Test-Admin) {
        try {
            Set-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" LongPathsEnabled 1 -Type DWord
        } catch { Write-Host "  (could not enable long paths: $_)" }
    }

    # --- 3. Python + libraries ---------------------------------------------------
    try {
        $free = [math]::Round((Get-PSDrive ($PSScriptRoot.Substring(0, 1)) -ErrorAction Stop).Free / 1GB)
        Write-Host "== disk: $free GB free on $($PSScriptRoot.Substring(0,2)) =="
        if ($free -lt 15) { Write-Host "WARNING: less than 15 GB free; the datasets need about 15 GB unpacked." -ForegroundColor Yellow }
    } catch { Write-Host "  (could not read free disk space)" }

    $py = Find-Python312
    if (-not $py) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            throw "Python 3.12 (64-bit) is required. Install it from https://www.python.org/downloads/ (tick 'Add to PATH') and run again."
        }
        Write-Host "== installing Python 3.12 (winget) =="
        winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements --silent
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
        $py = Find-Python312
        if (-not $py) { throw "Python 3.12 was installed but is not on PATH yet - close this window and run again." }
    }

    $vpy = "$PSScriptRoot\.venv\Scripts\python.exe"
    $healthy = $false
    if (Test-Path $vpy) { & $vpy -c "import sys" 2>$null; $healthy = ($LASTEXITCODE -eq 0) }
    if (-not $healthy) {
        if (Test-Path "$PSScriptRoot\.venv") { Write-Host "== repairing broken .venv =="; Remove-Item -Recurse -Force "$PSScriptRoot\.venv" -ErrorAction Stop }
        Write-Host "== creating .venv =="
        $rest = @($py | Select-Object -Skip 1)
        & $py[0] @rest -m venv "$PSScriptRoot\.venv"
        if ($LASTEXITCODE -ne 0) { throw "could not create the virtual environment" }
    }

    Write-Host "== installing libraries (this can take a while the first time) =="
    if (-not (Invoke-Retry { & $vpy -m pip install --upgrade pip } "pip upgrade")) { Write-Host "  continuing with the existing pip" }
    if (-not (Invoke-Retry { & $vpy -m pip install -r requirements.txt } "library install")) {
        throw "Library installation failed 3 times. Check the internet connection and run again (already-installed parts are kept)."
    }
    if (Test-Path "requirements-optional.txt") {
        # one by one: an optional extra that will not install never blocks the rest
        Get-Content requirements-optional.txt | Where-Object { $_ -match '^\s*[A-Za-z]' } | ForEach-Object {
            $pkg = ($_ -split '#')[0].Trim()
            & $vpy -m pip install $pkg 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) { Write-Host "  optional package skipped: $pkg" -ForegroundColor Yellow }
        }
    }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Host "== NVIDIA GPU found -> installing CUDA PyTorch =="
        $ok = Invoke-Retry { & $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --upgrade } "CUDA torch"
        & $vpy -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
        if (-not $ok -or $LASTEXITCODE -ne 0) { Write-Host "  CUDA not usable - training on CPU (slower, still correct)." -ForegroundColor Yellow }
    } else {
        Write-Host "== no NVIDIA GPU -> CPU PyTorch (full run will be slow; -Fast for a quick check) =="
    }

    # --- 4. AI engineer: data -> training -> testing -> validation -> report -----
    $a = @{}
    if ($Fast) { $a.Fast = $true }
    if ($Fresh) { $a.Fresh = $true }
    & "$PSScriptRoot\2_AI_ENGINEER.ps1" @a
    $rc = $LASTEXITCODE
} catch {
    Write-Host ""
    Write-Host "SETUP STOPPED: $_" -ForegroundColor Red
    Write-Host "Log: $log"
    $rc = 1
} finally {
    Exit-NoSleep
}

# --- 5. dashboard (also opened after failures, so you can see what happened) -----
$vpy = "$PSScriptRoot\.venv\Scripts\python.exe"
if (-not $NoDashboard -and (Test-Path $vpy)) {
    & $vpy -c "import streamlit" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "== opening the SplitShield-XAI dashboard: http://localhost:8501 (close this window to stop it) ==" -ForegroundColor Green
        Start-Job { Start-Sleep 8; Start-Process "http://localhost:8501" } | Out-Null
        try { Stop-Transcript | Out-Null } catch { }
        & $vpy -m streamlit run "$PSScriptRoot\dashboard.py" --server.address localhost --server.headless true --browser.gatherUsageStats false
        exit $rc
    }
}
try { Stop-Transcript | Out-Null } catch { }
if (-not $NoPause) { Read-Host "Finished (exit $rc). Press Enter to close" | Out-Null }
exit $rc
