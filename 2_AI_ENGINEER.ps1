# SplitShield-XAI - AI ENGINEER: dataset intake -> training -> testing -> validation
# -> final report. Needs the environment from 1_SETUP_AND_RUN.ps1 (it calls this).
#
#   powershell -ExecutionPolicy Bypass -File 2_AI_ENGINEER.ps1           # full run
#   powershell -ExecutionPolicy Bypass -File 2_AI_ENGINEER.ps1 -Fast     # minutes, fixture
#   powershell -ExecutionPolicy Bypass -File 2_AI_ENGINEER.ps1 -Fresh    # redo all phases
#
# Safe to re-run: finished phases are skipped (progress in artifacts\). Keeps the
# PC awake while it runs. Exit code 0 = everything passed.
param([switch]$Fast, [switch]$Fresh, [string]$Experiment = "")

# Native tools (pip, python) write warnings to stderr; under "Stop" Windows
# PowerShell 5.1 turns that into a fatal error, so exit codes are checked explicitly.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
. "$PSScriptRoot\scripts\power.ps1"
Disable-QuickEdit

$vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    Write-Host "No Python environment found. Run 1_SETUP_AND_RUN.ps1 first." -ForegroundColor Red
    exit 2
}
$env:PYTHONPATH = $PSScriptRoot
$env:PYTHONUTF8 = "1"            # UTF-8 console output (no cp1252 crashes on symbols)
$env:PYTHONIOENCODING = "utf-8"
$env:LOKY_MAX_CPU_COUNT = [Environment]::ProcessorCount  # Win11 has no wmic; avoids a joblib warning

$rc = 1
Enter-NoSleep
try {
    Write-Host "== [AI engineer] checks -> intake -> gate -> train -> test -> report =="
    $a = @()
    if ($Fast) { $a += "--fast" }
    if ($Fresh) { $a += "--fresh" }
    if ($Experiment) { $a += @("--experiment", $Experiment) }
    & $vpy ai_engineer.py @a
    $rc = $LASTEXITCODE
} catch {
    Write-Host "AI engineer stopped: $_" -ForegroundColor Red
    $rc = 1
} finally {
    Exit-NoSleep
}
if ($rc -eq 0) {
    Write-Host "AI engineer finished OK. Report: results\FINAL_REPORT.md" -ForegroundColor Green
} else {
    Write-Host "AI engineer finished with failures (see logs\ai_engineer.log). Re-run to retry only those." -ForegroundColor Yellow
}
exit $rc
