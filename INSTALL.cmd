@echo off
REM ============================================================================
REM  SplitShield-XAI - ONE-STEP INSTALL (double-click)
REM  Installs Python 3.12 if missing, creates .venv, installs the runtime libraries
REM  (CPU PyTorch, ~1.6 GB download, ~10 min), checks every model loads, adds a
REM  desktop shortcut and opens the SOC dashboard. Needs internet once.
REM  Already installed? Use START.cmd.
REM ============================================================================
cd /d "%~dp0"
where pwsh >nul 2>nul && (set "PS=pwsh") || (set "PS=powershell")
%PS% -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" %*
if errorlevel 1 (
  echo.
  echo Installation did not finish - see the message above.
  pause
)
