@echo off
REM Double-click to run everything: setup (as Administrator), dataset unzip,
REM training, testing, validation, final report, then the SOC dashboard.
REM   LAUNCH.cmd -Fast     quick end-to-end check on the small fixture
cd /d "%~dp0"
where pwsh >nul 2>nul && (set "PS=pwsh") || (set "PS=powershell")
%PS% -NoProfile -ExecutionPolicy Bypass -File "%~dp01_SETUP_AND_RUN.ps1" %*
