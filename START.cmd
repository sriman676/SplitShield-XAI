@echo off
REM Start the SplitShield-XAI SOC dashboard (after INSTALL.cmd). Opens http://localhost:8501
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Not installed yet - running INSTALL.cmd first.
  call "%~dp0INSTALL.cmd"
  exit /b
)
".venv\Scripts\python.exe" -m streamlit run "%~dp0dashboard.py" --server.address localhost --server.headless false --browser.gatherUsageStats false
