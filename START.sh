#!/bin/sh
# Start the SplitShield-XAI SOC dashboard on Linux / macOS (after install.sh).
# Opens http://localhost:8501 - reachable from this computer only.
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  echo "Not installed yet - running install.sh first."
  exec sh "$ROOT/install.sh"
fi
exec .venv/bin/python -m streamlit run "$ROOT/dashboard.py" --server.address localhost \
  --server.headless false --browser.gatherUsageStats false
