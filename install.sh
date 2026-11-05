#!/bin/sh
# SplitShield-XAI one-step installer for Linux and macOS (the Windows one is INSTALL.cmd).
#
#   sh install.sh               # install + background jobs + open the dashboard
#   sh install.sh --no-open     # install only
#   sudo sh install.sh --firewall [PORT]   # allow the log receiver (default 8600)
#                                           # from the local network only
#
# Steps: Python 3.12 check -> .venv + runtime libraries (CPU PyTorch) -> checks that the
# detectors, Sigma rules and assistant load -> background jobs (systemd user units /
# launchd agents) -> START.sh. Runs as you, without root (only --firewall needs sudo).
set -eu
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
OS=$(uname -s)
step() { printf '\n== %s ==\n' "$1"; }

if [ "${1:-}" = "--firewall" ]; then
  PORT=${2:-8600}
  [ "$(id -u)" -eq 0 ] || { echo "Run with sudo for --firewall."; exit 1; }
  if [ "$OS" = "Darwin" ]; then
    # pf anchor: the receiver port from private networks + the SOC's block table
    cat > /etc/pf.anchors/splitshield <<EOF
table <splitshield> persist
block in quick from <splitshield>
pass in proto tcp from { 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16 } to any port $PORT
EOF
    grep -q 'anchor "splitshield"' /etc/pf.conf || printf '\nanchor "splitshield"\nload anchor "splitshield" from "/etc/pf.anchors/splitshield"\n' >> /etc/pf.conf
    pfctl -f /etc/pf.conf 2>/dev/null && pfctl -e 2>/dev/null || true
    echo "pf: port $PORT allowed from private networks; SOC block table <splitshield> ready."
  elif command -v ufw >/dev/null 2>&1; then
    for net in 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16; do ufw allow from "$net" to any port "$PORT" proto tcp; done
    echo "ufw: port $PORT allowed from private networks."
  elif command -v firewall-cmd >/dev/null 2>&1; then
    for net in 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16; do
      firewall-cmd --permanent --add-rich-rule="rule family=ipv4 source address=$net port port=$PORT protocol=tcp accept"
    done
    firewall-cmd --reload
    echo "firewalld: port $PORT allowed from private networks."
  else
    echo "No ufw / firewalld found: allow TCP $PORT from your LAN in your firewall yourself."
  fi
  exit 0
fi

step "1/5 Python 3.12"
PY=""
for c in python3.12 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
    PY=$c; break
  fi
done
if [ -z "$PY" ]; then
  if [ "$OS" = "Darwin" ]; then echo "Install Python 3.12: brew install python@3.12 (or python.org), then run again."
  else echo "Install Python 3.12: e.g. sudo apt install python3.12 python3.12-venv (or dnf), then run again."; fi
  exit 1
fi
echo "  using $($PY --version)"
if [ "$OS" = "Darwin" ] && ! { [ -e /opt/homebrew/opt/libomp ] || [ -e /usr/local/opt/libomp ]; }; then
  echo "  LightGBM needs OpenMP on macOS: brew install libomp"
  command -v brew >/dev/null 2>&1 && brew install libomp || true
fi

step "2/5 Libraries (CPU PyTorch, ~1.6 GB download, ~10 min)"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
VPY="$ROOT/.venv/bin/python"
"$VPY" -m pip install --upgrade pip --quiet
n=0
until "$VPY" -m pip install -r requirements-run.txt --extra-index-url https://download.pytorch.org/whl/cpu --quiet; do
  n=$((n + 1)); [ $n -ge 3 ] && { echo "Could not install the libraries - check the internet and run again (it resumes)."; exit 1; }
  echo "  retrying in 10 s ..."; sleep 10
done

step "3/5 Checking that everything loads"
"$VPY" - <<'EOF' || echo "  Some models are missing; the dashboard will still open."
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
EOF

step "4/5 Background jobs (autopilot, backup, updates, live logs)"
"$VPY" -m src.soc.services install || echo "  Some jobs could not be registered - see above."
if [ "$OS" = "Linux" ]; then
  echo "  To read this computer's logs: sudo usermod -aG adm,systemd-journal $USER  (then sign out and in)"
fi

step "5/5 Dashboard"
chmod +x "$ROOT/START.sh"
echo "  Start it any time with: sh START.sh   (opens http://localhost:8501)"
[ "${1:-}" = "--no-open" ] || exec sh "$ROOT/START.sh"
