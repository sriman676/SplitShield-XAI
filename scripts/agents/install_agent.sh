#!/bin/sh
# Install the SplitShield-XAI agent on Linux (systemd) or macOS (launchd). Run as root:
#
#   sudo sh install_agent.sh            # install + setup (asks address, key, fingerprint)
#   sudo sh install_agent.sh --remove   # uninstall (config and spool are kept)
#
# Needs python3 (3.8+). The agent runs as root so it can read auth logs and the
# journal / unified log; its folder /opt/splitshield-agent is readable by root only.
set -eu
DEST=/opt/splitshield-agent
HERE=$(cd "$(dirname "$0")" && pwd)
OS=$(uname -s)

if [ "$(id -u)" -ne 0 ]; then echo "Please run with sudo."; exit 1; fi
command -v python3 >/dev/null 2>&1 || { echo "python3 is required (3.8 or newer)."; exit 1; }

if [ "${1:-}" = "--remove" ]; then
  if [ "$OS" = "Darwin" ]; then
    launchctl bootout system/ai.splitshield.agent 2>/dev/null || true
    rm -f /Library/LaunchDaemons/ai.splitshield.agent.plist
  else
    systemctl disable --now splitshield-agent 2>/dev/null || true
    rm -f /etc/systemd/system/splitshield-agent.service
    systemctl daemon-reload
  fi
  echo "Removed. $DEST (config, spool) was kept; delete it yourself if you want."
  exit 0
fi

mkdir -p "$DEST"
cp "$HERE/splitshield_agent.py" "$DEST/"
chmod 700 "$DEST"
python3 "$DEST/splitshield_agent.py" --setup --home "$DEST" || {
  echo "Setup did not connect - fix the address / key / fingerprint and run again."; exit 1; }

if [ "$OS" = "Darwin" ]; then
  cat > /Library/LaunchDaemons/ai.splitshield.agent.plist <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>ai.splitshield.agent</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/python3</string><string>$DEST/splitshield_agent.py</string>
    <string>--home</string><string>$DEST</string></array>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$DEST/agent.log</string>
  <key>StandardErrorPath</key><string>$DEST/agent.log</string>
</dict></plist>
EOF
  launchctl bootstrap system /Library/LaunchDaemons/ai.splitshield.agent.plist
  echo "Installed: launchd service ai.splitshield.agent (starts at boot, restarts if it stops)."
  echo "macOS: give python3 Full Disk Access (System Settings > Privacy) if log files are unreadable."
else
  cat > /etc/systemd/system/splitshield-agent.service <<EOF
[Unit]
Description=SplitShield-XAI log agent
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=$(command -v python3) $DEST/splitshield_agent.py --home $DEST
Restart=always
RestartSec=10
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=$DEST
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable --now splitshield-agent
  echo "Installed: systemd service splitshield-agent (starts at boot, restarts if it stops)."
fi
