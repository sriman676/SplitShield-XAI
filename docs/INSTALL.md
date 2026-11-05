# 🚀 Installation, setup and troubleshooting

[← Back to the main README](../README.md)

Everything needed to install and run SplitShield-XAI on Windows.

## 💻 Requirements

| | Minimum |
|---|---|
| OS | Windows 10/11 (64-bit) |
| RAM | 8 GB to run the SOC; 16 GB to retrain (the largest jobs need about 6 GB free) |
| Disk | about 4 GB once installed (libraries included); about 170 GB to rebuild every dataset |
| Python | 3.12 (installed for you by `INSTALL.cmd`) |
| GPU | optional (CUDA PyTorch is used when present) |
| Internet | once, for installation; the SOC itself runs offline. The update agent uses it daily (on by default; switch it off in Settings → Updates). |
| Admin rights | only for Sysmon, scheduled tasks and live packet capture |

## 📦 First: the trained models

The trained models and the two Word reports are too large for the repository. Download
**SplitShield-XAI-models.zip** from the repository's **Releases** page and unzip it into
the project folder, so that `artifacts/` and `results/` sit next to `README.md`
(choose *replace* if asked). Without it the installer warns "Some models are missing" and
the dashboard opens with no detectors.

## 🚀 One-step install

▶️ **Double-click `INSTALL.cmd`.** That is all.

It installs Python 3.12 if it is missing, creates `.venv`, installs the runtime
libraries (CPU PyTorch, about 1.6 GB, about 10 minutes, retried on network errors),
checks that every detector, the Sigma rules and the assistant load, adds a
**SplitShield SOC** desktop shortcut and opens the dashboard at
`http://localhost:8501` (this PC only).

| Afterwards | |
|---|---|
| ▶️ Start the SOC | Double-click **`START.cmd`** or the desktop shortcut |
| 🔬 Rebuild everything from the datasets (research + training, many hours) | Double-click `LAUNCH.cmd` |

## 🐧 Linux and macOS

```sh
sh install.sh            # Python 3.12 check, .venv, libraries, checks, background jobs, dashboard
sh START.sh              # start the dashboard later (http://localhost:8501)
```

- Python 3.12 must be installed first (`sudo apt install python3.12 python3.12-venv`,
  `dnf`, or `brew install python@3.12`). On macOS, LightGBM also needs
  `brew install libomp` (the installer tries it).
- Background jobs are systemd user units (Linux) or launchd agents (macOS). On Linux,
  run `loginctl enable-linger $USER` once so they also run while you are signed out.
- To read this computer's logs: Linux `sudo usermod -aG adm,systemd-journal $USER`
  (then sign out and in); macOS: give the .venv Python Full Disk Access.

## 🛰️ Agents on other machines

1. Dashboard → **Settings → Agents & API**: create an agent key (shown once) and note
   the certificate fingerprint.
2. Copy the agent files from that page to the machine and run the installer there:
   - Windows: `powershell -ExecutionPolicy Bypass -File install_agent.ps1` (as Administrator)
   - Linux / macOS: `sudo sh install_agent.sh`
3. It asks for the receiver address (`https://THIS-PC-IP:8600`), the key and the
   fingerprint, tests the connection and starts with the computer.

On the SOC computer, start the receiver (Agents & API, or
`python -m src.soc.collector serve`) and allow port 8600 from the local network once:
`scripts\allow_collector_firewall.ps1` as Administrator (Windows) or
`sudo sh install.sh --firewall` (Linux / macOS). Details: [ENTERPRISE.md](ENTERPRISE.md).

## 🩺 Troubleshooting

| Problem | Fix |
|---|---|
| `INSTALL.cmd` stops during the library download | Run it again: it resumes and retries. Check the internet connection or proxy. |
| "Python 3.12 is needed" | Install it from python.org (tick *Add to PATH*), then run `INSTALL.cmd` again |
| A job stops with "low memory" | Close other programs; large jobs run one at a time (≥ 6 GB free RAM) |
| Windows Defender quarantines dataset files | Expected for attack recordings (log text containing attack tools); results report how many files were lost, and no exclusion is added automatically |
| Dashboard does not open | Check `http://localhost:8501`. If another program uses port 8501, run `.venv\Scripts\python.exe -m streamlit run dashboard.py --server.port 8502` |
| Security / Sysmon log cannot be read | Once, as administrator: `net localgroup "Event Log Readers" %USERNAME% /add`, then sign out and in (the live watcher then reads them with normal rights) |
| New Sigma rules not used after an update | Restart the dashboard and the monitor; running processes keep the rules they loaded |


## Optional: OWASP Core Rule Set (web attack rules)

The SOC can check web-server access logs with the OWASP Core Rule Set. Windows Defender
flags the CRS rule files (they contain attack strings to look for), so exclude one folder
first:

1. **Windows Security** → **Virus & threat protection** → **Manage settings**.
2. **Exclusions** → **Add or remove exclusions** → **Add an exclusion** → **Folder**.
3. Choose `<install folder>\third_party\coreruleset`.

Then switch the update agent on (Settings) or run `python -m scripts.update_agent --force`;
it fetches CRS and refreshes it monthly. Without the exclusion CRS stays off and everything
else works as before. Turn CRS off in Settings (`crs.enabled`).


## Optional: the organisation's own certificate for the log receiver

By default the receiver makes a self-signed certificate and agents pin its SHA-256
fingerprint. To use a certificate from your own CA instead:

1. Put the certificate and key (PEM) on the SOC computer and set them in Settings, or in
   `settings.json`: `"collector": {"cert_file": "C:/certs/soc.pem", "key_file": "C:/certs/soc.key"}`.
   Restart the receiver.
2. Python agents (Linux / macOS): in `agent.json` set `"ca_file": "/path/to/company-ca.pem"`
   (or `"system"` for the system trust store); the fingerprint becomes optional (if given,
   it is checked too).
3. Windows (PowerShell) agent: leave the fingerprint empty; Windows checks the certificate
   against the Windows certificate store, where your company CA is installed.

An agent with neither a fingerprint nor a CA refuses to send.
