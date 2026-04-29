# TWS Connection Guide (WSL + VS Code, IDE-agnostic)

Everything needed to get IBKR TWS API connections working from WSL — including from VS Code with the Remote-WSL extension. When handshakes fail (they will), start here before anything else.

This guide is IDE-neutral. The TWS connection logic is the same whether you use VS Code, PyCharm, plain terminal, or Jupyter Lab. The only IDE-specific bit is interpreter selection, called out near the bottom.

---

## TL;DR — the 60-second check

From a WSL terminal in your project directory:

```bash
# 1. Discover the Windows host IP from inside WSL
WIN_HOST=$(ip route show default | awk '{print $3}')
echo "Windows host: $WIN_HOST"

# 2. Can WSL reach Windows host on TWS port 7497 (paper)?
timeout 2 bash -c "echo > /dev/tcp/$WIN_HOST/7497" && echo OPEN || echo CLOSED

# 3. What's actually listening on Windows port 7497?
powershell.exe -Command "netstat -ano | Select-String 'LISTEN' | Select-String '7497'"
#   Want: LISTENING  <TWS_PID>  (a Java process — TWS)
#   Bad:  LISTENING  <svchost_PID>  (Windows portproxy is squatting — see Issue #3)
#   Bad:  nothing listed  (TWS didn't bind — see Issue #2)

# 4. End-to-end probe:
.venv/bin/python scripts/tws_debug.py
```

If section 3 of `tws_debug.py` prints `RECEIVED N bytes` with N > 0, TWS is accepting handshakes — you are good.

---

## The normal (working) topology

```
WSL  (e.g. 172.29.x.y, rotates each reboot)
      |  TCP 7497
      v
Windows host  (the WSL→Windows NAT gateway, e.g. 172.29.208.1)
      |
      v
TWS (C:\Jts\tws.exe) listening on 0.0.0.0:7497 (all interfaces)
      |
      v
IBKR data farms (outbound, ports 4000/4001)
```

Two things have to be true:

1. **TWS is bound to `0.0.0.0:7497`** (not just `127.0.0.1`) so WSL can reach it.
2. **WSL's current IP** (from `hostname -I | awk '{print $1}'`) is in TWS's Trusted IP list.

That is it. No portproxy is needed when TWS binds to all interfaces.

---

## The three failure modes (and fixes)

### Issue #1 — Wrong WSL IP in TWS trusted list (easy)

**Symptom:** TCP connects, handshake RST'd immediately, "connection reset by peer."

**Diagnosis:** WSL IPs rotate between reboots. The IP that was trusted last session is not the current one.

```bash
hostname -I | awk '{print $1}'   # current WSL IP
```

Compare to the trusted list in TWS's active config XML (replace the wildcard with your actual hashed dir):

```bash
grep -A1 trustedIPAddresses /mnt/c/Jts/*/tws.xml | head -10
```

**Fix:** In TWS → *File → Global Configuration → API → Settings* → Trusted IPs → add current WSL IP → Apply → OK. No restart needed.

---

### Issue #2 — TWS didn't open the API port at all

**Symptom:** `netstat -ano | findstr 7497` on Windows shows **nothing listening**, or shows *only* a `svchost` PID.

**Diagnosis:** TWS is running but silently failed to bind port 7497 because some other process is already on it (usually a stale Windows portproxy, see Issue #3).

**Fix:**
1. Make sure nothing else owns 7497 (Issue #3 below).
2. In TWS → *File → Global Configuration → API → Settings* → uncheck **"Enable ActiveX and Socket Clients"** → Apply → re-check it → Apply → OK. That forces TWS to rebind. **Or** just close TWS and relaunch — on startup it will bind cleanly.

---

### Issue #3 — Stale Windows portproxy squatting on 7497

**Symptom:** TCP connection opens fast, handshake bytes are sent, **server sends zero bytes back** (`struct.error: unpack requires a buffer of 4 bytes` from shinybroker, `TimeoutError` from `ib_async`). Windows netstat shows **only `svchost`** on port 7497, not TWS.

**Root cause:** A portproxy rule like this sometimes gets added during early WSL setup to bridge WSL to TWS-on-localhost:

```
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=7497 connectaddress=127.0.0.1 connectport=7497
```

It persists across reboots. TWS now binds to `0.0.0.0:7497` by default, but the portproxy already grabbed it, so TWS's bind fails silently. Every connection hits the portproxy's dead forwarder, which is why the handshake never gets a reply.

**Check if a portproxy exists:**

```bash
powershell.exe -Command "netsh interface portproxy show all"
```

**Fix — remove the portproxy (needs Windows admin):**

From WSL, this triggers a UAC popup on Windows. Click **Yes**, then re-toggle TWS API as in Issue #2 (or restart TWS):

```bash
powershell.exe -Command "Start-Process powershell.exe -Verb RunAs -ArgumentList '-NoProfile','-Command','netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=7497; exit'"
```

Verify it is gone:

```bash
powershell.exe -Command "netsh interface portproxy show all"
# should print empty or no rules for 7497
```

After that, either toggle the API checkbox in TWS settings or restart TWS. TWS will bind 0.0.0.0:7497 cleanly and WSL can reach it directly with source IP preserved.

---

## Required TWS settings (File → Global Configuration → API → Settings)

| Setting | Value |
|---|---|
| Enable ActiveX and Socket Clients | checked |
| Read-Only API | unchecked (or checked if you only fetch data) |
| Socket port | 7497 (paper) / 7496 (live) |
| Master API client ID | blank or -1 (any client_id OK) |
| Allow connections from localhost only | UNCHECKED |
| Trusted IPs | must include current WSL IP (`hostname -I`) |

Apply → OK. Most changes take effect immediately; some only after a TWS restart.

---

## Things that looked like the bug but were not

- `ib_insync` (retired). Always use **`ib_async`**.
- `shinybroker`'s default host is `127.0.0.1` which is unreachable from WSL NAT. Always pass `host` explicitly (the discovered Windows gateway IP).
- `client_id=1` sometimes collides with a stale zombie session. Bump to `2`, `100`, or higher.
- Windows Firewall is usually not the culprit — if a TCP SYN gets through, the firewall is fine.
- "Allow connections from localhost only" UNCHECKED is **required** for WSL access.

---

## IDE setup

The TWS-side handshake is identical across IDEs, but the work to *get a working WSL Python environment your editor will actually use* differs significantly. PyCharm hides almost all of it; VS Code requires explicit configuration.

### PyCharm (low friction)

PyCharm Professional has first-class WSL interpreter support. The setup that worked for this project's authors:

1. *File → Open* → navigate to the WSL path (e.g. `\\wsl.localhost\Ubuntu\home\<user>\...\FinalProject_FinTech533`) or open from inside WSL.
2. *Settings → Project → Python Interpreter → Add Interpreter → On WSL* → distro `Ubuntu` (or whichever) → select `.venv/bin/python` from this project.
3. Set the working directory of every run config to the project root.
4. Run anything. PyCharm activates the venv, mounts the WSL filesystem, and TWS connections work immediately.

PyCharm Community technically supports this too via manual interpreter setup, but Pro's WSL integration is much smoother. If you have free GitHub Student access, it includes JetBrains Pro.

### VS Code (the path that worked for HW5)

This is the exact recipe that got HW5 (`FinTech533/Homeworks/HW5/Mario_BreakoutStrategy.ipynb`) running in VS Code after a PyCharm-to-VS Code migration. No `.vscode/settings.json` was needed — interpreter and kernel were picked manually each session via the command palette and the notebook's kernel selector. PyCharm hides every one of these steps; VS Code does not.

**Step 1 — Install the WSL Remote extension on Windows VS Code.**
Extension ID: `ms-vscode-remote.remote-wsl`. Without it, VS Code reads WSL files over a slow file share and Python integration silently misbehaves.

**Step 2 — Install these extensions *inside WSL*, not on Windows.**
Once VS Code is connected to WSL, the Extensions pane splits installed extensions into "Local - Installed in Windows" and "WSL: \<distro\>". Every extension below must appear under the WSL section (click "Install in WSL" if it only shows up locally). This is the exact set installed in the WSL distro that ran HW5:

- `ms-python.python` — Python language support
- `ms-python.debugpy` — debugger
- `ms-python.vscode-pylance` — language server
- `ms-python.vscode-python-envs` — venv detection and switching
- `ms-toolsai.jupyter` — notebook execution
- `ms-toolsai.jupyter-keymap`, `ms-toolsai.jupyter-renderers`, `ms-toolsai.vscode-jupyter-cell-tags`, `ms-toolsai.vscode-jupyter-slideshow` — notebook UX
- `mechatroner.rainbow-csv` — convenient for inspecting CSV outputs

**Step 3 — Open the project from a WSL terminal.**
Not Windows Explorer, not `code` from PowerShell — both drop you into local-Windows mode and the rest will not work.

```bash
cd ~/projects/FinTech533/FinalProject_FinTech533
code .
```

Verify the bottom-left status bar reads `WSL: <distro>`. If it reads anything else, close and reopen via `code .` from WSL.

**Step 4 — Keep the project on the WSL filesystem.**
`/home/<user>/...` is fine. `/mnt/c/...` is slow enough to stall Jupyter kernels and pip installs. (HW5 lived at `/home/mht120/projects/FinTech533/FinTech533/Homeworks/HW5`, with the venv one directory up at `Homeworks/.venv`. That nested layout worked fine. The simpler equivalent for this repo is `.venv` at the project root.)

**Step 5 — Create the venv inside WSL:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Step 6 — Pick the interpreter manually via the command palette.**
This is the step that PyCharm autodetects and VS Code does not. Press `Ctrl+Shift+P` → *Python: Select Interpreter* → choose `.venv/bin/python` from this project. VS Code's auto-pick will frequently land on system Python (`/bin/python`) or some other venv it found first; the manual selection is what makes it stick.

**Step 7 — Pin the Jupyter kernel per notebook.**
For every `.ipynb` you open, click the kernel selector in the top-right (it defaults to something like "Python 3.12.x"), and pick the same `.venv` interpreter. VS Code remembers the choice per-notebook after that. Without this step, the notebook executes against whatever kernel the Jupyter extension found, which is usually wrong.

**Step 8 — Use the WSL bash terminal, not PowerShell.**
`Ctrl+\`` opens whatever terminal VS Code used last. If the integrated terminal opens PowerShell, change the default profile in *Terminal → Configure Terminal Profiles* to bash. Without bash, `ip route show default` fails and the Windows host IP discovery in `scripts/tws_debug.py` breaks.

**Step 9 — Run `scripts/tws_debug.py` from the activated venv.**
If the script reports `RECEIVED N bytes` (N > 0) in section 3, the VS Code setup matches the working HW5 setup and any future TWS issue is genuinely on the TWS side.

### Plain terminal / Jupyter Lab

```bash
source .venv/bin/activate
jupyter lab            # or jupyter notebook
# or just: python scripts/your_script.py
```

If `scripts/tws_debug.py` passes from this venv, any editor pointed at the same venv will connect.

---

## Helper: full diagnostic script

`scripts/tws_debug.py` (in this repo) probes TCP, sends a raw handshake, tries several client IDs, and prints exactly where the failure is. Run it from this project's venv:

```bash
.venv/bin/python scripts/tws_debug.py
```

It auto-detects the Windows host IP and the current WSL IP, so there are no Mario-specific paths to edit.

---

## What this project's code uses

- **Library:** `shinybroker` (course-standard) and/or `ib_async` (lower-level).
- **Host:** auto-discovered from `ip route show default` (the WSL→Windows gateway).
- **Port:** `7497` (paper) by default; `7496` for live.
- **Client IDs:** unique per call, typically starting at 100 and incrementing.

Example (shinybroker):

```python
import subprocess, shinybroker as sb

WIN_HOST = subprocess.check_output(
    "ip route show default | awk '{print $3}'", shell=True, text=True
).strip()

r = sb.fetch_historical_data(
    contract=sb.Contract({'symbol':'SPY','secType':'STK','exchange':'SMART','currency':'USD'}),
    endDateTime='20241231 23:59:59',
    durationStr='2 Y', barSizeSetting='1 day', whatToShow='Trades',
    host=WIN_HOST, port=7497, client_id=100,
)
df = r['hst_dta']
```
