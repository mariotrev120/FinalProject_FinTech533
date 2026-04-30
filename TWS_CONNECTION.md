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

## Data-layer gotchas (learned the hard way during the April 2026 probe)

These are not connection bugs — TWS is up and accepting handshakes — but they look like connection bugs and ate hours of debugging time. Documenting so Robert and any future grader running this repo do not repeat them.

### Zombie Python processes hold TWS connections

A Python script that hangs on `fetch_historical_data` (because TWS is silently not responding) does not release its TWS connection when you `Ctrl+C` it or close the parent shell. The next session runs into the dead client_id and the new request hangs too. Symptom: a brand-new diagnostic that should take 5 seconds sits forever on the first call.

Before running any data pull, sweep:

```bash
ps -ef | grep -E "python.*shinybroker|python.*ib_async|python.*fetch" | grep -v grep
pkill -9 -f "shinybroker" 2>/dev/null
```

In the worst case found during the probe, a 24-hour-old Python process from yesterday's first pull was still attached to TWS and blocking every new request silently. Always sweep first.

### `shinybroker` has parsing bugs; use `ib_async` for contract qualification

Specifically: `sb.fetch_contract_details` on a current SPX option throws `KeyError: 'liquidHours'` inside shinybroker's own response parser. The TWS response is fine; shinybroker just can't decode it. The workaround is to use `ib_async`'s `IB.qualifyContracts(contract)` for option contract resolution and conId discovery, then pass the qualified contract back into shinybroker for historical bar pulls if needed.

```python
from ib_async import IB, Option
ib = IB(); ib.connect("172.29.208.1", 7497, clientId=1)
ib.reqMarketDataType(3)  # see "Always set delayed market data type" below
qualified = ib.qualifyContracts(Option("SPX","20260618",5500,"P","CBOE",multiplier="100",currency="USD",tradingClass="SPXW"))
print(qualified[0].conId)  # 846739350 — use this with downstream calls
```

### `ib_async` rejects historical `endDateTime` strings; use `shinybroker` for those

Conversely, `ib_async.IB.reqHistoricalData(..., endDateTime="20180601 23:59:59 US/Eastern")` errors with `Error 10314: End Date/Time format is invalid` even though that format matches `ib_async`'s own error-message example. Cause: ib_async runs a stricter pre-validation than TWS itself. `endDateTime=""` (defaults to now) works, and so does a `datetime` object — but explicit historical strings do not.

shinybroker passes the same string through to TWS without pre-validating, and it works. So the practical division of labor:

- `ib_async` → contract qualification (`qualifyContracts`), connection probes, anything that exercises shinybroker's known parsing bugs.
- `shinybroker` → historical bar pulls with explicit historical `endDateTime` strings.

### Indices: use `whatToShow="TRADES"`, never `MIDPOINT`

Indices (SPX, VIX, VIX3M, VVIX, treasury yields like ^TNX) are computed values, not traded contracts. Asking TWS for `MIDPOINT` bars on `secType="IND"` returns `Error 162: No historical market data for SPX/IND@CBOE MidPoint 1d`. `TRADES` returns the index level series correctly. This is the opposite of options where `MIDPOINT` is often what you want.

### Always call `reqMarketDataType(3)` before any paper-account request

TWS paper accounts default to live market data, which the account isn't actually subscribed to. Set `reqMarketDataType(3)` (delayed feed) on every connection before the first historical request — without it some calls silently return zero bars or hang. With ib_async:

```python
ib.reqMarketDataType(3)
```

shinybroker has no equivalent helper; you'd send the raw message via `ib_socket` or just stick to `ib_async` for the connection initialization step.

### The hard wall: paper accounts cannot pull historical option bars

This was the conclusive finding of the April 2026 probe. Across both `shinybroker` and `ib_async`, against current and expired option contracts, against SPX and SPY, against SMART/CBOE/BEST routing, with every `whatToShow` value (TRADES, MIDPOINT, BID_ASK), TWS paper returned exactly the same error:

```
No data of type EODChart is available for the exchange '<X>' and the security type 'Option' and '<dur>' and '1 day'
```

This is a market-data-subscription wall, not a code bug. A funded IBKR account with the OPRA Top of Book subscription (~$12/month) is required for historical option EOD bars. Paper accounts cannot fetch them at any duration, any whatToShow, any exchange.

Equally definitive: paper accounts return `No security definition has been found for the request` for any option contract whose expiry is in the past, even with `includeExpired=True` set on the Contract object — `includeExpired` works for funded accounts only.

What paper accounts *can* do for an options-strategy backtest:
- Pull SPX, VIX, VIX3M, VVIX index history (TRADES, daily bars, multi-year lookback). ✓
- Pull SPY/TLT/GLD/HYG/LQD equity ETF history. ✓
- Pull current option chain *definitions* via `fetch_sec_def_opt_params` (forward expirations and strike grid). ✓
- Resolve current option *conIds* via `qualifyContracts`. ✓

What paper accounts cannot do:
- Pull historical option bars (current or expired). ✗
- Resolve expired option contracts via symbol/strike/expiry lookup. ✗

The implication for any options-strategy backtest on a paper account: you need either an external historical option price source (yfinance has none for European-style index options, ORATS / OptionMetrics / CBOE LiveVol cost real money) or you reconstruct synthetic option prices analytically from the index level + implied volatility (Black-Scholes from VIX). The latter is the path this project takes; the synthetic-pricing limitation is disclosed in the writeup as a primary risk.

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

### VS Code on WSL (the delta from Vestal's PyCharm tutorial)

Vestal's official ShinyBroker walkthrough ([shinybroker.com](https://shinybroker.com), "ShinyBroker Part I: Hello World") assumes **PyCharm on Windows, with TWS on the same Windows machine**. Run that recipe verbatim and it works on the first try. PyCharm Pro autodetects the venv interpreter, the Jupyter integration picks up the kernel, the terminal is the right shell, and `host='127.0.0.1'` reaches TWS because Python and TWS share a localhost.

This project's authors run the strategy from **VS Code attached to WSL, with TWS on the Windows host**. That setup keeps the data and venv on the Linux filesystem (faster pip, cleaner shell, easier reproducibility for Robert and graders) but introduces a network boundary and a stack of VS Code-specific friction that PyCharm hides.

The list below is *only the deltas* from Vestal's PyCharm path. Every step PyCharm performs implicitly that VS Code does not is called out. No `.vscode/settings.json` is shipped — HW5 ran fine without one; manual interpreter and kernel selection per session is what worked.

---

**Delta 1 — VS Code is not a WSL editor by default.**
PyCharm Pro has built-in WSL interpreter support. VS Code on Windows treats WSL as foreign until you install the WSL Remote extension on the *Windows side* of VS Code. Extension ID: `ms-vscode-remote.remote-wsl`. Without it, VS Code reads WSL files over a slow file share and Python integration silently misbehaves.

**Delta 2 — Extensions must be installed inside WSL, not on Windows.**
PyCharm has one place for extensions. VS Code splits them: once VS Code is connected to WSL, the Extensions pane separates "Local - Installed in Windows" from "WSL: \<distro\>". Every extension below must appear under the WSL section (click "Install in WSL" if it only shows up locally). This is the exact set HW5 ran on:

- `ms-python.python` — Python language support
- `ms-python.debugpy` — debugger
- `ms-python.vscode-pylance` — language server
- `ms-python.vscode-python-envs` — venv detection and switching
- `ms-toolsai.jupyter` — notebook execution
- `ms-toolsai.jupyter-keymap`, `ms-toolsai.jupyter-renderers`, `ms-toolsai.vscode-jupyter-cell-tags`, `ms-toolsai.vscode-jupyter-slideshow` — notebook UX
- `mechatroner.rainbow-csv` — convenient for inspecting CSV outputs

**Delta 3 — Open the project from a WSL bash terminal, not Windows Explorer or PowerShell.**
Vestal's *File → New Project → choose folder* in PyCharm has no equivalent here. The `code` command launched from PowerShell or by double-clicking the folder in Windows Explorer drops VS Code into local-Windows mode, and the rest of these steps will silently fail.

```bash
cd ~/projects/FinTech533/FinalProject_FinTech533
code .
```

Verify the bottom-left status bar reads `WSL: <distro>`. If it reads anything else, close and reopen via `code .` from WSL.

**Delta 4 — Keep the project on the WSL filesystem.**
PyCharm's file paths sit on the same OS as the editor. VS Code over WSL crosses a boundary every file read. `/home/<user>/...` is fine. `/mnt/c/...` is slow enough to stall Jupyter kernels and pip installs.

**Delta 5 — Create the venv inside WSL by hand.**
PyCharm's New Project flow auto-creates `.venv/` and configures the interpreter. VS Code does not. From the WSL terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Delta 6 — Pick the interpreter manually via the command palette.**
The single biggest difference from PyCharm. PyCharm autodetects `.venv/bin/python`; VS Code does not, and its auto-pick frequently lands on system Python (`/bin/python`) or some other venv it found first.

`Ctrl+Shift+P` → *Python: Select Interpreter* → choose `.venv/bin/python` from this project. The active interpreter has a star next to it, and the path should end in `FinalProject_FinTech533/.venv/bin/python`.

**Delta 7 — Pin the Jupyter kernel per notebook.**
PyCharm Pro auto-binds the venv kernel to every notebook in the project. VS Code's Jupyter extension defaults to whatever kernel it found first, which is usually wrong. For every `.ipynb` you open: click the kernel selector in the top-right and pick the same `.venv` interpreter. VS Code remembers the choice per-notebook after that.

**Delta 8 — Use the WSL bash terminal inside VS Code, not PowerShell.**
PyCharm's terminal inherits the project's interpreter and shell. VS Code's `Ctrl+\`` opens whatever terminal it used last; on a fresh Windows VS Code install that defaults to PowerShell. Change the default profile in *Terminal → Configure Terminal Profiles* to bash. Without bash, `ip route show default` fails and the Windows host IP discovery in `scripts/tws_debug.py` breaks.

**Delta 9 — In your code, replace `host='127.0.0.1'` with the WSL gateway IP.**
This is the one delta that is not VS Code setup at all but a code-level change. Vestal's hello-world uses `host='127.0.0.1', port=7497, client_id=10742`. From WSL, `127.0.0.1` is the WSL distro's loopback, not Windows', so it cannot reach TWS. You must pass the Windows host gateway IP instead. Discover it dynamically:

```python
import subprocess
WIN_HOST = subprocess.check_output(
    "ip route show default | awk '{print $3}'", shell=True, text=True
).strip()

# Then in every shinybroker call:
sb.fetch_historical_data(..., host=WIN_HOST, port=7497, client_id=100)
```

Hardcoding the IP works too, but it changes on most WSL reboots, so dynamic discovery is the durable pattern. This applies to *every* shinybroker call — its default `host='127.0.0.1'` is always wrong from WSL.

**Final check — run `scripts/tws_debug.py` from the activated venv.**
If the script reports `RECEIVED N bytes` (N > 0) in section 3, the VS Code-on-WSL setup matches the working HW5 setup and any future TWS issue is genuinely on the TWS side.

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
