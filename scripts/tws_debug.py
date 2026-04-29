"""
TWS connectivity diagnostic, IDE-agnostic.

Probes:
  1. Network context (WSL IP, Windows host gateway, target host:port)
  2. TCP reachability
  3. Raw IBKR API handshake byte-by-byte with a long read timeout
  4. ib_async connection with several client IDs
  5. A checklist of TWS-side settings to verify

The Windows host IP is auto-discovered from `ip route show default`, so this
script is portable across machines and WSL distros. No hardcoded user paths.

Run from the project venv:

    .venv/bin/python scripts/tws_debug.py
"""
from __future__ import annotations

import socket
import struct
import subprocess
import time

PORT = 7497


def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception as e:
        return f"ERR: {e}"


def section(t: str) -> None:
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


def discover_host() -> str:
    host = sh("ip route show default | awk '{print $3}'")
    if not host or host.startswith("ERR"):
        raise RuntimeError(f"Could not discover Windows host IP from default route: {host}")
    return host


HOST = discover_host()


section("1. Network context")
print(f"  WSL IP (add THIS to TWS trusted IPs):   {sh('hostname -I | awk \"{print $1}\"')}")
print(f"  WSL gateway (= Windows host):           {HOST}")
print(f"  Target TWS:                             {HOST}:{PORT}")


section("2. TCP reachability")
s = socket.socket()
s.settimeout(3)
t0 = time.time()
try:
    s.connect((HOST, PORT))
    print(f"  TCP connect: OK in {1000 * (time.time() - t0):.0f} ms")
    s.close()
except Exception as e:
    print(f"  TCP connect FAILED: {e}")
    raise SystemExit(1)


section("3. Raw IBKR API handshake")
s = socket.socket()
s.settimeout(10)
s.connect((HOST, PORT))
body = b"v100..187"
msg = b"API\0" + struct.pack("!I", len(body)) + body
s.sendall(msg)
print(f"  sent {len(msg)} bytes (API-prefix + length-prefixed 'v100..187')")
print("  waiting up to 10s for response...")
try:
    data = s.recv(4096)
    print(f"  RECEIVED {len(data)} bytes: {data!r}")
    if data:
        print("  -> TWS accepted the handshake. Any subsequent code failures are NOT permissions.")
    else:
        print("  -> TWS closed the socket without replying. Classic permission rejection.")
except socket.timeout:
    print("  -> TIMEOUT: TWS held the socket but sent nothing. Likely waiting on a user-accept dialog.")
except ConnectionResetError:
    print("  -> RST from TWS. Usually trusted-IP mismatch or TWS not logged in.")
except Exception as e:
    print(f"  -> {type(e).__name__}: {e}")
finally:
    s.close()


section("4. ib_async attempts with several client IDs")
try:
    from ib_async import IB
except ImportError:
    print("  ib_async not installed, skipping (pip install ib_async if you want this section)")
else:
    for cid in [0, 1, 100, 2000]:
        ib = IB()
        t0 = time.time()
        try:
            ib.connect(HOST, PORT, clientId=cid, timeout=8)
            print(f"  client_id={cid}: CONNECTED in {1000 * (time.time() - t0):.0f} ms  accounts={ib.managedAccounts()}")
            ib.disconnect()
            break
        except Exception as e:
            print(f"  client_id={cid}: {type(e).__name__}: {str(e)[:120]}")


section("5. Things to verify in TWS now")
print("""
  [ ] TWS window is fully logged in (paper trading account name visible in title bar)
  [ ] No dialog popup waiting for a click (alt-tab through every TWS window)
  [ ] File -> Global Configuration -> API -> Settings:
        [x] Enable ActiveX and Socket Clients
        [ ] Allow connections from localhost only  (UNCHECKED)
        [ ] Read-Only API                          (either)
        Socket port: 7497 (paper) / 7496 (live)
        Trusted IP Addresses: must include the WSL IP printed in section 1
  [ ] If you changed ANY of the above: click Apply, then File -> Close TWS,
      wait 10 seconds, and relaunch. Some settings only take effect after restart.
  [ ] While reconnecting, tail the TWS API log on the Windows side:
        C:\\Users\\<your-username>\\Jts\\api.YYYYMMDD.log
      That log states the exact reason for each rejection.
""")
