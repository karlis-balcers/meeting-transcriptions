"""Smoke test a built engine sidecar: start it, connect, ask for hello and devices.

    python packaging/smoke_engine.py dist/windows/engine/meeting-engine.exe

The release workflow runs this on the packaged binary, so a sidecar that
crashes on start (missing module, bad bundle) fails the build instead of
leaving the app stuck on "Starting engine".
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TIMEOUT = 90.0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def dump_logs(home: Path) -> None:
    for log in sorted(home.rglob("*.log")):
        print(f"----- {log} -----")
        print(log.read_text(errors="replace")[-6000:])


def main() -> int:
    exe = Path(sys.argv[1]).resolve()
    home = Path(tempfile.mkdtemp(prefix="engine-smoke-"))
    settings = home / "settings.json"
    settings.write_text(json.dumps({"output_dir": str(home / "out"), "temp_dir": str(home / "tmp")}))
    port = free_port()
    env = dict(os.environ, MT_ENGINE_CRASH_DIR=str(home))
    proc = subprocess.Popen([str(exe), "--port", str(port), "--settings", str(settings)],
                            cwd=home, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    started = time.monotonic()
    sock = None
    try:
        while time.monotonic() - started < TIMEOUT:
            if proc.poll() is not None:
                print(f"Engine exited with code {proc.returncode} before listening")
                print(proc.stdout.read().decode(errors="replace"))
                dump_logs(home)
                return 1
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        if sock is None:
            print(f"Engine did not listen within {TIMEOUT:.0f}s")
            dump_logs(home)
            return 1
        print(f"Engine listening after {time.monotonic() - started:.1f}s")

        sock.settimeout(30)
        sock.sendall(b'{"cmd": "devices", "id": 1}\n')
        reader = sock.makefile("r", encoding="utf-8")
        got_hello = False
        for line in reader:
            msg = json.loads(line)
            if msg.get("type") == "hello":
                got_hello = True
                print(f"hello: version {msg.get('version')}")
            if msg.get("type") == "response" and msg.get("id") == 1:
                print(f"devices: ok={msg.get('ok')} {str(msg.get('data') or msg.get('error'))[:300]}")
                break
        if not got_hello:
            print("No hello from the engine")
            return 1
        return 0
    finally:
        if sock:
            sock.close()
        proc.kill()
        proc.wait()


if __name__ == "__main__":
    sys.exit(main())
