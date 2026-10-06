"""Laya: a small local decision model (https://pypi.org/project/laya/).

Laya doesn't write text, it answers typed questions about a text in one forward
pass: a choice between labels, a score on a scale, or a yes/no probability. That
fits the live checks well (mood is a choice, every custom check is a yes/no) and
it reads 100+ languages.

It needs PyTorch, which is far too big to bundle into the engine, so the Install
button puts it in its own virtual environment under the settings folder (made
with uv, which also brings its own Python, so nothing has to be installed on
the machine first) and runs `laya-serve` as a local HTTP server, the same way
Ollama runs next to the app.
"""
from __future__ import annotations

import io
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

from .proc import LogTail, log_fn, run_logged
from .settings import default_config_dir

logger = logging.getLogger("laya")

DEFAULT_URL = "http://127.0.0.1:8765"
PACKAGE = "laya[serve]"
PYTHON_VERSION = "3.12"
UV_RELEASE = "https://github.com/astral-sh/uv/releases/latest/download/"
MODEL_MARKER = "model-ready"

Progress = Callable[[str, float], None]


class LayaError(Exception):
    pass


def home() -> Path:
    return default_config_dir() / "laya"


def _venv_bin(name: str) -> Path:
    venv = home() / "venv"
    if sys.platform == "win32":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def find_server() -> Optional[str]:
    """laya-serve from our own install, else one on PATH (e.g. `pip install laya[serve]`)."""
    ours = _venv_bin("laya-serve")
    if ours.exists():
        return str(ours)
    return shutil.which("laya-serve")


# ------------------------------------------------------------------ HTTP client

def _request(url: str, payload: Optional[dict] = None, timeout: float = 10.0) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LayaError(f"Laya returned HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise LayaError(f"Laya is not reachable at {url} ({getattr(e, 'reason', e)})") from e
    except json.JSONDecodeError as e:
        raise LayaError("Laya returned something that isn't JSON") from e


def health(base_url: str) -> dict[str, Any]:
    return _request(f"{base_url.rstrip('/')}/health", timeout=3.0)


def predict(base_url: str, state: str, questions: dict[str, dict], timeout: float = 30.0) -> dict[str, Any]:
    """Ask all questions about `state` in one call; returns the `answers` dict."""
    data = _request(f"{base_url.rstrip('/')}/v1/systemone", {"state": state, "questions": questions},
                    timeout=timeout)
    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, dict):
        raise LayaError("Laya reply had no answers")
    return answers


# ------------------------------------------------------------------ server

def _popen_detached(args: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    kwargs: dict[str, Any] = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "env": env}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED | NEW_GROUP | NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(args, **kwargs)


def start_server(base_url: str = DEFAULT_URL) -> bool:
    server = find_server()
    if not server:
        return False
    port = base_url.rstrip("/").rsplit(":", 1)[-1]
    env = dict(os.environ)
    env.update({
        "LAYA_HOST": "127.0.0.1",
        "LAYA_PORT": port if port.isdigit() else "8765",
        # Load checkpoints on first use, so the server is up right away.
        "LAYA_PRELOAD": "0",
        "LAYA_LOG_LEVEL": "warning",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    })
    try:
        _popen_detached([server], env, home() / "laya-serve.log")
        return True
    except OSError as e:
        logger.warning("Could not start laya-serve: %s", e)
        return False


def wait_until_up(base_url: str, seconds: float = 90.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            health(base_url)
            return True
        except LayaError:
            time.sleep(1.0)
    return False


def status(base_url: str) -> dict[str, Any]:
    info: dict[str, Any] = {"installed": bool(find_server()), "running": False, "model_ready": False,
                            "models": []}
    try:
        data = health(base_url)
        info["running"] = True
        info["installed"] = True
        info["models"] = list(data.get("loaded") or [])
        # Checkpoints download on the first question; warm_up() does that and leaves a marker.
        info["model_ready"] = bool(info["models"]) or (home() / MODEL_MARKER).exists()
        info["device"] = data.get("device")
    except LayaError as e:
        info["error"] = str(e)
    return info


# ------------------------------------------------------------------ install

def _uv_target() -> tuple[str, str]:
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if sys.platform == "win32":
        return ("aarch64-pc-windows-msvc" if arm else "x86_64-pc-windows-msvc"), "zip"
    if sys.platform == "darwin":
        return ("aarch64-apple-darwin" if arm else "x86_64-apple-darwin"), "tar.gz"
    return ("aarch64-unknown-linux-gnu" if arm else "x86_64-unknown-linux-gnu"), "tar.gz"


def _download(url: str, on_progress: Progress, label: str) -> bytes:
    buf = io.BytesIO()
    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        last = 0.0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            buf.write(chunk)
            if time.time() - last > 0.3:
                last = time.time()
                on_progress(label, buf.tell() / total if total else -1.0)
    return buf.getvalue()


def ensure_uv(on_progress: Progress) -> str:
    found = shutil.which("uv")
    if found:
        return found
    exe_name = "uv.exe" if sys.platform == "win32" else "uv"
    dest = home() / "bin" / exe_name
    if dest.exists():
        return str(dest)
    target, ext = _uv_target()
    data = _download(f"{UV_RELEASE}uv-{target}.{ext}", on_progress, "Downloading uv (Python installer)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if ext == "zip":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            member = next(n for n in zf.namelist() if n.endswith(exe_name))
            dest.write_bytes(zf.read(member))
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith("/" + exe_name) or m.name == exe_name)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise LayaError("uv download did not contain the uv binary")
            dest.write_bytes(extracted.read())
        dest.chmod(0o755)
    return str(dest)


def _run(args: list[str], on_progress: Progress, label: str,
         on_line: Optional[Callable[[str], None]] = None) -> None:
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(home() / "python"), NO_COLOR="1")
    code, tail = run_logged(args, on_progress, label, env=env, on_line=on_line)
    if code != 0:
        raise LayaError(f"{label} failed: {' '.join(tail[-3:])}")


class _PipProgress:
    """uv prints "Downloading torch (110.0MiB)" / "Downloaded torch" for the big wheels; count them."""

    def __init__(self, on_progress: Progress, label: str):
        self.on_progress = on_progress
        self.label = label
        self.started: set[str] = set()
        self.done: set[str] = set()

    def __call__(self, line: str) -> None:
        words = line.split()
        if len(words) < 2:
            return
        if words[0] == "Downloading":
            self.started.add(words[1])
        elif words[0] == "Downloaded":
            self.done.add(words[1])
        else:
            return
        waiting = sorted(self.started - self.done)
        detail = f": {', '.join(waiting[:3])}" if waiting else ""
        fraction = len(self.done) / len(self.started) if self.started else -1.0
        self.on_progress(f"{self.label} ({len(self.done)}/{len(self.started)} big downloads{detail})",
                         min(fraction, 0.99))


def install(on_progress: Progress) -> str:
    """Install Laya into its own venv. Returns a short message for the UI."""
    if find_server():
        return "Laya is already installed"
    uv = ensure_uv(on_progress)
    venv = home() / "venv"
    _run([uv, "venv", "--python", PYTHON_VERSION, str(venv)], on_progress, "Setting up Python for Laya")
    python = _venv_bin("python")
    # PyTorch is the big part (a few hundred MB); the log shows each download as uv starts it.
    label = "Installing Laya and PyTorch"
    _run([uv, "pip", "install", "--python", str(python), PACKAGE], on_progress, label,
         on_line=_PipProgress(on_progress, label))
    if not find_server():
        raise LayaError("Laya installed, but laya-serve is missing")
    return "Laya installed"


_PERCENT = re.compile(r"(\d{1,3})%\|")


def warm_up(base_url: str, on_progress: Progress) -> None:
    """The first question downloads the checkpoints from Hugging Face; do it now, not mid-meeting.

    laya-serve does the download, so its log (where Hugging Face prints the
    progress bars) is followed and forwarded while we wait.
    """
    log = log_fn(on_progress)
    label = "Downloading the Laya model (first time only)"
    on_progress(label, -1.0)
    last_logged = [0.0]

    def on_server_line(line: str) -> None:
        match = _PERCENT.search(line)
        if match:
            name = line.split(":", 1)[0][:40]
            on_progress(f"{label}: {name}", int(match.group(1)) / 100.0)
            # Progress bars redraw many times a second; the log gets one every couple of seconds.
            if time.time() - last_logged[0] < 2.0 and not match.group(1) == "100":
                return
            last_logged[0] = time.time()
        log(line)

    question = {"ok": {"type": "noul", "instructions": "Does the speaker promise to do something?"}}
    with LogTail(home() / "laya-serve.log", on_server_line):
        for text in ("Thanks, I'll send the report tomorrow.", "Paldies, es rīt nosūtīšu atskaiti."):
            log(f"Asking Laya: {text}")
            started = time.time()
            answers = predict(base_url, text, question, timeout=1800.0)
            log(f"Answered in {time.time() - started:.1f}s (promise: {answers['ok'].get('noul', 0):.0%})")
    (home() / MODEL_MARKER).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
