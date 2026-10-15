"""Parakeet Redux: local speech-to-text, so transcription needs no API key or subscription.

Parakeet Redux (https://huggingface.co/moondream/parakeet-redux) is a small
(about 180 MB) 1.58-bit version of NVIDIA's Parakeet TDT 0.6B v3. It runs through
the `moondream` package, which needs PyTorch, so it gets the same treatment as
Laya: its own venv under the settings folder made with uv, and a small local
server (`parakeet_server.py`) that speaks the OpenAI transcription API. The
normal transcription client then just points at http://127.0.0.1:8766/v1.

The install picks the PyTorch build for the machine (`--torch-backend auto`:
CUDA when there is an NVIDIA driver, otherwise the CPU one; on Apple silicon
the normal build already has the GPU via MPS). The server then tries the GPU
first and falls back to the CPU when the GPU can't run the model.
"""
from __future__ import annotations

import json
import os
import re
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from . import laya
from .proc import LogTail, log_fn, run_logged
from .settings import PARAKEET_MODEL, PARAKEET_URL, default_config_dir

PACKAGE = "moondream>=2.4.0"
DEVICES = ("auto", "cuda", "mps", "cpu")
SERVER_SCRIPT = "parakeet_server.py"

Progress = Callable[[str, float], None]


class ParakeetError(Exception):
    pass


def home() -> Path:
    return default_config_dir() / "parakeet"


def _venv_bin(name: str) -> Path:
    venv = home() / "venv"
    if sys.platform == "win32":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def installed() -> bool:
    return _venv_bin("python").exists() and (home() / "installed").exists()


def server_script() -> Path:
    """The server file next to this module (the release build ships it as a data file)."""
    return Path(__file__).resolve().with_name(SERVER_SCRIPT)


def _log_path() -> Path:
    return home() / "parakeet-server.log"


# ------------------------------------------------------------------ HTTP

def health(base_url: str, timeout: float = 3.0) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
        raise ParakeetError(f"Parakeet is not reachable at {base_url} ({getattr(e, 'reason', e)})") from e
    if not isinstance(data, dict):
        raise ParakeetError("Parakeet health reply is not an object")
    return data


def _port(base_url: str) -> str:
    """Port of a URL like http://127.0.0.1:8766/v1."""
    hostport = base_url.split("://", 1)[-1].split("/", 1)[0]
    port = hostport.rsplit(":", 1)[-1] if ":" in hostport else ""
    return port if port.isdigit() else "8766"


# ------------------------------------------------------------------ server

def _pid_file() -> Path:
    return home() / "parakeet-server.pid"


def stop_server() -> None:
    try:
        pid = int(_pid_file().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, creationflags=0x08000000)
    else:
        import signal
        try:
            os.killpg(pid, signal.SIGTERM)  # started in its own session, so pid is the group
        except OSError:
            pass
    try:
        _pid_file().unlink()
    except OSError:
        pass


def start_server(base_url: str = PARAKEET_URL, model: str = PARAKEET_MODEL, device: str = "auto") -> bool:
    python = _venv_bin("python")
    script = server_script()
    if not python.exists() or not script.exists():
        return False
    env = dict(os.environ)
    env.update({
        "PARAKEET_HOST": "127.0.0.1",
        "PARAKEET_PORT": _port(base_url),
        "PARAKEET_MODEL": model or PARAKEET_MODEL,
        "PARAKEET_DEVICE": device if device in DEVICES else "auto",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONUNBUFFERED": "1",
    })
    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    kwargs: dict[str, Any] = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "env": env}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED | NEW_GROUP | NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen([str(python), "-I", "-u", str(script)], **kwargs)
    except OSError:
        return False
    try:
        _pid_file().write_text(str(proc.pid), encoding="utf-8")
    except OSError:
        pass
    return True


def wait_until_up(base_url: str, seconds: float = 60.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            health(base_url)
            return True
        except ParakeetError:
            time.sleep(0.5)
    return False


def wait_until_loaded(base_url: str, on_progress: Progress, seconds: float = 1800.0) -> dict[str, Any]:
    """Wait for the model to load (it downloads from Hugging Face the first time)."""
    on_progress("Loading Parakeet (downloads about 180 MB the first time)", -1.0)
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            info = health(base_url)
        except ParakeetError:
            info = {"loading": True}
        if not info.get("loading", True):
            if info.get("error"):
                raise ParakeetError(str(info["error"]))
            return info
        time.sleep(1.0)
    raise ParakeetError("Parakeet did not finish loading")


def ensure_running(base_url: str, model: str, device: str, on_progress: Progress) -> dict[str, Any]:
    """Start the server if it isn't up (or runs with another model/device) and wait for the model.

    The server log (where it says which device it picked, and why not the GPU) goes to the setup log."""
    with LogTail(_log_path(), log_fn(on_progress)):
        try:
            info: Optional[dict[str, Any]] = health(base_url)
        except ParakeetError:
            info = None
        stale = info is not None and (info.get("model") != model or info.get("requested_device") != device)
        failed = info is not None and bool(info.get("error")) and not info.get("loaded")
        if stale or failed:
            # Other settings, or the last load failed (e.g. no internet for the first download): try again.
            on_progress("Restarting Parakeet", -1.0)
            stop_server()
            time.sleep(1.0)
            info = None
        if info is None:
            if not start_server(base_url, model, device):
                raise ParakeetError("Parakeet is not installed. Press Install in Settings > Transcription AI.")
            if not wait_until_up(base_url):
                raise ParakeetError(f"Parakeet did not start, see {_log_path()}")
        return wait_until_loaded(base_url, on_progress)


def status(base_url: str, model: str, device: str) -> dict[str, Any]:
    info: dict[str, Any] = {"installed": installed(), "running": False, "model_ready": False, "models": []}
    try:
        data = health(base_url)
    except ParakeetError as e:
        info["error"] = str(e)
        return info
    info["installed"] = True
    info["running"] = True
    info["model_ready"] = bool(data.get("loaded"))
    info["models"] = [data.get("model")] if data.get("model") else []
    info["device"] = data.get("device")
    if data.get("loading"):
        info["hint"] = "Loading the model..."
    if data.get("error"):
        info["error"] = str(data["error"])
    if data.get("model") != model or data.get("requested_device") != device:
        info["hint"] = "Runs with older settings, press Start to restart it."
    elif data.get("device") == "cpu" and device == "auto" and laya.nvidia_gpu():
        info["hint"] = "Runs on the CPU although there is an NVIDIA GPU. Press Install to switch it to the GPU."
    elif data.get("loaded_model") and data.get("loaded_model") != model:
        info["hint"] = f"Runs {data['loaded_model']} on the CPU instead (this CPU can't run {model}), slower."
    return info


# ------------------------------------------------------------------ install

def install(on_progress: Progress) -> str:
    """Install moondream + PyTorch into Parakeet's own venv. Returns a short message for the UI."""
    if installed():
        return "Parakeet is already installed"
    if sys.platform == "darwin" and platform.machine() != "arm64":
        raise ParakeetError("Parakeet needs an Apple silicon Mac (there are no Intel Mac builds of its runtime)")
    uv = laya.ensure_uv(on_progress)
    venv = home() / "venv"
    if venv.exists():
        shutil.rmtree(venv, ignore_errors=True)  # a half-finished earlier try
    laya._run([uv, "venv", "--python", laya.PYTHON_VERSION, str(venv)], on_progress,
              "Setting up Python for Parakeet")
    # PyTorch is the big part. `--torch-backend auto` picks the CUDA build when there is an NVIDIA GPU.
    laya._pip_install(uv, _venv_bin("python"), [PACKAGE], on_progress, "Installing Parakeet and PyTorch")
    (home() / "installed").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return "Parakeet installed"


# ------------------------------------------------------------------ NVIDIA GPU

TORCH_INDEX = "https://download.pytorch.org/whl/"


def driver_cuda_version() -> Optional[tuple[int, int]]:
    """The newest CUDA the NVIDIA driver supports ("CUDA Version: 12.8" in nvidia-smi), or None."""
    if not laya.nvidia_gpu():
        return None
    try:
        kwargs: dict[str, Any] = {"capture_output": True, "text": True, "timeout": 20}
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000
        out = subprocess.run(["nvidia-smi"], **kwargs).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def cuda_backend(version: Optional[tuple[int, int]]) -> Optional[str]:
    """The PyTorch CUDA wheel set for a driver: the newest one it can run (PyTorch 2.8+ ships 12.6 and up)."""
    if version is None:
        return None
    if version >= (13, 0):
        return "cu130"
    if version >= (12, 8):
        return "cu128"
    if version >= (12, 0):
        return "cu126"  # 12.x wheels run on any 12.x driver (CUDA minor version compatibility)
    return None


def torch_build(on_progress: Progress) -> Optional[tuple[str, Optional[str]]]:
    """(torch version, CUDA version or None) in Parakeet's venv; None when there's no venv or torch."""
    python = _venv_bin("python")
    if not python.exists():
        return None
    found: list[str] = []
    code, _ = run_logged([str(python), "-c", "import torch; print('torch=' + torch.__version__ + '|' + str(torch.version.cuda))"],
                         on_progress, "Checking which PyTorch Parakeet has",
                         on_line=lambda line: found.append(line) if line.startswith("torch=") else None)
    if code != 0 or not found:
        return None
    version, cuda = found[-1][len("torch="):].split("|", 1)
    return version, (None if cuda == "None" else cuda)


def ensure_gpu(on_progress: Progress) -> bool:
    """Swap a CPU-only PyTorch for the CUDA build when there is an NVIDIA GPU. True if it did.

    `--torch-backend auto` should already pick CUDA, but an older uv on PATH (or one that can't see the
    driver) quietly installs the CPU build, and then the GPU is never tried."""
    backend = cuda_backend(driver_cuda_version())
    if backend is None:
        return False
    build = torch_build(on_progress)
    if build is None or build[1] is not None:
        return False
    version = build[0].split("+", 1)[0]
    log = log_fn(on_progress)
    log(f"PyTorch {build[0]} is the CPU build, but there is an NVIDIA GPU: installing the {backend} build")
    uv = laya.ensure_uv(on_progress)
    on_progress("Stopping Parakeet to replace PyTorch", -1.0)
    stop_server()  # Windows locks the loaded torch DLLs
    time.sleep(2.0)
    base = [uv, "pip", "install", "--python", str(_venv_bin("python")), "--reinstall-package", "torch",
            "--index-url", TORCH_INDEX + backend]
    label = f"Installing the GPU build of PyTorch ({backend}, about 2.5 GB)"
    try:
        # Same version first, so the compiled kestrel extensions keep matching it.
        laya._run(base + [f"torch=={version}"], on_progress, label)
    except laya.LayaError:
        log(f"No torch {version} for {backend}, taking the newest one there")
        laya._run(base + ["torch"], on_progress, label)
    return True
