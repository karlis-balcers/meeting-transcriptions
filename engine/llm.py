"""Local LLM access: Ollama by default, or any OpenAI-compatible local server.

Uses only the standard library so the engine has no extra dependency for it.
Also knows how to find, start and install Ollama and pull a model, which backs
the "Install" button in the UI.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("llm")

OLLAMA_WINDOWS_INSTALLER = "https://ollama.com/download/OllamaSetup.exe"
OLLAMA_MAC_ZIP = "https://ollama.com/download/Ollama-darwin.zip"
OLLAMA_LINUX_SCRIPT = "https://ollama.com/install.sh"


class LLMError(Exception):
    pass


def _request(url: str, payload: Optional[dict] = None, timeout: float = 10.0) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:300]
        except Exception:
            pass
        raise LLMError(f"HTTP {e.code} from {url}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"Cannot reach {url}: {e}") from e
    try:
        return json.loads(body) if body else {}
    except ValueError as e:
        raise LLMError(f"Invalid JSON from {url}") from e


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (models like to wrap it in prose)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except ValueError:
        pass
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:i + 1])
                        if isinstance(value, dict):
                            return value
                    except ValueError:
                        pass
                    break
        start = text.find("{", start + 1)
    raise LLMError("Model reply had no JSON object")


class LLMClient:
    def __init__(self, api: str, base_url: str, model: str, timeout: float = 30.0):
        self.api = api
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_settings(cls, settings: dict) -> "LLMClient":
        return cls(
            api=settings.get("llm_api", "ollama"),
            base_url=settings.get("llm_base_url", "http://127.0.0.1:11434"),
            model=settings.get("llm_model", "llama3.2:3b"),
            timeout=float(settings.get("llm_timeout_seconds", 30.0)),
        )

    def _openai_base(self) -> str:
        return self.base_url if self.base_url.endswith("/v1") else self.base_url + "/v1"

    def list_models(self) -> list[str]:
        if self.api == "ollama":
            data = _request(f"{self.base_url}/api/tags", timeout=3.0)
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        data = _request(f"{self._openai_base()}/models", timeout=3.0)
        return [m.get("id", "") for m in data.get("data", []) if m.get("id")]

    def has_model(self, models: list[str]) -> bool:
        if self.api != "ollama":
            return True
        want = self.model if ":" in self.model else self.model + ":latest"
        return want in models or self.model in models

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        if self.api == "ollama":
            data = _request(
                f"{self.base_url}/api/chat",
                {
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.1},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                },
                timeout=self.timeout,
            )
            content = (data.get("message") or {}).get("content", "")
        else:
            data = _request(
                f"{self._openai_base()}/chat/completions",
                {
                    "model": self.model,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                },
                timeout=self.timeout,
            )
            choices = data.get("choices") or [{}]
            content = (choices[0].get("message") or {}).get("content", "")
        return extract_json(content)

    def pull(self, on_progress: Callable[[str, float], None]) -> None:
        """Download the configured model through Ollama, streaming progress."""
        if self.api != "ollama":
            raise LLMError("Model download only works with Ollama")
        req = urllib.request.Request(
            f"{self.base_url}/api/pull",
            data=json.dumps({"model": self.model, "stream": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("error"):
                        raise LLMError(event["error"])
                    total = event.get("total") or 0
                    done = event.get("completed") or 0
                    on_progress(event.get("status", ""), (done / total) if total else -1.0)
        except (urllib.error.URLError, OSError) as e:
            raise LLMError(f"Model download failed: {e}") from e


# ---------------------------------------------------------------- Ollama install

def find_ollama() -> Optional[str]:
    found = shutil.which("ollama")
    if found:
        return found
    candidates: list[Path] = []
    if sys.platform == "win32":
        local = os.getenv("LOCALAPPDATA")
        if local:
            candidates.append(Path(local) / "Programs" / "Ollama" / "ollama.exe")
        candidates.append(Path("C:/Program Files/Ollama/ollama.exe"))
    elif sys.platform == "darwin":
        candidates += [
            Path("/Applications/Ollama.app/Contents/Resources/ollama"),
            Path.home() / "Applications" / "Ollama.app" / "Contents" / "Resources" / "ollama",
            Path("/opt/homebrew/bin/ollama"),
            Path("/usr/local/bin/ollama"),
        ]
    else:
        candidates += [Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama")]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _detached_popen(args: list[str]) -> None:
    kwargs: dict[str, Any] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED | NEW_GROUP | NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(args, **kwargs)


def start_ollama_server(binary: Optional[str] = None) -> bool:
    binary = binary or find_ollama()
    if not binary:
        return False
    try:
        _detached_popen([binary, "serve"])
        return True
    except OSError as e:
        logger.warning("Could not start ollama serve: %s", e)
        return False


def _download(url: str, dest: Path, on_progress: Callable[[str, float], None]) -> None:
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last = 0.0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if time.time() - last > 0.3:
                last = time.time()
                on_progress("Downloading Ollama", done / total if total else -1.0)


def install_ollama(on_progress: Callable[[str, float], None]) -> str:
    """Install Ollama for this platform. Returns a short message for the UI."""
    if find_ollama():
        return "Ollama is already installed"

    if sys.platform == "win32":
        if shutil.which("winget"):
            on_progress("Installing Ollama with winget", -1.0)
            result = subprocess.run(
                ["winget", "install", "-e", "--id", "Ollama.Ollama", "--silent",
                 "--accept-source-agreements", "--accept-package-agreements"],
                capture_output=True, text=True, creationflags=0x08000000,
            )
            if result.returncode == 0 or find_ollama():
                return "Ollama installed"
            logger.warning("winget install failed (%s): %s", result.returncode, result.stdout[-400:])
        dest = Path(tempfile.gettempdir()) / "OllamaSetup.exe"
        _download(OLLAMA_WINDOWS_INSTALLER, dest, on_progress)
        on_progress("Running the Ollama installer", -1.0)
        subprocess.run([str(dest), "/SILENT", "/NORESTART"], check=False)
        return "Ollama installed" if find_ollama() else "Finish the Ollama installer, then press Check"

    if sys.platform == "darwin":
        if shutil.which("brew"):
            on_progress("Installing Ollama with Homebrew", -1.0)
            result = subprocess.run(["brew", "install", "ollama"], capture_output=True, text=True)
            if result.returncode == 0 or find_ollama():
                return "Ollama installed"
            logger.warning("brew install failed (%s): %s", result.returncode, result.stderr[-400:])
        dest = Path(tempfile.gettempdir()) / "Ollama-darwin.zip"
        _download(OLLAMA_MAC_ZIP, dest, on_progress)
        on_progress("Unpacking Ollama", -1.0)
        apps = Path.home() / "Applications"
        apps.mkdir(exist_ok=True)
        # zipfile drops the executable bits, so let the system unzip handle the app bundle.
        result = subprocess.run(["ditto", "-x", "-k", str(dest), str(apps)], capture_output=True, text=True)
        if result.returncode != 0:
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(apps)
        subprocess.run(["open", str(apps / "Ollama.app")], check=False)
        return "Ollama installed" if find_ollama() else "Ollama downloaded, finish its setup window"

    on_progress("Installing Ollama", -1.0)
    script = Path(tempfile.gettempdir()) / "ollama-install.sh"
    _download(OLLAMA_LINUX_SCRIPT, script, on_progress)
    result = subprocess.run(["sh", str(script)], capture_output=True, text=True)
    if result.returncode != 0:
        raise LLMError("Linux install needs sudo, run: curl -fsSL https://ollama.com/install.sh | sh")
    return "Ollama installed"


def status(settings: dict) -> dict[str, Any]:
    """What the UI shows on the local AI chip."""
    client = LLMClient.from_settings(settings)
    info: dict[str, Any] = {
        "enabled": bool(settings.get("llm_enabled")),
        "api": client.api,
        "base_url": client.base_url,
        "model": client.model,
        "installed": True if client.api != "ollama" else bool(find_ollama()),
        "running": False,
        "model_ready": False,
        "models": [],
    }
    try:
        models = client.list_models()
        info["running"] = True
        info["installed"] = True
        info["models"] = models
        info["model_ready"] = client.has_model(models)
    except LLMError as e:
        info["error"] = str(e)
    return info
