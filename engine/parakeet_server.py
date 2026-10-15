"""Local speech-to-text server for Parakeet Redux, run with the Python of Parakeet's own venv.

It is NOT imported by the engine: `engine/parakeet.py` starts it as a separate
process (`<venv>/python parakeet_server.py`), because it needs PyTorch and the
`moondream` package, which are far too big to bundle into the engine. So this
file uses only the standard library plus moondream.

It speaks the bit of the OpenAI audio API the app uses, so the normal
transcription client works against it unchanged:

  GET  /v1/models                    -> {"data": [{"id": "<model>"}]}
  GET  /v1/health                    -> {"loaded", "loading", "device", "error", ...}
  POST /v1/audio/transcriptions      multipart with a `file` -> {"text": "..."}

`language` and `prompt` fields are accepted and ignored: Parakeet detects the
language by itself (25 European languages) and takes no prompt.

Device: PARAKEET_DEVICE=auto (default) tries an NVIDIA GPU (cuda), then an
Apple GPU (mps), then the CPU, and keeps the first one that loads the model and
gets through a short test clip. A GPU that is too old or has a broken driver
falls back to the CPU instead of failing.

On the CPU the 1.58-bit weights need an int8 matrix-multiply kernel. When
kestrel's own pick says there is none ("scalar", seen on Windows with CPUs
that do have AVX2) the server asks which paths the CPU can run and uses the
best one. When even that fails, it loads the full Parakeet TDT 0.6B v3 on the
CPU instead (same accuracy, a bigger download), so transcription still works.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
import wave
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

MODEL = os.environ.get("PARAKEET_MODEL") or "moondream/parakeet-redux"
DEVICE = (os.environ.get("PARAKEET_DEVICE") or "auto").strip().lower()
HOST = os.environ.get("PARAKEET_HOST") or "127.0.0.1"
PORT = int(os.environ.get("PARAKEET_PORT") or "8766")
MAX_UPLOAD = 200 * 1024 * 1024
# The model Parakeet Redux was made from: plain weights, so it runs on any CPU torch runs on.
FULL_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
# Int8 matrix-multiply paths kestrel has for the ternary weights, best first.
CPU_GEMM_PATHS = ("avx512vnni", "avxvnni", "avx2", "neon-i8mm", "neon-dotprod", "neon-mull")

log = logging.getLogger("parakeet-server")


def device_candidates(requested: str, cuda: bool, mps: bool) -> list[str]:
    """Devices to try, best first. The CPU is always the last resort."""
    if requested in ("cuda", "mps", "cpu"):
        order = [requested]
    else:
        order = (["cuda"] if cuda else []) + (["mps"] if mps else [])
    if "cpu" not in order:
        order.append("cpu")
    return order


def pick_cpu_gemm(current: Optional[str], available) -> Optional[str]:
    """The path to force when kestrel picked none (None/"scalar"): the best one `available(name)` accepts."""
    if current and current != "scalar":
        return None
    for name in CPU_GEMM_PATHS:
        try:
            if available(name):
                return name
        except Exception:
            continue
    return None


def fix_cpu_gemm() -> None:
    """Log the CPU kernel kestrel picked, and force a real one when it fell to the scalar reference."""
    try:
        from kestrel_kernels import ternary
    except Exception as e:
        log.info("No kestrel CPU kernels to check: %s", e)
        return
    cpu = getattr(ternary, "_cpu", None)
    if cpu is None:
        log.info("kestrel_kernels has no CPU extension in this build")
        return
    current = cpu.ternary_gemm_isa()
    log.info("CPU int8 kernel picked by kestrel: %s", current)
    forced = pick_cpu_gemm(current, cpu.gemm_isa_available)
    if forced:
        log.info("Forcing the CPU int8 kernel to %s", forced)
        ternary.set_gemm_isa(forced)
    elif not current or current == "scalar":
        log.warning("This CPU reports none of %s for the 1.58-bit weights", ", ".join(CPU_GEMM_PATHS))


def _write_silence(path: str, seconds: float = 1.0, rate: int = 16000) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))


class Model:
    """Loads the model once, in the background, and runs one transcription at a time."""

    def __init__(self, name: str, requested: str):
        self.name = name
        self.requested = requested
        self.device: Optional[str] = None
        self.loaded_model: Optional[str] = None
        self.error: Optional[str] = None
        self.tried: list[dict[str, str]] = []
        self._speech: Any = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self.started = time.time()

    @property
    def loaded(self) -> bool:
        return self._speech is not None

    def load(self) -> None:
        try:
            import moondream as md
            import torch

            cuda = bool(torch.cuda.is_available())
            mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
            log.info("torch %s (CUDA build: %s), cuda available: %s, mps available: %s",
                     torch.__version__, getattr(torch.version, "cuda", None), cuda, mps)
            if cuda:
                log.info("GPU: %s", torch.cuda.get_device_name(0))
            fix_cpu_gemm()
            probe = os.path.join(tempfile.gettempdir(), f"parakeet-probe-{os.getpid()}.wav")
            _write_silence(probe)
            attempts = [(self.name, d) for d in device_candidates(self.requested, cuda, mps)]
            if self.name != FULL_MODEL:
                attempts.append((FULL_MODEL, "cpu"))  # last resort for a CPU without the int8 kernels
            try:
                for name, device in attempts:
                    if name == FULL_MODEL and name != self.name:
                        log.info("Falling back to %s on the CPU (about 2.5 GB download the first time)", name)
                    log.info("Loading %s on %s", name, device)
                    try:
                        speech = md.photon(name, device=device)
                        speech.transcribe(audio=probe, timestamps="none")
                    except Exception as e:  # unsupported GPU, driver trouble, out of memory...
                        log.warning("%s did not work on %s: %s", name, device, e)
                        label = device if name == self.name else f"{device} ({name})"
                        self.tried.append({"device": label, "error": str(e)[:300]})
                        continue
                    self._speech = speech
                    self.device = device
                    self.loaded_model = name
                    log.info("Ready: %s on %s after %.1fs", name, device, time.time() - self.started)
                    return
            finally:
                try:
                    os.remove(probe)
                except OSError:
                    pass
            self.error = "Could not load the model on any device: " + "; ".join(
                f"{t['device']}: {t['error']}" for t in self.tried)
        except Exception as e:
            log.error("Loading failed: %s\n%s", e, traceback.format_exc())
            self.error = f"Loading failed: {e}"
        finally:
            self._ready.set()

    def transcribe(self, path: str, wait: float = 1800.0) -> str:
        if not self._ready.wait(wait):
            raise RuntimeError("The model is still loading")
        if self._speech is None:
            raise RuntimeError(self.error or "The model is not loaded")
        with self._lock:
            result = self._speech.transcribe(audio=path, timestamps="none")
        return str(result.get("text") or "").strip() if isinstance(result, dict) else ""

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok", "model": self.name, "loaded_model": self.loaded_model,
            "requested_device": self.requested, "device": self.device,
            "loaded": self.loaded, "loading": not self._ready.is_set(), "error": self.error, "tried": self.tried,
        }


def parse_multipart(content_type: str, body: bytes) -> dict[str, tuple[Optional[str], bytes]]:
    """Form fields of a multipart/form-data body: name -> (filename, data)."""
    head = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("latin-1")
    message = BytesParser(policy=policy.HTTP).parsebytes(head + body)
    fields: dict[str, tuple[Optional[str], bytes]] = {}
    if not message.is_multipart():
        return fields
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        data = part.get_payload(decode=True) or b""
        fields[str(name)] = (part.get_filename(), data)
    return fields


class Handler(BaseHTTPRequestHandler):
    server_version = "parakeet-server/1"
    model: Model

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("%s %s", self.address_string(), fmt % args)

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code: int, message: str) -> None:
        self._send(code, {"error": {"message": message, "type": "server_error" if code >= 500 else "invalid_request"}})

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("/health", "/v1/health"):
            self._send(200, self.model.health())
        elif path in ("/models", "/v1/models"):
            self._send(200, {"object": "list", "data": [{"id": self.model.name, "object": "model"}]})
        else:
            self._error(404, f"Unknown path {path}")

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in ("/audio/transcriptions", "/v1/audio/transcriptions"):
            self._error(404, f"Unknown path {path}")
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD:
            self._error(400, "Send the audio as multipart/form-data with a `file` field")
            return
        fields = parse_multipart(self.headers.get("Content-Type", ""), self.rfile.read(length))
        upload = fields.get("file")
        if not upload or not upload[1]:
            self._error(400, "No `file` in the request")
            return
        suffix = os.path.splitext(upload[0] or "")[1] or ".wav"
        fd, tmp = tempfile.mkstemp(prefix="parakeet-", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(upload[1])
            started = time.time()
            text = self.model.transcribe(tmp)
            log.debug("Transcribed %d bytes in %.2fs", len(upload[1]), time.time() - started)
            self._send(200, {"text": text})
        except Exception as e:
            log.warning("Transcription failed: %s", e)
            self._error(500, str(e))
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass


def main() -> int:
    logging.basicConfig(level=os.environ.get("PARAKEET_LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    model = Model(MODEL, DEVICE)
    Handler.model = model
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=model.load, name="load", daemon=True).start()
    log.info("Listening on http://%s:%s/v1 (model %s, device %s)", HOST, PORT, MODEL, DEVICE)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
