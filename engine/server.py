"""Localhost socket the Godot UI talks to.

Protocol: newline-delimited JSON over TCP on 127.0.0.1.
  UI -> engine: {"id": 7, "cmd": "start", ...args}
  engine -> UI: {"type": "response", "id": 7, "ok": true, "data": {...}}  (reply to a command)
                {"type": "<event>", ...}                                  (pushed events)
Events: hello, state, status, transcript, level, active_speaker, stats,
speaker_stats, speaker_renamed, check, llm, profiles, settings, devices.
The llm_* setup commands and llm events carry a "role": "llm" (checks AI,
the default) or "answer" (answer AI).
"""
from __future__ import annotations

import json
import logging
import socket
import socketserver
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

from . import __version__, laya, llm, speedtest
from .session import Session
from .settings import SettingsStore, default_config_dir, env_key, role_settings

logger = logging.getLogger("server")


class EventHub:
    """Fan-out of events to every connected UI client."""

    def __init__(self):
        self._lock = threading.Lock()
        self._clients: list[socket.socket] = []

    def add(self, sock: socket.socket) -> None:
        with self._lock:
            self._clients.append(sock)

    def remove(self, sock: socket.socket) -> int:
        with self._lock:
            if sock in self._clients:
                self._clients.remove(sock)
            return len(self._clients)

    def send(self, message: dict[str, Any], only: Optional[socket.socket] = None) -> None:
        data = (json.dumps(message, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        with self._lock:
            targets = [only] if only is not None else list(self._clients)
            for sock in targets:
                try:
                    sock.sendall(data)
                except OSError:
                    pass


class Engine:
    """Command handling, separate from the socket so it can be tested directly."""

    def __init__(self, store: SettingsStore, hub: EventHub, session_factory: Callable[..., Session] = Session):
        self.store = store
        self.hub = hub
        self.session = session_factory(store, hub.send)
        self._llm_busy = threading.Lock()
        self._laya_autostarted = False
        # Install/download output, kept so a UI that (re)connects mid-task can show it.
        self._llm_log: deque[str] = deque(maxlen=500)
        self._llm_current: Optional[dict[str, Any]] = None

    def hello(self) -> dict[str, Any]:
        return {
            "type": "hello",
            "version": __version__,
            "settings": self.store.public(),
            "state": self.session.state(),
            "stats": self.session.stats.snapshot(),
        }

    def _background(self, name: str, fn: Callable[[], Any]) -> None:
        def run():
            try:
                result = fn()
                if isinstance(result, dict) and result.get("ok") is False:
                    self.hub.send({"type": "status", "level": "error", "message": result.get("error", "Failed")})
            except Exception as e:
                logger.exception("%s failed: %s", name, e)
                self.hub.send({"type": "status", "level": "error", "message": f"{name} failed: {e}"})
        threading.Thread(target=run, name=name, daemon=True).start()

    def _devices(self) -> dict[str, Any]:
        result = self.session.list_devices()
        return {k: v for k, v in result.items() if not k.startswith("_")}

    def _role(self, role: Any) -> str:
        return "answer" if role == "answer" else "llm"

    def _role_settings(self, role: str) -> dict[str, Any]:
        return role_settings(self.store.get(), role)

    def llm_status(self, role: str = "llm") -> dict[str, Any]:
        settings = self._role_settings(role)
        info = llm.status(settings)
        # Laya runs as our own background server; bring it up once when it's installed but not running.
        if (role == "llm" and info["api"] == "laya" and info["enabled"] and info["installed"]
                and not info["running"] and not self._laya_autostarted):
            self._laya_autostarted = True
            if laya.start_server(info["base_url"]):
                info["starting"] = True
                threading.Thread(target=lambda: laya.wait_until_up(info["base_url"]) and self.llm_status(),
                                 name="laya-start", daemon=True).start()
        self.hub.send({"type": "llm", "state": "status", "role": role, **info})
        return info

    def _llm_task(self, name: str, work: Callable[[Callable[[str, float], None]], str],
                  refresh: bool = True, role: str = "llm") -> dict[str, Any]:
        if not self._llm_busy.acquire(blocking=False):
            return {"ok": False, "error": "AI setup already running"}

        started = time.time()
        log_path = default_config_dir() / "local-ai-setup.log"

        def log(line: str) -> None:
            line = line.rstrip()[:500]
            self._llm_log.append(line)
            self.hub.send({"type": "llm", "state": "log", "role": role, "line": line})
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass

        def progress(message: str, fraction: float) -> None:
            if self._llm_current is None or self._llm_current.get("message") != message:
                log(f"== {message}")
            self._llm_current = {"message": message, "progress": fraction, "started": started, "role": role}
            self.hub.send({"type": "llm", "state": "busy", "role": role, "message": message, "progress": fraction,
                           "elapsed": round(time.time() - started)})

        progress.log = log  # type: ignore[attr-defined]  # proc.log_fn picks this up

        def run():
            self._llm_log.clear()
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("", encoding="utf-8")
            except OSError:
                pass
            log(f"{name} started {time.strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                message = work(progress)
                log(f"Done: {message}")
                self.hub.send({"type": "llm", "state": "done", "role": role, "message": message})
            except Exception as e:
                logger.warning("%s failed: %s", name, e)
                log(f"Failed: {e}")
                self.hub.send({"type": "llm", "state": "error", "role": role, "message": str(e)})
            finally:
                self._llm_current = None
                self._llm_busy.release()
                if refresh:
                    self.llm_status(role)

        threading.Thread(target=run, name=name, daemon=True).start()
        return {"ok": True}

    def _enable(self, role: str) -> None:
        key = "answer_enabled" if role == "answer" else "llm_enabled"
        if not self.store.get().get(key):
            self.store.update({key: True})
            self.hub.send({"type": "settings", "settings": self.store.public()})

    def _install_laya(self, progress: Callable[[str, float], None]) -> str:
        laya.install(progress)
        on_gpu = laya.ensure_gpu(progress)
        base_url = self.store.get().get("llm_base_url") or laya.DEFAULT_URL
        try:
            laya.health(base_url)
        except laya.LayaError:
            progress("Starting Laya", -1.0)
            laya.start_server(base_url)
            if not laya.wait_until_up(base_url):
                raise laya.LayaError(f"Laya installed but did not start, see {laya.home() / 'laya-serve.log'}")
        laya.warm_up(base_url, progress)
        self._enable("llm")
        return "Local AI ready (Laya, now on the GPU)" if on_gpu else "Local AI ready (Laya)"

    def _install_and_pull(self, progress: Callable[[str, float], None], role: str = "llm") -> str:
        settings = self._role_settings(role)
        if settings.get("llm_api") == "laya":
            return self._install_laya(progress)
        message = llm.install_ollama(progress)
        client = llm.LLMClient.from_settings(self._role_settings(role))
        if client.api != "ollama":
            return message
        # Make sure the server runs, then download the model.
        for attempt in range(30):
            try:
                client.list_models()
                break
            except llm.LLMError:
                if attempt == 0:
                    progress("Starting Ollama", -1.0)
                    llm.start_ollama_server()
                threading.Event().wait(1.0)
        else:
            return message + ", but it is not running yet. Start Ollama and press Check."
        progress(f"Downloading model {client.model}", 0.0)
        client.pull(progress)
        self._enable(role)
        return f"{'Answer' if role == 'answer' else 'Local'} AI ready ({client.model})"

    def transcribe_check(self, args: dict[str, Any]) -> dict[str, Any]:
        """Ask the transcription server for its models: proves the URL and key work, costs nothing."""
        settings = self.store.get()
        base_url = str(args.get("transcribe_base_url", settings["transcribe_base_url"]) or "").strip().rstrip("/")
        key_env = str(args.get("transcribe_api_key_env", settings["transcribe_api_key_env"]) or "").strip()
        key = settings.get("openai_api_key") or env_key(key_env)
        url = (base_url or "https://api.openai.com/v1") + "/models"
        if not key and not base_url:
            name = key_env or "an environment variable"
            return {"ok": False, "error": f"No API key: paste one above or set {name}"}
        try:
            data = llm._request(url, timeout=10.0, api_key=key)
        except llm.LLMError as e:
            return {"ok": False, "error": str(e)}
        models = [m.get("id", "") for m in (data.get("data") or []) if isinstance(m, dict)]
        model = str(args.get("transcript_model") or settings["transcript_model"])
        found = "found" if model in models else ("not in the list" if models else "list is empty")
        return {"ok": True, "message": f"Connected to {url.rsplit('/', 1)[0]} ({len(models)} models, {model} {found})"}

    def handle(self, cmd: str, args: dict[str, Any]) -> Any:
        s = self.session
        if cmd == "hello":
            return self.hello()
        if cmd == "get_settings":
            return self.store.public()
        if cmd == "save_settings":
            values = dict(args.get("settings") or {})
            if not values.get("openai_api_key"):
                values.pop("openai_api_key", None)
            self.store.update(values)
            public = self.store.public()
            self.hub.send({"type": "settings", "settings": public})
            return public
        if cmd == "devices":
            return self._devices()
        if cmd == "select_device":
            kind = args.get("kind")
            if kind not in ("input", "output"):
                raise ValueError("kind must be input or output")
            self.store.update({f"{kind}_device_index": args.get("index"), f"{kind}_device_name": args.get("name", "")})
            return {"ok": True}
        if cmd == "start":
            self._background("start", lambda: s.start(args.get("language")))
            return {"ok": True}
        if cmd == "stop":
            self._background("stop", s.stop)
            return {"ok": True}
        if cmd == "mute":
            s.set_mute(bool(args.get("muted")))
            return {"ok": True}
        if cmd == "split":
            s.split()
            return {"ok": True}
        if cmd == "rename_speaker":
            return s.rename_speaker(str(args.get("old") or ""), str(args.get("new") or ""))
        if cmd == "profiles":
            return s.profiles()
        if cmd == "rename_profile":
            from .profiles import ProfileBook

            ProfileBook(self.store.get()["output_dir"]).rename(str(args.get("old") or ""), str(args.get("new") or ""))
            self.hub.send({"type": "profiles", "profiles": s.profiles()})
            return {"ok": True}
        if cmd == "delete_profile":
            s.delete_profile(str(args.get("name") or ""))
            return {"ok": True}
        if cmd == "transcribe_check":
            return self.transcribe_check(args)
        role = self._role(args.get("role"))
        if cmd.startswith("llm_"):
            # The setup buttons act on the server type / URL / model as set in the form, saved or not.
            # The form always sends them as llm_*; for the answer AI they land in answer_*.
            prefix = "answer_" if role == "answer" else "llm_"
            server = {prefix + k: args["llm_" + k] for k in ("api", "base_url", "model", "api_key_env")
                      if args.get("llm_" + k) is not None}
            current = self.store.get()
            if any(current.get(k) != v for k, v in server.items()):
                self.store.update(server)
                self.hub.send({"type": "settings", "settings": self.store.public()})
        if cmd == "llm_status":
            return self.llm_status(role)
        if cmd == "llm_log":
            current = self._llm_current
            return {"lines": list(self._llm_log), "busy": current is not None,
                    "role": current.get("role") if current else None,
                    "message": current.get("message") if current else None,
                    "progress": current.get("progress") if current else None,
                    "elapsed": round(time.time() - current["started"]) if current else None}
        if cmd == "llm_install":
            return self._llm_task("llm-install", lambda progress: self._install_and_pull(progress, role), role=role)
        settings = self._role_settings(role)
        on_laya = settings.get("llm_api") == "laya"
        laya_url = settings.get("llm_base_url") or laya.DEFAULT_URL
        if cmd == "llm_pull" and on_laya:
            return self._llm_task("llm-pull", lambda progress: (laya.warm_up(laya_url, progress), "Laya model ready")[1])
        if cmd == "llm_start" and on_laya:
            started = laya.start_server(laya_url)
            return {"ok": started, "error": None if started else "Laya is not installed, press Install"}
        if cmd == "llm_test" and on_laya:
            return self._llm_task("llm-test", lambda progress: speedtest.laya_speed(settings, progress), refresh=False)
        client = llm.LLMClient.from_settings(settings)
        if cmd == "llm_pull":
            return self._llm_task("llm-pull", lambda progress: (client.pull(progress), f"Model {client.model} ready")[1],
                                  role=role)
        if cmd == "llm_start":
            started = llm.start_ollama_server()
            return {"ok": started, "error": None if started else "Ollama is not installed"}
        if cmd == "llm_test":
            return self._llm_task("llm-test", lambda progress: speedtest.llm_speed(settings, role, progress),
                                  refresh=False, role=role)
        if cmd == "ping":
            return {"pong": True}
        raise ValueError(f"Unknown command: {cmd}")


# Commands that can take seconds (audio device scan, HTTP to a local AI server
# that may be down, disk). Commands on one connection otherwise run in order,
# and the UI sends all of these right after connecting, so in order they would
# stack up and the window would sit half-empty while they finish.
SLOW_COMMANDS = {"devices", "llm_status", "profiles", "transcribe_check"}


class _Handler(socketserver.StreamRequestHandler):
    server: "EngineServer"

    def _respond(self, cmd: str, msg: dict[str, Any]) -> None:
        engine = self.server.engine
        req_id = msg.get("id")
        try:
            data = engine.handle(cmd, msg)
            engine.hub.send({"type": "response", "id": req_id, "cmd": cmd, "ok": True, "data": data},
                            only=self.connection)
        except Exception as e:
            logger.warning("Command %s failed: %s", cmd, e)
            engine.hub.send({"type": "response", "id": req_id, "cmd": cmd, "ok": False, "error": str(e)},
                            only=self.connection)

    def handle(self) -> None:
        engine = self.server.engine
        hub = engine.hub
        hub.add(self.connection)
        logger.info("UI connected from %s", self.client_address)
        hub.send(engine.hello(), only=self.connection)
        try:
            for raw in self.rfile:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    cmd = str(msg.get("cmd", ""))
                except ValueError:
                    continue
                if cmd in SLOW_COMMANDS:
                    # Answered in their own thread so a slow one doesn't hold up the rest.
                    threading.Thread(target=self._respond, args=(cmd, msg), name=cmd, daemon=True).start()
                else:
                    self._respond(cmd, msg)
        except OSError:
            pass
        finally:
            remaining = hub.remove(self.connection)
            logger.info("UI disconnected (%s left)", remaining)
            if remaining == 0 and self.server.exit_when_alone:
                threading.Thread(target=self.server.shutdown, daemon=True).start()


class EngineServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, engine: Engine, port: int, exit_when_alone: bool = False):
        self.engine = engine
        self.exit_when_alone = exit_when_alone
        super().__init__(("127.0.0.1", port), _Handler)
