"""Localhost socket the Godot UI talks to.

Protocol: newline-delimited JSON over TCP on 127.0.0.1.
  UI -> engine: {"id": 7, "cmd": "start", ...args}
  engine -> UI: {"type": "response", "id": 7, "ok": true, "data": {...}}  (reply to a command)
                {"type": "<event>", ...}                                  (pushed events)
Events: hello, state, status, transcript, level, active_speaker, stats,
speaker_stats, speaker_renamed, check, llm, profiles, settings, devices.
"""
from __future__ import annotations

import json
import logging
import socket
import socketserver
import threading
import time
from typing import Any, Callable, Optional

from . import __version__, laya, llm
from .session import Session
from .settings import SettingsStore

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

    def llm_status(self) -> dict[str, Any]:
        settings = self.store.get()
        info = llm.status(settings)
        # Laya runs as our own background server; bring it up once when it's installed but not running.
        if (info["api"] == "laya" and info["enabled"] and info["installed"] and not info["running"]
                and not self._laya_autostarted):
            self._laya_autostarted = True
            if laya.start_server(info["base_url"]):
                info["starting"] = True
                threading.Thread(target=lambda: laya.wait_until_up(info["base_url"]) and self.llm_status(),
                                 name="laya-start", daemon=True).start()
        self.hub.send({"type": "llm", "state": "status", **info})
        return info

    def _llm_task(self, name: str, work: Callable[[Callable[[str, float], None]], str],
                  refresh: bool = True) -> dict[str, Any]:
        if not self._llm_busy.acquire(blocking=False):
            return {"ok": False, "error": "Local AI setup already running"}

        def progress(message: str, fraction: float) -> None:
            self.hub.send({"type": "llm", "state": "busy", "message": message, "progress": fraction})

        def run():
            try:
                message = work(progress)
                self.hub.send({"type": "llm", "state": "done", "message": message})
            except Exception as e:
                logger.warning("%s failed: %s", name, e)
                self.hub.send({"type": "llm", "state": "error", "message": str(e)})
            finally:
                self._llm_busy.release()
                if refresh:
                    self.llm_status()

        threading.Thread(target=run, name=name, daemon=True).start()
        return {"ok": True}

    def _enable_llm(self) -> None:
        if not self.store.get().get("llm_enabled"):
            self.store.update({"llm_enabled": True})
            self.hub.send({"type": "settings", "settings": self.store.public()})

    def _install_laya(self, progress: Callable[[str, float], None]) -> str:
        laya.install(progress)
        base_url = self.store.get().get("llm_base_url") or laya.DEFAULT_URL
        try:
            laya.health(base_url)
        except laya.LayaError:
            progress("Starting Laya", -1.0)
            laya.start_server(base_url)
            if not laya.wait_until_up(base_url):
                raise laya.LayaError(f"Laya installed but did not start, see {laya.home() / 'laya-serve.log'}")
        laya.warm_up(base_url, progress)
        self._enable_llm()
        return "Local AI ready (Laya)"

    def _install_and_pull(self, progress: Callable[[str, float], None]) -> str:
        if self.store.get().get("llm_api") == "laya":
            return self._install_laya(progress)
        message = llm.install_ollama(progress)
        settings = self.store.get()
        client = llm.LLMClient.from_settings(settings)
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
        self._enable_llm()
        return f"Local AI ready ({client.model})"

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
        if cmd == "llm_status":
            return self.llm_status()
        if cmd == "llm_install":
            return self._llm_task("llm-install", self._install_and_pull)
        settings = self.store.get()
        on_laya = settings.get("llm_api") == "laya"
        laya_url = settings.get("llm_base_url") or laya.DEFAULT_URL
        if cmd == "llm_pull" and on_laya:
            return self._llm_task("llm-pull", lambda progress: (laya.warm_up(laya_url, progress), "Laya model ready")[1])
        if cmd == "llm_start" and on_laya:
            started = laya.start_server(laya_url)
            return {"ok": started, "error": None if started else "Laya is not installed, press Install"}
        if cmd == "llm_test" and on_laya:
            def test_laya(progress: Callable[[str, float], None]) -> str:
                progress("Asking Laya...", -1.0)
                started = time.time()
                answers = laya.predict(laya_url, "Sure, I'll send you the slides tomorrow morning.",
                                       {"promise": {"type": "noul", "instructions": "Does the speaker promise to do something?"}},
                                       timeout=300.0)
                return (f"Laya answered in {time.time() - started:.1f}s: "
                        f"promise = {answers['promise'].get('noul', 0):.0%} yes")

            return self._llm_task("llm-test", test_laya, refresh=False)
        if cmd == "llm_pull":
            client = llm.LLMClient.from_settings(self.store.get())
            return self._llm_task("llm-pull", lambda progress: (client.pull(progress), f"Model {client.model} ready")[1])
        if cmd == "llm_start":
            started = llm.start_ollama_server()
            return {"ok": started, "error": None if started else "Ollama is not installed"}
        if cmd == "llm_test":
            client = llm.LLMClient.from_settings(self.store.get())

            def test(progress: Callable[[str, float], None]) -> str:
                progress(f"Asking {client.model}...", -1.0)
                started = time.time()
                reply = client.chat_json("Reply with JSON only.", 'Return {"ok": true}')
                return f"{client.model} answered in {time.time() - started:.1f}s: {json.dumps(reply)}"

            return self._llm_task("llm-test", test, refresh=False)
        if cmd == "ping":
            return {"pong": True}
        raise ValueError(f"Unknown command: {cmd}")


class _Handler(socketserver.StreamRequestHandler):
    server: "EngineServer"

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
                req_id = msg.get("id")
                try:
                    data = engine.handle(cmd, msg)
                    hub.send({"type": "response", "id": req_id, "cmd": cmd, "ok": True, "data": data},
                             only=self.connection)
                except Exception as e:
                    logger.warning("Command %s failed: %s", cmd, e)
                    hub.send({"type": "response", "id": req_id, "cmd": cmd, "ok": False, "error": str(e)},
                             only=self.connection)
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
