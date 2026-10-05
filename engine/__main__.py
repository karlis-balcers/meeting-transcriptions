"""Entry point: `python -m engine --port 47321`.

The Godot UI starts this itself (with --exit-when-alone so it goes away with the
window), but it can also be run by hand for debugging.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

from .logging_utils import setup_logging
from .server import Engine, EngineServer, EventHub
from .settings import SettingsStore

DEFAULT_PORT = 47321


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="engine", description="Meeting transcription engine")
    parser.add_argument("--port", type=int, default=int(os.getenv("MT_ENGINE_PORT", DEFAULT_PORT)))
    parser.add_argument("--settings", type=Path, default=None, help="settings.json path")
    parser.add_argument("--import-env", type=Path, default=None,
                        help="old .env to import on first run (default: ./.env)")
    parser.add_argument("--demo", action="store_true", help="play a scripted meeting instead of recording")
    parser.add_argument("--exit-when-alone", action="store_true", help="quit when the last UI disconnects")
    args = parser.parse_args(argv)

    # Packaged without a console (PyInstaller --noconsole on Windows) there is no
    # stdout/stderr; give print and logging somewhere harmless to write.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")

    store = SettingsStore(args.settings, env_file=args.import_env)
    settings = store.get()
    setup_logging(
        log_dir=os.path.join(settings["output_dir"], "logs"),
        level_name=settings["log_level"],
        max_mb=settings["log_file_max_mb"],
        backup_count=settings["log_file_backup_count"],
    )
    log = logging.getLogger("engine")
    log.info("Settings: %s", store.path)

    hub = EventHub()
    if args.demo or os.getenv("MT_ENGINE_DEMO") == "1":
        from .demo import DemoSession

        engine = Engine(store, hub, session_factory=DemoSession)
        log.info("Demo mode: playing a scripted meeting")
    else:
        engine = Engine(store, hub)
    try:
        server = EngineServer(engine, args.port, exit_when_alone=args.exit_when_alone)
    except OSError as e:
        log.error("Cannot listen on 127.0.0.1:%s (%s). Is another engine running?", args.port, e)
        return 2

    def on_signal(_signum, _frame):
        import threading

        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)

    log.info("Engine listening on 127.0.0.1:%s", args.port)
    print(f"ENGINE_READY {args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.3)
    finally:
        engine.session.shutdown()
        server.server_close()
        log.info("Engine stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
