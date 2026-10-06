"""Run installer commands and stream their output to the UI's install log.

Setup steps (uv, pip, winget, brew) can run for minutes; the user sees every
line they print instead of a spinner that looks stuck.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

Progress = Callable[[str, float], None]
LogFn = Callable[[str], None]

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def log_fn(on_progress: Progress) -> LogFn:
    """The engine passes a progress callback that can also take log lines (`.log`)."""
    return getattr(on_progress, "log", None) or (lambda _line: None)


def clean(line: str) -> str:
    return _ANSI.sub("", line).strip()


def split_lines(buffer: str) -> tuple[list[str], str]:
    """Complete lines from `buffer` (split on \\n and on \\r for progress bars), plus the rest."""
    parts = re.split(r"[\r\n]", buffer)
    return parts[:-1], parts[-1]


def run_logged(args: list[str], on_progress: Progress, label: str,
               env: Optional[dict[str, str]] = None,
               on_line: Optional[Callable[[str], None]] = None) -> tuple[int, list[str]]:
    """Run a command, sending each output line to the install log.

    Returns the exit code and the last lines of output (for error messages).
    `on_line` can turn lines into progress (e.g. counting downloaded packages).
    """
    log = log_fn(on_progress)
    on_progress(label, -1.0)
    log(f"$ {' '.join(args)}")
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
        "env": env if env is not None else dict(os.environ),
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # no console window
    proc = subprocess.Popen(args, **kwargs)
    tail: deque[str] = deque(maxlen=20)
    buffer = ""
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
        if not chunk:
            break
        buffer += chunk.decode("utf-8", "replace")
        lines, buffer = split_lines(buffer)
        for raw in lines:
            line = clean(raw)
            if not line:
                continue
            tail.append(line)
            log(line)
            if on_line:
                on_line(line)
    if clean(buffer):
        tail.append(clean(buffer))
        log(clean(buffer))
    code = proc.wait()
    log(f"(exit code {code})")
    return code, list(tail)


class LogTail:
    """Follow a log file another process writes (laya-serve's model download) on a thread."""

    def __init__(self, path: Path, on_line: Callable[[str], None]):
        self.path = path
        self.on_line = on_line
        self._stop = threading.Event()
        self._start_size = path.stat().st_size if path.exists() else 0
        self._thread = threading.Thread(target=self._run, name="log-tail", daemon=True)

    def __enter__(self) -> "LogTail":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        pos = self._start_size
        buffer = ""
        while not self._stop.is_set():
            try:
                if self.path.exists():
                    with open(self.path, "rb") as f:
                        f.seek(pos)
                        data = f.read()
                        pos = f.tell()
                    if data:
                        buffer += data.decode("utf-8", "replace")
                        lines, buffer = split_lines(buffer)
                        for raw in lines:
                            line = clean(raw)
                            if not line:
                                continue
                            self.on_line(line)
            except OSError:
                pass
            self._stop.wait(0.3)
