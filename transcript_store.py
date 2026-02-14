from __future__ import annotations

from threading import Lock
from typing import Optional
import logging


class TranscriptStore:
    """Thread-safe transcript persistence and in-memory ordering."""

    def __init__(self, file_path: str, agent_name: str, logger: Optional[logging.Logger] = None):
        self._file_path = file_path
        self._agent_name = agent_name
        self._entries: list[list] = []
        self._lock = Lock()
        self._logger = logger or logging.getLogger("transcript_store")

    @property
    def file_path(self) -> str:
        return self._file_path

    def set_file_path(self, file_path: str) -> None:
        with self._lock:
            self._file_path = file_path

    def clear(self) -> None:
        with self._lock:
            self._entries = []

    def add(self, user: str, text: str, start_time: float) -> None:
        with self._lock:
            self._entries.append([user, text, start_time])
            self._entries = sorted(self._entries, key=lambda entry: entry[2])

    def snapshot(self) -> list[list]:
        with self._lock:
            return list(self._entries)

    def append_to_file_if_user(self, user: str, text: str) -> None:
        if user == self._agent_name:
            return

        try:
            with open(self._file_path, "a", encoding="utf-8") as f:
                f.write(f"{user}: {text}\\n\\n")
        except Exception as e:
            self._logger.warning("Failed to append transcript to %s: %s", self._file_path, e)

    def to_text(self) -> str:
        parts = []
        for user, text, _ in self.snapshot():
            parts.append(f"{user}: {text}")
        return "\\n".join(parts)
