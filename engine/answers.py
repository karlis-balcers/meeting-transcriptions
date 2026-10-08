"""Answer AI: drafts an answer when someone in the meeting asks a question.

This is the third AI role, next to transcription and the live checks. It needs
a model that writes text (Ollama or any OpenAI-compatible server, local or
cloud), so it has its own server, model, key and background context.

Questions are spotted the same way the speaker stats count them (a "?" or a
question word up front). Like the checks, one worker thread works through a
short queue and drops the oldest questions when the model can't keep up.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from threading import Condition, Event, Thread
from typing import Any, Callable, Optional

from .checks import Utterance, applies
from .llm import LLMClient, LLMError
from .profiles import count_questions
from .settings import role_settings

logger = logging.getLogger("answers")

MAX_PENDING = 3
MAX_ANSWER_CHARS = 600


def answer_messages(u: Utterance, instruction: str, background: str = "") -> tuple[str, str]:
    system = "You help me in a live meeting. " + (instruction or "").strip()
    background = (background or "").strip()
    if background:
        system += "\n\nWhat you know about me and this meeting:\n" + background
    lines = "\n".join(u.context) if u.context else "(no earlier lines)"
    user = f"Earlier conversation:\n{lines}\n\nQuestion from {u.speaker}:\n{u.text}\n\nYour answer:"
    return system, user


def is_question(text: str) -> bool:
    return count_questions(text) > 0


class AnswerRunner:
    def __init__(self, settings_getter: Callable[[], dict], emit: Callable[[dict], None],
                 client_factory: Callable[[dict], LLMClient] = LLMClient.from_settings):
        self._settings_getter = settings_getter
        self._emit = emit
        self._client_factory = client_factory
        self._queue: deque[Utterance] = deque()
        self._cond = Condition()
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._last_error_at = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="answers", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._queue.clear()
            self._cond.notify_all()

    def submit(self, utterance: Utterance) -> None:
        settings = self._settings_getter()
        if not settings.get("answer_enabled"):
            return
        if not applies(settings.get("answer_applies_to", "others"), utterance.is_me):
            return
        if not is_question(utterance.text):
            return
        with self._cond:
            self._queue.append(utterance)
            while len(self._queue) > MAX_PENDING:
                self._queue.popleft()
            self._cond.notify()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                while not self._queue and not self._stop.is_set():
                    self._cond.wait(timeout=1.0)
                if self._stop.is_set():
                    return
                utterance = self._queue.popleft()
            try:
                self.answer(utterance)
            except Exception as e:  # never let the worker die
                logger.exception("Answer failed: %s", e)

    def answer(self, u: Utterance) -> Optional[str]:
        settings = self._settings_getter()
        if not settings.get("answer_enabled"):
            return None
        client = self._client_factory(role_settings(settings, "answer"))
        system, user = answer_messages(u, settings.get("answer_prompt", ""), settings.get("answer_context", ""))
        try:
            text = client.chat_text(system, user)
        except LLMError as e:
            if time.time() - self._last_error_at > 30:
                self._last_error_at = time.time()
                self._emit({"type": "status", "level": "warning", "message": f"Answer AI: {e}"})
            logger.debug("Answer AI call failed: %s", e)
            return None
        text = (text or "").strip()[:MAX_ANSWER_CHARS]
        if not text:
            return None
        result: dict[str, Any] = {"label": u.text[:60], "note": text}
        self._emit({"type": "check", "utterance_id": u.id, "speaker": u.speaker, "at": u.at,
                    "check_id": "answer", "kind": "answer", "name": "Answer", "color": "#4cc9f0",
                    "result": result})
        return text
