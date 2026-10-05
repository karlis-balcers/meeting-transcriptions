"""Live per-utterance checks on a local LLM: mood, fact check and user-defined checks.

Each finished utterance is queued; one worker thread runs the enabled checks
against the local model and emits results. When the model can't keep up the
oldest pending utterances are dropped, so results stay close to "now".
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Condition, Event, Thread
from typing import Any, Callable, Optional

from .llm import LLMClient, LLMError
from .profiles import MOOD_VALENCE

logger = logging.getLogger("checks")

MOODS = list(MOOD_VALENCE.keys())
FACT_VERDICTS = ("correct", "incorrect", "doubtful", "no_claim")
MAX_PENDING = 6


@dataclass
class Utterance:
    id: str
    speaker: str
    text: str
    at: float
    is_me: bool
    context: list[str] = field(default_factory=list)


def applies(applies_to: str, is_me: bool) -> bool:
    if applies_to == "me":
        return is_me
    if applies_to == "others":
        return not is_me
    return True


def _context_block(u: Utterance) -> str:
    lines = "\n".join(u.context) if u.context else "(no earlier lines)"
    return f"Earlier conversation:\n{lines}\n\nLATEST utterance by {u.speaker}:\n{u.text}"


def mood_messages(u: Utterance, instruction: str) -> tuple[str, str]:
    system = (
        "You analyse the emotional tone of people in a live meeting transcript. "
        f"{instruction} Reply with JSON only: "
        '{"mood": one of ' + ", ".join(f'"{m}"' for m in MOODS) + ', '
        '"intensity": number 0..1, "reason": "max 8 words"}'
    )
    return system, _context_block(u)


def fact_messages(u: Utterance, instruction: str) -> tuple[str, str]:
    system = (
        "You fact-check statements made in a live meeting, using only your own knowledge (no internet). "
        f"{instruction} Reply with JSON only: "
        '{"verdict": "correct" | "incorrect" | "doubtful" | "no_claim", '
        '"claim": "the claim, max 15 words", "note": "one short sentence with the correct info if wrong"}'
    )
    return system, _context_block(u)


def custom_messages(u: Utterance, check: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You monitor a live meeting transcript and apply one check to the LATEST utterance.\n"
        f"Check '{check['name']}': {check['prompt']}\n"
        'Reply with JSON only: {"hit": true|false, "label": "max 5 words", "note": "one short sentence"}'
    )
    return system, _context_block(u)


def parse_mood(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    mood = str(data.get("mood") or "").strip().lower()
    if mood not in MOOD_VALENCE:
        return None
    try:
        intensity = min(1.0, max(0.0, float(data.get("intensity", 0.5))))
    except (TypeError, ValueError):
        intensity = 0.5
    return {
        "mood": mood,
        "intensity": round(intensity, 2),
        "valence": round(MOOD_VALENCE[mood] * (0.5 + intensity / 2), 2),
        "reason": str(data.get("reason") or "")[:80],
    }


def parse_fact(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    verdict = str(data.get("verdict") or "").strip().lower().replace(" ", "_")
    if verdict not in FACT_VERDICTS or verdict == "no_claim":
        return None
    claim = str(data.get("claim") or "").strip()
    if not claim:
        return None
    return {"verdict": verdict, "claim": claim[:160], "note": str(data.get("note") or "")[:240]}


def parse_custom(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    hit = data.get("hit")
    if isinstance(hit, str):
        hit = hit.strip().lower() in ("true", "yes", "1")
    if not hit:
        return None
    return {"label": str(data.get("label") or "")[:60], "note": str(data.get("note") or "")[:240]}


class CheckRunner:
    def __init__(self, settings_getter: Callable[[], dict], emit: Callable[[dict], None],
                 on_mood: Optional[Callable[[str, str, float, float], None]] = None,
                 on_fact: Optional[Callable[[str, str], None]] = None,
                 on_hit: Optional[Callable[[str, str], None]] = None,
                 client_factory: Callable[[dict], LLMClient] = LLMClient.from_settings):
        self._settings_getter = settings_getter
        self._emit = emit
        self._on_mood = on_mood
        self._on_fact = on_fact
        self._on_hit = on_hit
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
        self._thread = Thread(target=self._run, name="checks", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._queue.clear()
            self._cond.notify_all()

    def submit(self, utterance: Utterance) -> None:
        settings = self._settings_getter()
        if not settings.get("llm_enabled"):
            return
        with self._cond:
            self._queue.append(utterance)
            while len(self._queue) > MAX_PENDING:
                dropped = self._queue.popleft()
                logger.debug("Check backlog full, skipping utterance %s", dropped.id)
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
                self.run_checks(utterance)
            except Exception as e:  # never let the worker die
                logger.exception("Check run failed: %s", e)

    def _ask(self, client: LLMClient, messages: tuple[str, str]) -> Optional[dict[str, Any]]:
        try:
            return client.chat_json(*messages)
        except LLMError as e:
            if time.time() - self._last_error_at > 30:
                self._last_error_at = time.time()
                self._emit({"type": "status", "level": "warning", "message": f"Local AI: {e}"})
            logger.debug("LLM call failed: %s", e)
            return None

    def run_checks(self, u: Utterance) -> None:
        settings = self._settings_getter()
        if not settings.get("llm_enabled"):
            return
        client = self._client_factory(settings)
        base = {"type": "check", "utterance_id": u.id, "speaker": u.speaker, "at": u.at}

        if settings.get("mood_enabled"):
            data = self._ask(client, mood_messages(u, settings.get("mood_prompt", "")))
            mood = parse_mood(data) if data else None
            if mood:
                if self._on_mood:
                    self._on_mood(u.speaker, mood["mood"], mood["valence"], u.at)
                self._emit({**base, "check_id": "mood", "kind": "mood", "name": "Mood", "result": mood})

        if settings.get("fact_check_enabled") and applies(settings.get("fact_check_applies_to", "everyone"), u.is_me):
            data = self._ask(client, fact_messages(u, settings.get("fact_check_prompt", "")))
            fact = parse_fact(data) if data else None
            if fact:
                if self._on_fact:
                    self._on_fact(u.speaker, fact["verdict"])
                self._emit({**base, "check_id": "fact", "kind": "fact", "name": "Fact check", "result": fact})

        for check in settings.get("custom_checks") or []:
            if not check.get("enabled") or not applies(check.get("applies_to", "everyone"), u.is_me):
                continue
            data = self._ask(client, custom_messages(u, check))
            hit = parse_custom(data) if data else None
            if hit:
                if self._on_hit:
                    self._on_hit(u.speaker, check["name"])
                self._emit({
                    **base, "check_id": check["id"], "kind": "custom", "name": check["name"],
                    "color": check.get("color"), "result": hit,
                })
