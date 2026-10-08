"""The Test button for the checks and answer AIs: how long a short line takes, and a full one.

Early in a meeting every check sees one short line; a few minutes in it sees the
whole context window (earlier lines plus the background), which is the slow case
and the one that times out. So the test sends both and puts the times next to
the timeout.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from . import laya
from .answers import answer_messages
from .checks import laya_questions, laya_state, mood_messages, sample_utterance
from .llm import LLMClient
from .proc import log_fn
from .settings import env_key

Progress = Callable[[str, float], None]

# The test waits this long, so a slow server still gets a number instead of a timeout.
TEST_TIMEOUT = 300.0
SHORT_QUESTION = {"promise": {"type": "noul", "instructions": "Does the speaker promise to do something?"}}


def _timed(work: Callable[[], Any]) -> tuple[float, Any]:
    started = time.time()
    result = work()
    return time.time() - started, result


def _timeout_note(slowest: float, timeout: float) -> str:
    if slowest >= timeout:
        return f" That is over the {timeout:.0f}s timeout, so lines will time out in meetings."
    if slowest >= timeout * 0.6:
        return f" That is close to the {timeout:.0f}s timeout, so busy moments will time out."
    return ""


def enabled_checks(settings: dict[str, Any]) -> int:
    count = int(bool(settings.get("mood_enabled"))) + int(bool(settings.get("fact_check_enabled")))
    return count + sum(1 for c in settings.get("custom_checks") or [] if c.get("enabled") and c.get("prompt"))


def laya_speed(settings: dict[str, Any], progress: Progress) -> str:
    log = log_fn(progress)
    url = settings.get("llm_base_url") or laya.DEFAULT_URL
    key = env_key(settings.get("llm_api_key_env"))
    timeout = float(settings.get("llm_timeout_seconds", 30.0))
    background = settings.get("llm_context", "")
    # The questions a real line gets with the checks turned on right now.
    questions = laya_questions(sample_utterance("en"), settings) or SHORT_QUESTION

    def ask(state: str, qs: dict[str, dict]) -> dict[str, Any]:
        return laya.predict(url, state, qs, timeout=TEST_TIMEOUT, api_key=key)

    runs = [
        ("short line, 1 question", sample_utterance(full=False).text, SHORT_QUESTION),
        (f"full context, {len(questions)} questions", laya_state(sample_utterance("en"), background), questions),
        ("same in Latvian", laya_state(sample_utterance("lv"), background), questions),
    ]
    times = []
    for label, state, qs in runs:
        progress(f"Asking Laya: {label}...", -1.0)
        seconds, answers = _timed(lambda: ask(state, qs))
        times.append(seconds)
        log(f"{label}: {seconds:.1f}s ({len(state)} characters)")
        if qs is SHORT_QUESTION:
            log(f"  promise = {answers['promise'].get('noul', 0):.0%} yes")

    device = str(laya.health(url, key).get("device") or "unknown device")
    message = (f"Laya on {device}: short line {times[0]:.1f}s, full context {times[1]:.1f}s, "
               f"in Latvian {times[2]:.1f}s.") + _timeout_note(max(times[1:]), timeout)
    if device == "cpu" and laya.nvidia_gpu():
        message += " It runs on the CPU, press Install to move it to the NVIDIA GPU."
    return message


def llm_speed(settings: dict[str, Any], role: str, progress: Progress) -> str:
    """`settings` are the role's own (role_settings), so llm_* keys either way."""
    log = log_fn(progress)
    client = LLMClient.from_settings(settings)
    timeout = client.timeout
    client.timeout = max(timeout, TEST_TIMEOUT)
    background = settings.get("llm_context", "")
    full = sample_utterance("en")

    progress(f"Asking {client.model}: short sample...", -1.0)
    if role == "answer":
        short, reply = _timed(lambda: client.chat_text("Answer in one short sentence.", "What is the capital of Latvia?"))
        log(f"short: {short:.1f}s: {reply[:200]}")
        progress(f"Asking {client.model}: question with full context...", -1.0)
        long, reply = _timed(lambda: client.chat_text(*answer_messages(full, settings.get("answer_prompt", ""), background)))
        log(f"full context: {long:.1f}s: {reply[:200]}")
        return (f"{client.model}: short question {short:.1f}s, with full context {long:.1f}s."
                + _timeout_note(long, timeout))

    short, reply = _timed(lambda: client.chat_json("Reply with JSON only.", 'Return {"ok": true}'))
    log(f"short: {short:.1f}s: {json.dumps(reply)}")
    progress(f"Asking {client.model}: mood check with full context...", -1.0)
    long, reply = _timed(lambda: client.chat_json(*mood_messages(full, settings.get("mood_prompt", ""), background)))
    log(f"full context: {long:.1f}s: {json.dumps(reply)}")
    # Ollama gets one prompt per check, one after the other.
    checks = max(1, enabled_checks(settings))
    return (f"{client.model}: short line {short:.1f}s, one check with full context {long:.1f}s, "
            f"so about {long * checks:.0f}s per line with {checks} checks on."
            + _timeout_note(long, timeout))
