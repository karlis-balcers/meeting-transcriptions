"""Live per-utterance checks on local AI: mood, fact check and user-defined checks.

With Laya all checks of one utterance are asked in a single call as typed
questions (a mood choice, yes/no probabilities). With Ollama or an
OpenAI-compatible server each check is a prompt that returns JSON.

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

from . import laya
from .llm import LLMClient, LLMError
from .profiles import MOOD_VALENCE
from .settings import env_key

logger = logging.getLogger("checks")

MOODS = list(MOOD_VALENCE.keys())
FACT_VERDICTS = ("correct", "incorrect", "doubtful", "no_claim")
MAX_PENDING = 6
# Laya gives probabilities; how sure it has to be before a check counts as a hit.
LAYA_HIT = 0.6
LAYA_FACT = 0.7
LAYA_INTENSITY = ["mild", "clear", "strong"]
# Laya keeps the start of the text and drops whatever doesn't fit its window (512 or 1024
# tokens, minus the question), and every question re-reads the whole text, so a longer text
# costs time on every check. Send about what fits, newest first.
LAYA_STATE_CHARS = 1500


@dataclass
class Utterance:
    id: str
    speaker: str
    text: str
    at: float
    is_me: bool
    context: list[str] = field(default_factory=list)


# For the speed test: a full context window, like late in a real meeting.
SAMPLE_CONTEXT = {
    "en": [
        "Anna: So the main thing for today is the migration plan, we said last week that the database part would be done by the end of the month.",
        "Me: Right, and I think the database part is mostly fine, the risky bit is the card processing service because it still talks to the old cluster.",
        "Peter: We had two incidents there in September, both around the nightly batch, so I would really like to see a rollback plan before we touch it.",
        "Anna: Fair point. Can we split it into two steps, first the read traffic and then the writes, and keep the old cluster warm for a week?",
        "Me: That works for me, but we need someone from the network team, the firewall rules alone took three weeks last time.",
        "Peter: I can ask Laura, she did the rules for the payments gateway, and I think she still has the change templates from that project.",
    ],
    "lv": [
        "Anna: Tātad šodien galvenais ir migrācijas plāns, pagājušajā nedēļā sarunājām, ka datubāzes daļa būs gatava līdz mēneša beigām.",
        "Es: Jā, un man liekas, ka datubāzes daļa ir kārtībā, riskantākā ir karšu apstrādes serviss, jo tas joprojām runā ar veco klasteri.",
        "Pēteris: Septembrī tur bija divi incidenti, abi ap nakts batch apstrādi, tāpēc es gribētu redzēt atjaunošanas plānu pirms to aiztiekam.",
        "Anna: Labi. Vai varam to sadalīt divos soļos, vispirms lasīšanas plūsmu un tad rakstīšanu, un vienu nedēļu paturēt veco klasteri?",
        "Es: Man der, bet mums vajag kādu no tīkla komandas, pagājušajā reizē tikai ugunsmūra noteikumi aizņēma trīs nedēļas.",
        "Pēteris: Es varu pajautāt Laurai, viņa taisīja noteikumus maksājumu vārtejai, un viņai laikam vēl ir izmaiņu veidnes no tā projekta.",
    ],
}
SAMPLE_LATEST = {
    "en": "Okay, then I will write the rollback plan by Friday and send it to everyone, and Peter, can you check with Laura?",
    "lv": "Labi, tad es līdz piektdienai uzrakstīšu atjaunošanas plānu un nosūtīšu visiem, un Pēteri, vai tu vari parunāt ar Lauru?",
}


def sample_utterance(lang: str = "en", full: bool = True) -> Utterance:
    """A test line: short and alone, or with a full window of earlier lines."""
    if not full:
        return Utterance("test", "Anna", "Sure, I'll send you the slides tomorrow morning.", 0.0, False)
    return Utterance("test", "Me", SAMPLE_LATEST[lang], 0.0, True, list(SAMPLE_CONTEXT[lang]))


def applies(applies_to: str, is_me: bool) -> bool:
    if applies_to == "me":
        return is_me
    if applies_to == "others":
        return not is_me
    return True


def _context_block(u: Utterance, background: str = "") -> str:
    """The text every check looks at. `background` is the user's extra context for this AI
    (who is in the meeting, project names...), put first so the model reads the lines with it."""
    lines = "\n".join(u.context) if u.context else "(no earlier lines)"
    block = f"Earlier conversation:\n{lines}\n\nLATEST utterance by {u.speaker}:\n{u.text}"
    background = (background or "").strip()
    return f"Background:\n{background}\n\n{block}" if background else block


def laya_state(u: Utterance, background: str = "", max_chars: int = LAYA_STATE_CHARS) -> str:
    """The text Laya reads. The LATEST line comes first so it is never the part cut off;
    earlier lines are dropped oldest first and the background gets what room is left."""
    text = f"LATEST utterance by {u.speaker}:\n{u.text}"
    earlier: list[str] = []
    room = max_chars - len(text) - len("\n\nEarlier conversation:")
    for line in reversed(u.context):
        room -= len(line) + 1
        if room < 0:
            break
        earlier.insert(0, line)
    if earlier:
        text += "\n\nEarlier conversation:\n" + "\n".join(earlier)
    background = (background or "").strip()
    room = max_chars - len(text) - len("\n\nBackground:\n")
    if background and room > 40:
        text += "\n\nBackground:\n" + background[:room]
    return text


def mood_messages(u: Utterance, instruction: str, background: str = "") -> tuple[str, str]:
    system = (
        "You analyse the emotional tone of people in a live meeting transcript. "
        f"{instruction} Reply with JSON only: "
        '{"mood": one of ' + ", ".join(f'"{m}"' for m in MOODS) + ', '
        '"intensity": number 0..1, "reason": "max 8 words"}'
    )
    return system, _context_block(u, background)


def fact_messages(u: Utterance, instruction: str, background: str = "") -> tuple[str, str]:
    system = (
        "You fact-check statements made in a live meeting, using only your own knowledge (no internet). "
        f"{instruction} Reply with JSON only: "
        '{"verdict": "correct" | "incorrect" | "doubtful" | "no_claim", '
        '"claim": "the claim, max 15 words", "note": "one short sentence with the correct info if wrong"}'
    )
    return system, _context_block(u, background)


def custom_messages(u: Utterance, check: dict[str, Any], background: str = "") -> tuple[str, str]:
    system = (
        "You monitor a live meeting transcript and apply one check to the LATEST utterance.\n"
        f"Check '{check['name']}': {check['prompt']}\n"
        'Reply with JSON only: {"hit": true|false, "label": "max 5 words", "note": "one short sentence"}'
    )
    return system, _context_block(u, background)


def laya_questions(u: Utterance, settings: dict[str, Any]) -> dict[str, dict]:
    """All enabled checks for this utterance as Laya questions, keyed by check id."""
    q: dict[str, dict] = {}
    if settings.get("mood_enabled"):
        q["mood"] = {"type": "choice", "instructions": "What is the mood of the speaker of the LATEST utterance?",
                     "criteria": {m: m for m in MOODS}}
        q["intensity"] = {"type": "score", "instructions": "How strong is the speaker's emotion in the LATEST utterance?",
                          "criteria": LAYA_INTENSITY}
    if settings.get("fact_check_enabled") and applies(settings.get("fact_check_applies_to", "everyone"), u.is_me):
        q["fact"] = {"type": "noul", "instructions": (
            "Does the LATEST utterance state a concrete fact (a number, date, name, technical or historical fact) "
            "that is wrong?")}
    for check in settings.get("custom_checks") or []:
        if check.get("enabled") and check.get("prompt") and applies(check.get("applies_to", "everyone"), u.is_me):
            q["custom:" + check["id"]] = {"type": "noul", "instructions": check["prompt"]}
    return q


def parse_laya_mood(answers: dict[str, Any]) -> Optional[dict[str, Any]]:
    mood = answers.get("mood") or {}
    intensity = answers.get("intensity") or {}
    try:
        level = float(intensity.get("score", 1.0)) / (len(LAYA_INTENSITY) - 1)
    except (TypeError, ValueError):
        level = 0.5
    parsed = parse_mood({"mood": mood.get("choice"), "intensity": level})
    if parsed:
        sure = mood.get("answer_confidence", mood.get("confidence"))
        parsed["reason"] = f"{float(sure):.0%} sure" if isinstance(sure, (int, float)) else ""
    return parsed


def _probability(answer: Any) -> float:
    try:
        return float((answer or {}).get("noul", 0.0))
    except (TypeError, ValueError, AttributeError):
        return 0.0


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
                 client_factory: Callable[[dict], LLMClient] = LLMClient.from_settings,
                 laya_predict: Callable[..., dict[str, Any]] = laya.predict):
        self._settings_getter = settings_getter
        self._emit = emit
        self._on_mood = on_mood
        self._on_fact = on_fact
        self._on_hit = on_hit
        self._client_factory = client_factory
        self._laya_predict = laya_predict
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

    def _drop_backlog(self) -> int:
        """Keep only the newest waiting utterance; returns how many were dropped."""
        with self._cond:
            dropped = max(0, len(self._queue) - 1)
            while len(self._queue) > 1:
                self._queue.popleft()
        return dropped

    def _warn(self, error: Exception) -> None:
        if time.time() - self._last_error_at > 30:
            self._last_error_at = time.time()
            self._emit({"type": "status", "level": "warning", "message": f"Local AI: {error}"})
        logger.debug("Local AI call failed: %s", error)

    def _ask(self, client: LLMClient, messages: tuple[str, str]) -> Optional[dict[str, Any]]:
        try:
            return client.chat_json(*messages)
        except LLMError as e:
            self._warn(e)
            return None

    def run_checks(self, u: Utterance) -> None:
        settings = self._settings_getter()
        if not settings.get("llm_enabled"):
            return
        base = {"type": "check", "utterance_id": u.id, "speaker": u.speaker, "at": u.at}
        if settings.get("llm_api") == "laya":
            self._run_laya(u, settings, base)
            return
        client = self._client_factory(settings)
        background = settings.get("llm_context", "")

        if settings.get("mood_enabled"):
            data = self._ask(client, mood_messages(u, settings.get("mood_prompt", ""), background))
            mood = parse_mood(data) if data else None
            if mood:
                if self._on_mood:
                    self._on_mood(u.speaker, mood["mood"], mood["valence"], u.at)
                self._emit({**base, "check_id": "mood", "kind": "mood", "name": "Mood", "result": mood})

        if settings.get("fact_check_enabled") and applies(settings.get("fact_check_applies_to", "everyone"), u.is_me):
            data = self._ask(client, fact_messages(u, settings.get("fact_check_prompt", ""), background))
            fact = parse_fact(data) if data else None
            if fact:
                if self._on_fact:
                    self._on_fact(u.speaker, fact["verdict"])
                self._emit({**base, "check_id": "fact", "kind": "fact", "name": "Fact check", "result": fact})

        for check in settings.get("custom_checks") or []:
            if not check.get("enabled") or not applies(check.get("applies_to", "everyone"), u.is_me):
                continue
            data = self._ask(client, custom_messages(u, check, background))
            hit = parse_custom(data) if data else None
            if hit:
                if self._on_hit:
                    self._on_hit(u.speaker, check["name"])
                self._emit({
                    **base, "check_id": check["id"], "kind": "custom", "name": check["name"],
                    "color": check.get("color"), "result": hit,
                })

    def _run_laya(self, u: Utterance, settings: dict[str, Any], base: dict[str, Any]) -> None:
        questions = laya_questions(u, settings)
        if not questions:
            return
        try:
            answers = self._laya_predict(settings.get("llm_base_url") or laya.DEFAULT_URL,
                                         laya_state(u, settings.get("llm_context", "")), questions,
                                         timeout=float(settings.get("llm_timeout_seconds", 30.0)),
                                         api_key=env_key(settings.get("llm_api_key_env")))
        except laya.LayaTimeout as e:
            # Laya is still busy with that line and answers one at a time, so lines that
            # queued up meanwhile would only time out behind it. Skip to the newest.
            skipped = self._drop_backlog()
            self._warn(laya.LayaError(f"{e}, skipped {skipped} lines to catch up. "
                                      "Test in the Checks AI settings shows how fast it is."))
            return
        except laya.LayaError as e:
            self._warn(e)
            return

        if "mood" in questions:
            mood = parse_laya_mood(answers)
            if mood:
                if self._on_mood:
                    self._on_mood(u.speaker, mood["mood"], mood["valence"], u.at)
                self._emit({**base, "check_id": "mood", "kind": "mood", "name": "Mood", "result": mood})

        if "fact" in questions:
            p = _probability(answers.get("fact"))
            if p >= LAYA_FACT:
                # Laya can only say "this looks wrong", not what is right.
                fact = {"verdict": "doubtful", "claim": u.text[:160],
                        "note": f"Laya thinks this is likely wrong ({p:.0%}). It has no internet, so double-check."}
                if self._on_fact:
                    self._on_fact(u.speaker, fact["verdict"])
                self._emit({**base, "check_id": "fact", "kind": "fact", "name": "Fact check", "result": fact})

        for check in settings.get("custom_checks") or []:
            key = "custom:" + str(check.get("id"))
            if key not in questions:
                continue
            p = _probability(answers.get(key))
            if p < LAYA_HIT:
                continue
            if self._on_hit:
                self._on_hit(u.speaker, check["name"])
            self._emit({
                **base, "check_id": check["id"], "kind": "custom", "name": check["name"],
                "color": check.get("color"), "result": {"label": check["name"], "note": f"{p:.0%} sure"},
            })
