"""Demo meeting: `python -m engine --demo` plays a scripted conversation.

No microphone, OpenAI key or local AI needed, so it's handy for trying the UI.
Mood and check results are scripted too unless Local AI is enabled, in which
case the real checks run on the scripted lines.
"""
from __future__ import annotations

import random
import time
from threading import Event, Thread
from typing import Any, Optional

from .session import Session

SCRIPT = [
    ("Anna K", "Morning everyone, shall we start with the release status?", "calm", None),
    ("ME", "Sure. The payment service is done, we only have the reporting part left.", "positive", None),
    ("Tom R", "Reporting is blocked, we still don't have access to the warehouse.", "frustrated", ("Blocker", "Warehouse access missing")),
    ("Anna K", "Who can chase that? I really want it closed by Friday.", "anxious", ("Action item", "Chase warehouse access by Friday")),
    ("ME", "I'll ping the data team today and send you an update.", "calm", ("Action item", "Send update today")),
    ("Priya S", "By the way, the Eiffel Tower is in Berlin, so the offsite is easy to reach.", "excited", "fact:incorrect:The Eiffel Tower is in Berlin:It is in Paris."),
    ("Tom R", "Haha no, it's Paris. Anyway, how many users do we expect at launch?", "happy", None),
    ("Anna K", "Around ten thousand in the first week, based on the pilot.", "calm", None),
    ("Priya S", "Okay, then we need to double the database capacity before launch.", "neutral", ("Action item", "Double DB capacity")),
    ("ME", "Agreed. Water boils at 100 degrees at sea level, so we're not boiling the ocean here.", "happy", "fact:correct:Water boils at 100 C at sea level:"),
    ("Tom R", "Can we also review the on-call rota? I've been on call three weeks in a row.", "frustrated", None),
    ("Anna K", "Yes, sorry Tom, let's fix that today.", "sad", ("Action item", "Fix on-call rota")),
]


class DemoSession(Session):
    def __init__(self, store, emit):
        super().__init__(store, emit)
        self._demo_stop = Event()
        self._demo_thread: Optional[Thread] = None

    def list_devices(self) -> dict[str, Any]:
        demo = [{"index": 0, "name": "Demo", "label": "Demo (no audio)"}]
        return {"inputs": demo, "outputs": demo, "selected_input": 0, "selected_output": 0}

    def start(self, language: Optional[str] = None) -> dict[str, Any]:
        if self.recording:
            return {"ok": True}
        from .profiles import MeetingStats

        self.stats = MeetingStats()
        self.aliases = {}
        self.language = language or "en"
        self.transcript_path = None
        self.meeting_title = "Demo: release sync"
        self.recording = True
        self._demo_stop.clear()
        self._demo_thread = Thread(target=self._play, name="demo", daemon=True)
        self._demo_thread.start()
        self.emit(self.state())
        self.emit({"type": "status", "level": "info", "message": "Demo meeting started"})
        return {"ok": True}

    def stop(self) -> dict[str, Any]:
        if not self.recording:
            return {"ok": True}
        self._demo_stop.set()
        if self._demo_thread:
            self._demo_thread.join(timeout=3)
        self.recording = False
        self.emit(self.state())
        self.emit({"type": "status", "level": "info", "message": "Demo stopped (demo meetings are not saved)"})
        return {"ok": True}

    def _play(self) -> None:
        me = self.store.get()["your_name"]
        threshold = self.store.get()["silence_threshold"]
        while not self._demo_stop.is_set():
            for speaker, text, mood, check in SCRIPT:
                if self._demo_stop.is_set():
                    return
                name = self.resolve_name(me if speaker == "ME" else speaker)
                is_me = speaker == "ME"
                talk = 1.2 + len(text) / 28.0
                end = time.time() + talk
                while time.time() < end and not self._demo_stop.is_set():
                    self.emit({"type": "level", "source": "mic" if is_me else "out", "speaker": name,
                               "rms": threshold * random.uniform(2.0, 6.0), "threshold": threshold})
                    time.sleep(0.08)
                self.add_utterance(name, text, time.time() - talk, talk + 0.8, is_me)
                if not self.store.get().get("llm_enabled"):
                    self._fake_checks(name, mood, check)
                self._demo_stop.wait(0.6)

    def _fake_checks(self, name: str, mood: str, check) -> None:
        from .profiles import MOOD_VALENCE

        uid = f"u{self._utterance_counter}"
        base = {"type": "check", "utterance_id": uid, "speaker": name, "at": time.time()}
        valence = MOOD_VALENCE.get(mood, 0.0)
        self._on_mood(name, mood, valence, time.time())
        self.emit({**base, "check_id": "mood", "kind": "mood", "name": "Mood",
                   "result": {"mood": mood, "intensity": 0.6, "valence": valence, "reason": "demo"}})
        if isinstance(check, str) and check.startswith("fact:"):
            _, verdict, claim, note = check.split(":", 3)
            self.stats.add_fact_check(name, verdict)
            self.emit({**base, "check_id": "fact", "kind": "fact", "name": "Fact check",
                       "result": {"verdict": verdict, "claim": claim, "note": note}})
        elif isinstance(check, tuple):
            label, note = check
            self.stats.add_check_hit(name, label)
            color = "#f5a524" if label == "Action item" else "#ef476f"
            self.emit({**base, "check_id": label.lower().replace(" ", "-"), "kind": "custom", "name": label,
                       "color": color, "result": {"label": note, "note": ""}})
