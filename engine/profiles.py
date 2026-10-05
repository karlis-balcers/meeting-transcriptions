"""Per-speaker stats for the live meeting and long-lived speaker profiles.

`MeetingStats` is built up while a session runs (talk time, words, pace,
questions, interruptions, moods, check hits...). When the session stops it is
merged into `ProfileBook`, a JSON file in the output folder, so every speaker
builds up a profile across meetings.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger("profiles")

PROFILE_FILE_NAME = "speaker-profiles.json"

STOPWORDS = set(
    """
    a about above after again against all also am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further get got had has have having he her
    here hers herself him himself his how i if in into is it its itself just let like me more most my myself no nor
    not now of off on once only or other our ours ourselves out over own really right same she should so some such
    than that the their theirs them themselves then there these they this those through to too under until up very
    was we were what when where which while who whom why will with would you your yours yourself yourselves yeah yes
    okay ok um uh mm hmm gonna wanna think know mean going thing things kind sort maybe well one two also actually
    see say said lot
    """.split()
)

FILLER_PATTERNS = [
    r"\bum+\b", r"\buh+\b", r"\berm\b", r"\bhmm+\b", r"\byou know\b", r"\bi mean\b", r"\bkind of\b",
    r"\bsort of\b", r"\bbasically\b", r"\bactually\b", r"\blike\b",
]
_FILLER_RX = [re.compile(p, re.IGNORECASE) for p in FILLER_PATTERNS]
_WORD_RX = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)

QUESTION_STARTERS = (
    "what", "why", "how", "when", "where", "who", "which", "can", "could", "would", "should", "do", "does", "did",
    "is", "are", "will", "shall", "have", "has",
)

MOOD_VALENCE = {
    "happy": 0.8, "excited": 0.7, "positive": 0.6, "calm": 0.3, "neutral": 0.0, "confused": -0.2,
    "anxious": -0.4, "sad": -0.5, "frustrated": -0.6, "angry": -0.8,
}


def words_in(text: str) -> list[str]:
    return _WORD_RX.findall(text or "")


def count_questions(text: str) -> int:
    count = (text or "").count("?")
    if count:
        return count
    first = (words_in(text)[:1] or [""])[0].lower()
    return 1 if first in QUESTION_STARTERS and len(words_in(text)) > 2 else 0


def count_fillers(text: str) -> int:
    return sum(len(rx.findall(text or "")) for rx in _FILLER_RX)


def topic_words(text: str) -> list[str]:
    return [w.lower() for w in words_in(text) if len(w) > 3 and w.lower() not in STOPWORDS]


class SpeakerMeetingStats:
    def __init__(self, name: str, is_me: bool):
        self.name = name
        self.is_me = is_me
        self.utterances = 0
        self.words = 0
        self.talk_seconds = 0.0
        self.questions = 0
        self.fillers = 0
        self.interruptions = 0
        self.longest_turn_seconds = 0.0
        self.first_seen: Optional[float] = None
        self.last_seen: Optional[float] = None
        self.topics: Counter[str] = Counter()
        self.moods: list[dict[str, Any]] = []
        self.mood_counts: Counter[str] = Counter()
        self.check_hits: Counter[str] = Counter()
        self.fact_checks: Counter[str] = Counter()

    def to_dict(self, total_talk: float) -> dict[str, Any]:
        minutes = self.talk_seconds / 60.0
        last_mood = self.moods[-1] if self.moods else None
        valence = [m["valence"] for m in self.moods[-10:]]
        return {
            "name": self.name,
            "is_me": self.is_me,
            "utterances": self.utterances,
            "words": self.words,
            "talk_seconds": round(self.talk_seconds, 1),
            "talk_share": round(self.talk_seconds / total_talk, 3) if total_talk > 0 else 0.0,
            "wpm": round(self.words / minutes, 1) if minutes > 0.05 else 0.0,
            "questions": self.questions,
            "fillers": self.fillers,
            "interruptions": self.interruptions,
            "longest_turn_seconds": round(self.longest_turn_seconds, 1),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "top_topics": [w for w, _ in self.topics.most_common(8)],
            "mood": last_mood["mood"] if last_mood else None,
            "mood_valence": round(sum(valence) / len(valence), 2) if valence else None,
            "mood_counts": dict(self.mood_counts),
            "mood_history": self.moods[-40:],
            "check_hits": dict(self.check_hits),
            "fact_checks": dict(self.fact_checks),
        }


class MeetingStats:
    """Live stats for one recording session."""

    def __init__(self):
        self._lock = Lock()
        self.speakers: dict[str, SpeakerMeetingStats] = {}
        self.started_at = time.time()
        self._last_turn: Optional[tuple[str, float, float]] = None  # speaker, start, end
        self.transitions: Counter[tuple[str, str]] = Counter()

    def _get(self, name: str, is_me: bool = False) -> SpeakerMeetingStats:
        stats = self.speakers.get(name)
        if stats is None:
            stats = SpeakerMeetingStats(name, is_me)
            self.speakers[name] = stats
        return stats

    def add_utterance(self, speaker: str, text: str, start: float, duration: float, is_me: bool) -> None:
        with self._lock:
            stats = self._get(speaker, is_me)
            words = words_in(text)
            # Chunks end on ~1s of silence, so trim that from the measured talk time.
            spoken = max(0.0, duration - 0.8) if duration > 0 else len(words) / 2.5
            stats.utterances += 1
            stats.words += len(words)
            stats.talk_seconds += spoken
            stats.questions += count_questions(text)
            stats.fillers += count_fillers(text)
            stats.longest_turn_seconds = max(stats.longest_turn_seconds, spoken)
            stats.first_seen = stats.first_seen or start
            stats.last_seen = start + spoken
            stats.topics.update(topic_words(text))

            if self._last_turn is not None:
                prev_speaker, _prev_start, prev_end = self._last_turn
                if prev_speaker != speaker:
                    self.transitions[(prev_speaker, speaker)] += 1
                    if start < prev_end - 0.5:
                        stats.interruptions += 1
            if self._last_turn is None or start + spoken >= self._last_turn[2]:
                self._last_turn = (speaker, start, start + spoken)

    def add_mood(self, speaker: str, mood: str, valence: float, at: float) -> None:
        with self._lock:
            stats = self._get(speaker)
            stats.moods.append({"t": at, "mood": mood, "valence": round(float(valence), 2)})
            stats.mood_counts[mood] += 1

    def add_check_hit(self, speaker: str, check_name: str) -> None:
        with self._lock:
            self._get(speaker).check_hits[check_name] += 1

    def add_fact_check(self, speaker: str, verdict: str) -> None:
        with self._lock:
            self._get(speaker).fact_checks[verdict] += 1

    def rename(self, old: str, new: str) -> None:
        with self._lock:
            if old not in self.speakers or old == new:
                return
            src = self.speakers.pop(old)
            if new in self.speakers:
                dst = self.speakers[new]
                dst.utterances += src.utterances
                dst.words += src.words
                dst.talk_seconds += src.talk_seconds
                dst.questions += src.questions
                dst.fillers += src.fillers
                dst.interruptions += src.interruptions
                dst.longest_turn_seconds = max(dst.longest_turn_seconds, src.longest_turn_seconds)
                dst.topics.update(src.topics)
                dst.moods = sorted(dst.moods + src.moods, key=lambda m: m["t"])
                dst.mood_counts.update(src.mood_counts)
                dst.check_hits.update(src.check_hits)
                dst.fact_checks.update(src.fact_checks)
            else:
                src.name = new
                self.speakers[new] = src
            renamed: Counter[tuple[str, str]] = Counter()
            for (a, b), n in self.transitions.items():
                a = new if a == old else a
                b = new if b == old else b
                if a != b:
                    renamed[(a, b)] += n
            self.transitions = renamed
            if self._last_turn and self._last_turn[0] == old:
                self._last_turn = (new,) + self._last_turn[1:]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = sum(s.talk_seconds for s in self.speakers.values())
            return {
                "started_at": self.started_at,
                "duration_seconds": round(time.time() - self.started_at, 1),
                "speakers": {name: s.to_dict(total) for name, s in self.speakers.items()},
                "transitions": [
                    {"from": a, "to": b, "count": n} for (a, b), n in self.transitions.most_common()
                ],
            }

    def speaker_snapshot(self, name: str) -> Optional[dict[str, Any]]:
        with self._lock:
            stats = self.speakers.get(name)
            if stats is None:
                return None
            total = sum(s.talk_seconds for s in self.speakers.values())
            return stats.to_dict(total)


class ProfileBook:
    """Speaker profiles accumulated across meetings, saved as JSON."""

    def __init__(self, directory: str):
        self._lock = Lock()
        self.path = Path(directory) / PROFILE_FILE_NAME
        self.profiles: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("profiles"), dict):
                return data["profiles"]
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as e:
            logger.warning("Could not read speaker profiles %s: %s", self.path, e)
        return {}

    def save(self) -> None:
        with self._lock:
            data = {"version": 1, "updated_at": time.time(), "profiles": self.profiles}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            logger.error("Could not save speaker profiles to %s: %s", self.path, e)

    def all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return json.loads(json.dumps(self.profiles))

    def merge_meeting(self, meeting: dict[str, Any], meeting_title: Optional[str] = None) -> None:
        with self._lock:
            when = meeting.get("started_at") or time.time()
            for name, s in (meeting.get("speakers") or {}).items():
                if not s.get("utterances"):
                    continue
                p = self.profiles.setdefault(name, {
                    "name": name,
                    "is_me": bool(s.get("is_me")),
                    "first_seen": when,
                    "meetings": 0,
                    "utterances": 0,
                    "words": 0,
                    "talk_seconds": 0.0,
                    "questions": 0,
                    "fillers": 0,
                    "interruptions": 0,
                    "longest_turn_seconds": 0.0,
                    "talk_share_sum": 0.0,
                    "mood_counts": {},
                    "check_hits": {},
                    "fact_checks": {},
                    "topics": {},
                    "history": [],
                })
                p["is_me"] = p.get("is_me") or bool(s.get("is_me"))
                p["meetings"] += 1
                p["last_seen"] = when
                for key in ("utterances", "words", "questions", "fillers", "interruptions"):
                    p[key] = p.get(key, 0) + int(s.get(key) or 0)
                p["talk_seconds"] = round(p.get("talk_seconds", 0.0) + float(s.get("talk_seconds") or 0.0), 1)
                p["talk_share_sum"] = round(p.get("talk_share_sum", 0.0) + float(s.get("talk_share") or 0.0), 3)
                p["longest_turn_seconds"] = max(p.get("longest_turn_seconds", 0.0), float(s.get("longest_turn_seconds") or 0.0))
                for key in ("mood_counts", "check_hits", "fact_checks"):
                    merged = Counter(p.get(key) or {})
                    merged.update(s.get(key) or {})
                    p[key] = dict(merged)
                topics = Counter(p.get("topics") or {})
                topics.update({t: 1 for t in s.get("top_topics") or []})
                p["topics"] = dict(topics.most_common(40))
                history = list(p.get("history") or [])
                history.append({
                    "at": when,
                    "title": meeting_title or "",
                    "talk_seconds": s.get("talk_seconds"),
                    "talk_share": s.get("talk_share"),
                    "wpm": s.get("wpm"),
                    "mood_valence": s.get("mood_valence"),
                })
                p["history"] = history[-50:]
        self.save()

    def rename(self, old: str, new: str) -> None:
        with self._lock:
            if old not in self.profiles or old == new:
                return
            src = self.profiles.pop(old)
            if new not in self.profiles:
                src["name"] = new
                self.profiles[new] = src
            else:
                dst = self.profiles[new]
                for key in ("meetings", "utterances", "words", "questions", "fillers", "interruptions"):
                    dst[key] = dst.get(key, 0) + src.get(key, 0)
                for key in ("talk_seconds", "talk_share_sum"):
                    dst[key] = round(dst.get(key, 0.0) + src.get(key, 0.0), 3)
                for key in ("mood_counts", "check_hits", "fact_checks", "topics"):
                    merged = Counter(dst.get(key) or {})
                    merged.update(src.get(key) or {})
                    dst[key] = dict(merged)
                dst["history"] = sorted((dst.get("history") or []) + (src.get("history") or []), key=lambda h: h["at"])[-50:]
                dst["first_seen"] = min(dst.get("first_seen") or 0, src.get("first_seen") or 0) or dst.get("first_seen")
        self.save()

    def delete(self, name: str) -> None:
        with self._lock:
            self.profiles.pop(name, None)
        self.save()


def summarize_profile(p: dict[str, Any]) -> dict[str, Any]:
    """Derived numbers the UI shows for a saved profile."""
    minutes = float(p.get("talk_seconds") or 0.0) / 60.0
    meetings = max(1, int(p.get("meetings") or 0))
    moods = Counter(p.get("mood_counts") or {})
    total_moods = sum(moods.values())
    valence = (
        sum(MOOD_VALENCE.get(m, 0.0) * n for m, n in moods.items()) / total_moods if total_moods else None
    )
    return {
        "name": p.get("name"),
        "is_me": p.get("is_me", False),
        "meetings": p.get("meetings", 0),
        "talk_minutes": round(minutes, 1),
        "avg_talk_share": round(float(p.get("talk_share_sum") or 0.0) / meetings, 3),
        "wpm": round(float(p.get("words") or 0) / minutes, 1) if minutes > 0.05 else 0.0,
        "questions_per_meeting": round(int(p.get("questions") or 0) / meetings, 1),
        "interruptions": p.get("interruptions", 0),
        "fillers_per_100_words": round(100.0 * int(p.get("fillers") or 0) / max(1, int(p.get("words") or 0)), 1),
        "top_mood": moods.most_common(1)[0][0] if moods else None,
        "mood_valence": round(valence, 2) if valence is not None else None,
        "top_topics": [t for t, _ in Counter(p.get("topics") or {}).most_common(8)],
        "check_hits": p.get("check_hits") or {},
        "fact_checks": p.get("fact_checks") or {},
        "first_seen": p.get("first_seen"),
        "last_seen": p.get("last_seen"),
        "history": (p.get("history") or [])[-12:],
    }
