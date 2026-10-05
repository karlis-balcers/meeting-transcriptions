"""Recording session: the transcription pipeline from the Python version, driven by the UI.

Flow per Start (same as the old `transcribe.py`):
  mic + output capture threads -> silence-gated WAV chunks -> OpenAI transcription
  -> transcript filter -> speaker naming -> Markdown transcript file
and on top of that, for the Godot UI:
  -> live events (transcript, audio levels, active speaker) -> per-speaker stats
  -> local LLM checks (mood / fact / custom) -> speaker profiles on Stop.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, Optional

from . import audio_capture
from .checks import CheckRunner, Utterance
from .profiles import MeetingStats, ProfileBook, summarize_profile
from .settings import SettingsStore, parse_language_candidates
from .transcript_filter import TranscriptFilter

logger = logging.getLogger("session")

LEVEL_EVENT_INTERVAL = 0.08


def _load_pyaudio():
    if sys.platform == "win32":
        import pyaudiowpatch as pyaudio  # WASAPI loopback on Windows
    else:
        import pyaudio
    return pyaudio


def _normalize_text(text: str) -> str:
    normalized = str(text or "").replace("\\r\\n", "\\n").replace("\\r", "\\n")
    return normalized.replace("\\\\n", "\n").strip()


class Session:
    def __init__(self, store: SettingsStore, emit: Callable[[dict], None]):
        self.store = store
        self.emit = emit
        self._lock = Lock()
        self.recording = False
        self.language: Optional[str] = None
        self.transcript_path: Optional[str] = None
        self.stats = MeetingStats()
        self.aliases: dict[str, str] = {}
        self.meeting_title: Optional[str] = None
        self.active_speaker: Optional[str] = None

        self._stop_capture = Event()
        self._stop_store = Event()
        self._mute = Event()
        self._flush_lock = Lock()
        self._flush_mic: Optional[str] = None
        self._flush_out: Optional[str] = None
        self._threads: list[Thread] = []
        self._capture_threads: list[Thread] = []
        self._pa = None
        self._detector = None
        self._transcriber = None
        self._filter: Optional[TranscriptFilter] = None
        self._file_lock = Lock()
        self._utterance_counter = 0
        self._context: deque[str] = deque(maxlen=50)
        self._last_level_emit = {"mic": 0.0, "out": 0.0}
        self._levels_peak = {"mic": 0.0, "out": 0.0}

        self.checks = CheckRunner(
            settings_getter=self.store.get,
            emit=self.emit,
            on_mood=self._on_mood,
            on_fact=lambda speaker, verdict: self.stats.add_fact_check(speaker, verdict),
            on_hit=lambda speaker, name: self.stats.add_check_hit(speaker, name),
        )
        self.checks.start()

    # ------------------------------------------------------------------ devices

    def list_devices(self) -> dict[str, Any]:
        settings = self.store.get()
        try:
            pyaudio = _load_pyaudio()
        except ImportError as e:
            return {"inputs": [], "outputs": [], "error": f"Audio library missing: {e}"}
        p = pyaudio.PyAudio()
        try:
            catalog = audio_capture.enumerate_recording_devices(p, platform_name=sys.platform)
        finally:
            p.terminate()
        selected_in = audio_capture.pick_preferred_device(
            catalog["inputs"], settings.get("input_device_index"), settings.get("input_device_name"),
            catalog.get("default_input"),
        )
        selected_out = audio_capture.pick_preferred_device(
            catalog["outputs"], settings.get("output_device_index"), settings.get("output_device_name"),
            catalog.get("default_output"),
        )

        def brief(d: dict) -> dict:
            return {"index": d.get("index"), "name": d.get("name"), "label": audio_capture.device_label(d)}

        return {
            "inputs": [brief(d) for d in catalog["inputs"]],
            "outputs": [brief(d) for d in catalog["outputs"]],
            "selected_input": selected_in.get("index"),
            "selected_output": selected_out.get("index"),
            "_raw": catalog,
            "_selected": (selected_in, selected_out),
        }

    # -------------------------------------------------------------- lifecycle

    def state(self) -> dict[str, Any]:
        return {
            "type": "state",
            "recording": self.recording,
            "muted": self._mute.is_set(),
            "language": self.language,
            "transcript_path": self.transcript_path,
            "meeting_title": self.meeting_title,
        }

    def start(self, language: Optional[str] = None) -> dict[str, Any]:
        with self._lock:
            if self.recording:
                return {"ok": True}
            settings = self.store.get()
            api_key = self.store.api_key()
            if not api_key:
                return {"ok": False, "error": "Set your OpenAI API key in Settings first."}

            candidates = parse_language_candidates(settings["languages"])
            self.language = language if language in candidates else candidates[0]

            try:
                pyaudio = _load_pyaudio()
            except ImportError as e:
                return {"ok": False, "error": f"Audio library missing: {e}"}

            devices = self.list_devices()
            device_in, device_out = devices["_selected"]
            if not device_in:
                return {"ok": False, "error": "No microphone found."}
            if not device_out:
                return {"ok": False, "error": "No output capture device found (on Mac install BlackHole)."}

            try:
                from .transcriber import OpenAITranscribe

                self._transcriber = OpenAITranscribe(
                    api_key=api_key,
                    model=settings["transcript_model"],
                    language=self.language,
                    keywords=settings["keywords"] or None,
                    timeout_seconds=settings["transcribe_timeout_seconds"],
                    max_retries=settings["transcribe_max_retries"],
                    retry_base_seconds=settings["transcribe_retry_base_seconds"],
                    status_callback=lambda m, lvl: self.emit({"type": "status", "message": m, "level": lvl}),
                )
            except Exception as e:
                return {"ok": False, "error": f"Could not start OpenAI client: {e}"}

            self._filter = TranscriptFilter.from_settings(settings)
            output_dir = Path(settings["output_dir"])
            temp_dir = Path(settings["temp_dir"])
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                temp_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                return {"ok": False, "error": f"Cannot create folders: {e}"}
            for old in temp_dir.glob("*.wav"):
                try:
                    old.unlink()
                except OSError:
                    pass

            now = datetime.now()
            self.transcript_path = str(output_dir / f"transcription-{now.strftime('%Y%m%d_%H%M%S')}.md")
            try:
                with open(self.transcript_path, "w", encoding="utf-8") as f:
                    f.write("# Transcription Log\n\n")
                    f.write(f"**Created:** {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            except OSError as e:
                return {"ok": False, "error": f"Cannot write transcript file: {e}"}

            self.stats = MeetingStats()
            self.aliases = {}
            self._context.clear()
            self.meeting_title = None
            self.active_speaker = None
            self._stop_capture.clear()
            self._stop_store.clear()
            with self._flush_lock:
                self._flush_mic = None
                self._flush_out = None

            self._pa = pyaudio.PyAudio()
            sample_size = self._pa.get_sample_size(pyaudio.paInt16)
            q_in: Queue = Queue()
            q_out: Queue = Queue()

            self._detector = None
            if sys.platform == "win32":
                from .speaker_detection import TeamsSpeakerDetector

                self._detector = TeamsSpeakerDetector(
                    stop_event=self._stop_capture, logger=logging.getLogger("teams"),
                    window_name=settings["teams_window_name"],
                )

            def store(queue: Queue, suffix: str, device: dict) -> None:
                audio_capture.store_audio_stream(
                    queue=queue, filename_suffix=suffix, device_info=device, temp_dir=str(temp_dir),
                    stop_event=self._stop_store, sample_size_getter=lambda: sample_size,
                    transcribe_callback=self._transcribe_chunk, logger=logger,
                )

            def collect(queue: Queue, device: dict, from_mic: bool) -> None:
                source = "mic" if from_mic else "out"
                audio_capture.collect_from_stream(
                    queue=queue, input_device=device, p_instance=self._pa, from_microphone=from_mic,
                    stop_event=self._stop_capture, mute_mic_event=self._mute, flush_lock=self._flush_lock,
                    get_flush_letters=lambda: (self._flush_mic, self._flush_out),
                    clear_flush_letters=self._clear_flush,
                    speaker_snapshot_getter=self._speaker_snapshot, speaker_setter=self._set_speaker,
                    frame_duration_ms=settings["frame_duration_ms"],
                    silence_threshold=settings["silence_threshold"],
                    silence_duration=settings["silence_duration"],
                    record_seconds=settings["record_seconds"], logger=logger,
                    level_callback=lambda rms: self._on_level(source, rms),
                )

            self._capture_threads = [
                Thread(target=collect, args=(q_in, device_in, True), name="capture-mic", daemon=True),
                Thread(target=collect, args=(q_out, device_out, False), name="capture-out", daemon=True),
            ]
            self._threads = [
                Thread(target=store, args=(q_in, "in", device_in), name="store-mic", daemon=True),
                Thread(target=store, args=(q_out, "out", device_out), name="store-out", daemon=True),
            ]
            if self._detector is not None:
                self._threads.append(Thread(target=self._run_detector, name="teams", daemon=True))
            for t in self._capture_threads + self._threads:
                t.start()

            self.recording = True
            self.store.update({
                "input_device_index": device_in.get("index"), "input_device_name": device_in.get("name"),
                "output_device_index": device_out.get("index"), "output_device_name": device_out.get("name"),
            })
            logger.info("Recording started: mic=%s out=%s lang=%s file=%s", device_in.get("name"),
                        device_out.get("name"), self.language, self.transcript_path)
        self.emit(self.state())
        self.emit({"type": "status", "level": "info", "message": f"Transcription started ({self.language})"})
        return {"ok": True}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self.recording:
                return {"ok": True}
            self._stop_capture.set()
            for t in self._capture_threads:
                t.join(timeout=3.0)
            # Collectors flushed their last frames; let the store threads drain them.
            self._stop_store.set()
            for t in self._threads:
                t.join(timeout=90.0)
            self._threads = []
            self._capture_threads = []
            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None
            self.recording = False
            snapshot = self.stats.snapshot()
            stats_path = None
            if self.transcript_path:
                stats_path = self.transcript_path[:-3] + "-stats.json"
                try:
                    with open(stats_path, "w", encoding="utf-8") as f:
                        json.dump({"meeting_title": self.meeting_title, **snapshot}, f, indent=2, ensure_ascii=False)
                except OSError as e:
                    logger.warning("Could not write stats file: %s", e)
            book = ProfileBook(self.store.get()["output_dir"])
            book.merge_meeting(snapshot, self.meeting_title)
        self.emit(self.state())
        self.emit({"type": "profiles", "profiles": self.profiles()})
        self.emit({"type": "status", "level": "info", "message": f"Saved {self.transcript_path}"})
        return {"ok": True, "stats_path": stats_path}

    def shutdown(self) -> None:
        self.stop()
        self.checks.stop()

    def set_mute(self, muted: bool) -> None:
        if muted:
            self._mute.set()
        else:
            self._mute.clear()
        self.emit(self.state())

    def split(self) -> None:
        """Manual split: flush both buffers now (the old 'S' key)."""
        with self._flush_lock:
            self._flush_mic = "_"
            self._flush_out = "_"

    # ------------------------------------------------------------- speakers

    def _clear_flush(self, is_mic: bool) -> None:
        if is_mic:
            self._flush_mic = None
        else:
            self._flush_out = None

    def _speaker_snapshot(self):
        if self._detector is not None:
            return self._detector.get_speaker_snapshot()
        return None, None

    def _set_speaker(self, speaker):
        if self._detector is not None:
            return self._detector.set_current_speaker(speaker)
        return None, speaker

    def _run_detector(self) -> None:
        detector = self._detector

        def changed(previous, current):
            if previous:
                with self._flush_lock:
                    self._flush_mic = "_"
                    self._flush_out = "_"
            name = self.resolve_name(current) if current else None
            self.active_speaker = name
            self.emit({"type": "active_speaker", "name": name})
            title = detector.get_meeting_title_snapshot()
            if title and title != self.meeting_title:
                self.meeting_title = title
                self.emit(self.state())

        detector.run_detection_loop(on_speaker_changed=changed)

    def resolve_name(self, raw: Optional[str]) -> str:
        seen = set()
        name = raw or ""
        while name in self.aliases and name not in seen:
            seen.add(name)
            name = self.aliases[name]
        return name

    def rename_speaker(self, old: str, new: str) -> dict[str, Any]:
        new = (new or "").strip()
        if not old or not new or old == new:
            return {"ok": False, "error": "Pick a new name"}
        settings = self.store.get()
        if old == settings["your_name"]:
            self.store.update({"your_name": new})
        self.aliases[old] = new
        self.stats.rename(old, new)
        if not self.recording:
            book = ProfileBook(settings["output_dir"])
            book.rename(old, new)
        self.emit({"type": "speaker_renamed", "old": old, "new": new})
        self.emit({"type": "stats", "stats": self.stats.snapshot()})
        self.emit({"type": "profiles", "profiles": self.profiles()})
        return {"ok": True}

    def _on_mood(self, speaker: str, mood: str, valence: float, at: float) -> None:
        self.stats.add_mood(speaker, mood, valence, at)
        snap = self.stats.speaker_snapshot(speaker)
        if snap:
            self.emit({"type": "speaker_stats", "speaker": speaker, "stats": snap})

    def _on_level(self, source: str, rms: float) -> None:
        now = time.time()
        self._levels_peak[source] = max(self._levels_peak[source], rms)
        if now - self._last_level_emit[source] < LEVEL_EVENT_INTERVAL:
            return
        self._last_level_emit[source] = now
        peak = self._levels_peak[source]
        self._levels_peak[source] = 0.0
        settings = self.store.get()
        if source == "mic":
            speaker = settings["your_name"]
            if self._mute.is_set():
                peak = 0.0
        else:
            _, current = self._speaker_snapshot()
            speaker = self.resolve_name(current) if current else self.resolve_name(settings["remote_speaker_name"])
        self.emit({
            "type": "level", "source": source, "speaker": speaker, "rms": round(peak, 1),
            "threshold": settings["silence_threshold"],
        })

    # ---------------------------------------------------------- transcription

    def _transcribe_chunk(self, path: str, from_mic: bool, letter: Optional[str]) -> None:
        if self._transcriber is None:
            return
        duration = audio_capture.wav_duration_seconds(path)
        try:
            start_time = float(os.path.basename(path).split("-")[0])
        except ValueError:
            start_time = time.time() - duration
        segments = self._transcriber.transcribe(path)
        settings = self.store.get()
        for segment in segments:
            text = _normalize_text(segment.text)
            if not text:
                continue
            should_filter, reason = self._filter.should_filter(text) if self._filter else (False, "")
            if should_filter:
                logger.debug("Filtered segment (%s): %s", reason, text)
                continue
            if from_mic:
                speaker = settings["your_name"]
            elif letter and letter != "_":
                speaker = letter if len(letter) > 1 else f"Person_{letter.upper()}"
            else:
                speaker = settings["remote_speaker_name"]
            self.add_utterance(self.resolve_name(speaker), text, start_time + segment.start, duration, from_mic)

    def add_utterance(self, speaker: str, text: str, at: float, duration: float, is_me: bool) -> None:
        with self._file_lock:
            self._utterance_counter += 1
            utterance_id = f"u{self._utterance_counter}"
            if self.transcript_path:
                try:
                    with open(self.transcript_path, "a", encoding="utf-8") as f:
                        f.write(f"{speaker}: {text}\n\n")
                except OSError as e:
                    logger.warning("Failed to append transcript: %s", e)
            context = list(self._context)[-int(self.store.get()["llm_context_lines"]):] if self._context else []
            self._context.append(f"{speaker}: {text}")

        self.stats.add_utterance(speaker, text, at, duration, is_me)
        self.emit({
            "type": "transcript", "id": utterance_id, "speaker": speaker, "text": text, "at": at,
            "duration": round(duration, 2), "is_me": is_me,
        })
        snap = self.stats.snapshot()
        self.emit({"type": "stats", "stats": snap})
        self.checks.submit(Utterance(id=utterance_id, speaker=speaker, text=text, at=at, is_me=is_me, context=context))

    # --------------------------------------------------------------- profiles

    def profiles(self) -> dict[str, Any]:
        book = ProfileBook(self.store.get()["output_dir"])
        return {name: summarize_profile(p) for name, p in book.all().items()}

    def delete_profile(self, name: str) -> None:
        ProfileBook(self.store.get()["output_dir"]).delete(name)
        self.emit({"type": "profiles", "profiles": self.profiles()})
