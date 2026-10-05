import json
import os
import socket
import tempfile
import threading
import unittest
import wave
from pathlib import Path

from engine.server import Engine, EngineServer, EventHub
from engine.session import Session
from engine.settings import SettingsStore
from engine.transcriber import Segment
from engine.transcript_filter import TranscriptFilter


class FakeTranscriber:
    def __init__(self, text):
        self.text = text

    def transcribe(self, path):
        return [Segment(0, 1.0, self.text)]


def write_wav(path, seconds=2.0, rate=16000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * int(seconds * rate))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.store = SettingsStore(self.dir / "settings.json", env_file=self.dir / "none.env")
        self.store.update({"output_dir": str(self.dir / "out"), "your_name": "Karlis"})
        self.events = []
        self.session = Session(self.store, self.events.append)

    def tearDown(self):
        self.session.checks.stop()
        self.tmp.cleanup()

    def test_chunk_goes_to_file_events_and_stats(self):
        self.session.transcript_path = str(self.dir / "t.md")
        self.session._transcriber = FakeTranscriber("Can we ship it on Friday?")
        self.session._filter = TranscriptFilter()
        wav = self.dir / "1700000000.00-out.wav"
        write_wav(wav)

        self.session._transcribe_chunk(str(wav), False, None)
        self.session._transcribe_chunk(str(wav), True, None)

        transcripts = [e for e in self.events if e["type"] == "transcript"]
        self.assertEqual([t["speaker"] for t in transcripts], ["Remote", "Karlis"])
        self.assertAlmostEqual(transcripts[0]["duration"], 2.0, places=1)
        self.assertEqual(transcripts[0]["at"], 1700000000.0)
        content = Path(self.session.transcript_path).read_text(encoding="utf-8")
        self.assertIn("Remote: Can we ship it on Friday?", content)
        snap = self.session.stats.snapshot()["speakers"]
        self.assertEqual(snap["Remote"]["questions"], 1)

    def test_filtered_text_is_dropped(self):
        self.session._transcriber = FakeTranscriber("Thanks for watching!")
        self.session._filter = TranscriptFilter()
        wav = self.dir / "1.00-in.wav"
        write_wav(wav, 0.5)
        self.session._transcribe_chunk(str(wav), True, None)
        self.assertFalse([e for e in self.events if e["type"] == "transcript"])

    def test_rename_applies_to_later_utterances(self):
        self.session.add_utterance("Remote", "hello", 1.0, 2.0, False)
        result = self.session.rename_speaker("Remote", "Anna")
        self.assertTrue(result["ok"])
        self.assertEqual(self.session.resolve_name("Remote"), "Anna")
        self.assertIn("Anna", self.session.stats.snapshot()["speakers"])

    def test_renaming_me_updates_settings(self):
        self.session.rename_speaker("Karlis", "KB")
        self.assertEqual(self.store.get()["your_name"], "KB")

    def test_start_without_key_fails_cleanly(self):
        os.environ.pop("OPENAI_API_KEY", None)
        result = self.session.start()
        self.assertFalse(result["ok"])
        self.assertIn("API key", result["error"])


class FakeSession:
    def __init__(self, store, emit):
        from engine.profiles import MeetingStats

        self.store = store
        self.emit = emit
        self.stats = MeetingStats()
        self.muted = None

    def state(self):
        return {"type": "state", "recording": False}

    def set_mute(self, muted):
        self.muted = muted
        self.emit({"type": "state", "recording": False, "muted": muted})

    def shutdown(self):
        pass


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        store = SettingsStore(Path(self.tmp.name) / "settings.json", env_file=Path(self.tmp.name) / "x.env")
        self.engine = Engine(store, EventHub(), session_factory=FakeSession)
        self.server = EngineServer(self.engine, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()
        self.sock = socket.create_connection(self.server.server_address, timeout=5)
        self.reader = self.sock.makefile("r", encoding="utf-8")

    def tearDown(self):
        self.sock.close()
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def send(self, msg):
        self.sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))

    def read_until(self, predicate):
        for _ in range(20):
            msg = json.loads(self.reader.readline())
            if predicate(msg):
                return msg
        self.fail("message not received")

    def test_hello_then_commands(self):
        hello = self.read_until(lambda m: m["type"] == "hello")
        self.assertIn("settings", hello)
        self.assertNotIn("openai_api_key", hello["settings"])

        self.send({"id": 1, "cmd": "mute", "muted": True})
        resp = self.read_until(lambda m: m["type"] == "response")
        self.assertTrue(resp["ok"])
        self.assertEqual(self.engine.session.muted, True)

        self.send({"id": 2, "cmd": "nope"})
        resp = self.read_until(lambda m: m["type"] == "response" and m["id"] == 2)
        self.assertFalse(resp["ok"])

        self.send({"id": 3, "cmd": "save_settings", "settings": {"your_name": "Karlis", "openai_api_key": ""}})
        resp = self.read_until(lambda m: m["type"] == "response" and m["id"] == 3)
        self.assertEqual(resp["data"]["your_name"], "Karlis")


if __name__ == "__main__":
    unittest.main()


class FakeStream:
    def __init__(self, rate, chunk):
        self.chunk = chunk
        self.calls = 0

    def read(self, n, exception_on_overflow=False):
        import struct
        import time as _t

        _t.sleep(0.002)
        self.calls += 1
        # ~0.5s of loud audio, then silence, repeating.
        loud = (self.calls % 200) < 5
        value = 3000 if loud else 0
        return struct.pack(f"{n}h", *([value] * n))

    def is_active(self):
        return True

    def stop_stream(self):
        pass

    def close(self):
        pass


class FakePyAudio:
    paInt16 = 8

    def __init__(self):
        self.devices = [
            {"index": 0, "name": "Mic", "maxInputChannels": 1, "maxOutputChannels": 0, "defaultSampleRate": 16000},
            {"index": 1, "name": "BlackHole", "maxInputChannels": 1, "maxOutputChannels": 2, "defaultSampleRate": 16000},
        ]

    def get_device_count(self):
        return len(self.devices)

    def get_device_info_by_index(self, i):
        return self.devices[i]

    def get_default_input_device_info(self):
        return self.devices[0]

    def get_default_output_device_info(self):
        return self.devices[1]

    def get_sample_size(self, fmt):
        return 2

    def get_format_from_width(self, w):
        return 8

    def open(self, rate, frames_per_buffer, **kwargs):
        return FakeStream(rate, frames_per_buffer)

    def terminate(self):
        pass


class PipelineTests(unittest.TestCase):
    """Start/stop with fake audio and a fake transcriber, end to end through the threads."""

    def test_start_record_stop_saves_transcript_stats_and_profiles(self):
        import sys
        import time
        import types
        from unittest import mock

        fake_module = types.SimpleNamespace(PyAudio=FakePyAudio, paInt16=8)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        store = SettingsStore(d / "settings.json", env_file=d / "none.env")
        store.update({
            "output_dir": str(d / "out"), "temp_dir": str(d / "tmp"), "openai_api_key": "sk-test",
            "silence_duration": 0.2, "frame_duration_ms": 20, "your_name": "Karlis",
        })
        events = []

        class Transcriber:
            def __init__(self, **kwargs):
                pass

            def transcribe(self, path):
                return [Segment(0, 1.0, "We should ship the release on Friday")]

        with mock.patch.dict(sys.modules, {"pyaudio": fake_module, "pyaudiowpatch": fake_module}), \
                mock.patch("engine.transcriber.OpenAITranscribe", Transcriber), \
                mock.patch("engine.session.sys.platform", "darwin"):
            session = Session(store, events.append)
            self.addCleanup(session.checks.stop)
            result = session.start()
            self.assertTrue(result["ok"], result)
            deadline = time.time() + 10
            while time.time() < deadline and len([e for e in events if e["type"] == "transcript"]) < 2:
                time.sleep(0.05)
            session.stop()

        transcripts = [e for e in events if e["type"] == "transcript"]
        self.assertGreaterEqual(len(transcripts), 2)
        self.assertEqual({t["speaker"] for t in transcripts}, {"Karlis", "Remote"})
        self.assertTrue(any(e["type"] == "level" for e in events))
        md = Path(session.transcript_path)
        self.assertIn("Karlis: We should ship", md.read_text(encoding="utf-8"))
        self.assertTrue(Path(str(md)[:-3] + "-stats.json").exists())
        profiles = session.profiles()
        self.assertIn("Karlis", profiles)
        self.assertEqual(profiles["Karlis"]["meetings"], 1)
        self.assertFalse(session.recording)
