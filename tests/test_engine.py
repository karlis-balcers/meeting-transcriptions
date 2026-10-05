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
