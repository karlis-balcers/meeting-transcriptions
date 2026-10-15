import json
import sys
import tempfile
import threading
import time
import types
import unittest
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from engine import parakeet, parakeet_server
from engine.server import Engine, EventHub
from engine.settings import (PARAKEET_MODEL, PARAKEET_URL, SettingsStore, normalize, role_settings)


class FakeModel:
    """Stands in for the loaded Parakeet model in the server."""

    def __init__(self, text="labdien, šis ir tests", device="cpu"):
        self.name = PARAKEET_MODEL
        self.text = text
        self.device = device
        self.paths = []

    def transcribe(self, path, wait=0):
        with open(path, "rb") as f:
            self.paths.append(f.read()[:4])
        return self.text

    def health(self):
        return {"status": "ok", "model": self.name, "requested_device": "auto", "device": self.device,
                "loaded": True, "loading": False, "error": None, "tried": []}


def write_wav(path: Path, seconds: float = 0.5) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x01\x00" * int(16000 * seconds))


class ParakeetSettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_new_install_uses_local_parakeet_without_a_key(self):
        store = SettingsStore(self.dir / "settings.json", env_file=self.dir / "none.env")
        values = store.get()
        self.assertEqual(values["transcribe_api"], "parakeet")
        self.assertEqual(values["transcribe_base_url"], PARAKEET_URL)
        self.assertEqual(values["transcribe_model"], PARAKEET_MODEL)
        self.assertEqual(values["transcribe_device"], "auto")
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-env"}):
            self.assertEqual(store.api_key(), "")  # never sent to the local server

    def test_openai_users_move_to_parakeet(self):
        path = self.dir / "settings.json"
        path.write_text(json.dumps({"settings_version": 2, "transcribe_base_url": "",
                                    "transcript_model": "gpt-4o-transcribe", "openai_api_key": "sk-x"}))
        values = SettingsStore(path).get()
        self.assertEqual(values["transcribe_api"], "parakeet")
        self.assertEqual(values["transcribe_base_url"], PARAKEET_URL)
        self.assertEqual(values["transcribe_model"], PARAKEET_MODEL)
        self.assertEqual(json.loads(path.read_text())["settings_version"], 3)

    def test_own_server_users_keep_it(self):
        path = self.dir / "settings.json"
        path.write_text(json.dumps({"settings_version": 2, "transcribe_base_url": "http://127.0.0.1:8000/v1",
                                    "transcript_model": "Systran/faster-whisper-large-v3"}))
        values = SettingsStore(path).get()
        self.assertEqual(values["transcribe_api"], "openai")
        self.assertEqual(values["transcribe_base_url"], "http://127.0.0.1:8000/v1")
        self.assertEqual(values["transcribe_model"], "Systran/faster-whisper-large-v3")

    def test_switching_type_swaps_default_url_and_model(self):
        to_openai = normalize({"transcribe_api": "openai", "transcribe_base_url": PARAKEET_URL,
                               "transcribe_model": PARAKEET_MODEL})
        self.assertEqual(to_openai["transcribe_base_url"], "")
        self.assertEqual(to_openai["transcribe_model"], "gpt-4o-mini-transcribe")
        back = normalize({"transcribe_api": "parakeet", "transcribe_base_url": "",
                          "transcribe_model": "gpt-4o-mini-transcribe", "transcribe_device": "tpu"})
        self.assertEqual(back["transcribe_base_url"], PARAKEET_URL)
        self.assertEqual(back["transcribe_model"], PARAKEET_MODEL)
        self.assertEqual(back["transcribe_device"], "auto")

    def test_role_settings_for_transcription(self):
        view = role_settings(normalize({}), "transcribe")
        self.assertEqual(view["llm_api"], "parakeet")
        self.assertEqual(view["llm_base_url"], PARAKEET_URL)
        self.assertEqual(view["llm_model"], PARAKEET_MODEL)


class ParakeetServerTests(unittest.TestCase):
    """The server script with a fake model: it must look like the OpenAI transcription API."""

    @classmethod
    def setUpClass(cls):
        cls.model = FakeModel()

        class Handler(parakeet_server.Handler):
            model = cls.model

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_openai_client_transcribes_through_it(self):
        from engine.transcriber import OpenAITranscribe

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "1700000000.0-in.wav"
            write_wav(wav)
            client = OpenAITranscribe(api_key="", base_url=self.url, model=PARAKEET_MODEL, language="lv",
                                      keywords="Acme", max_retries=0)
            segments = client.transcribe(str(wav))
        self.assertEqual([s.text for s in segments], ["labdien, šis ir tests"])
        self.assertEqual(self.model.paths[-1], b"RIFF")

    def test_health_and_models(self):
        info = parakeet.health(self.url)
        self.assertTrue(info["loaded"])
        status = parakeet.status(self.url, PARAKEET_MODEL, "auto")
        self.assertTrue(status["running"])
        self.assertTrue(status["model_ready"])
        self.assertEqual(status["device"], "cpu")
        from engine import llm

        models = llm._request(self.url + "/models")
        self.assertEqual(models["data"][0]["id"], PARAKEET_MODEL)

    def test_status_when_not_running(self):
        with mock.patch.object(parakeet, "installed", return_value=False):
            status = parakeet.status("http://127.0.0.1:9/v1", PARAKEET_MODEL, "auto")
        self.assertFalse(status["running"])
        self.assertFalse(status["installed"])
        self.assertIn("error", status)


class DeviceTests(unittest.TestCase):
    def test_candidates(self):
        self.assertEqual(parakeet_server.device_candidates("auto", True, False), ["cuda", "cpu"])
        self.assertEqual(parakeet_server.device_candidates("auto", False, True), ["mps", "cpu"])
        self.assertEqual(parakeet_server.device_candidates("auto", False, False), ["cpu"])
        self.assertEqual(parakeet_server.device_candidates("cuda", False, False), ["cuda", "cpu"])
        self.assertEqual(parakeet_server.device_candidates("cpu", True, True), ["cpu"])

    def test_gpu_that_fails_falls_back_to_cpu(self):
        loaded = []

        class Speech:
            def __init__(self, device):
                self.device = device

            def transcribe(self, audio, timestamps):
                if self.device == "cuda":
                    raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
                return {"text": "hello" if loaded else ""}

        def photon(name, device):
            return Speech(device)

        fake_torch = types.SimpleNamespace(
            __version__="2.8.0", cuda=types.SimpleNamespace(is_available=lambda: True),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)))
        fake_md = types.SimpleNamespace(photon=photon)
        with mock.patch.dict(sys.modules, {"torch": fake_torch, "moondream": fake_md}):
            model = parakeet_server.Model(PARAKEET_MODEL, "auto")
            model.load()
        self.assertEqual(model.device, "cpu")
        self.assertEqual(model.tried[0]["device"], "cuda")
        self.assertIn("no kernel image", model.tried[0]["error"])
        loaded.append(True)
        self.assertEqual(model.transcribe("x.wav"), "hello")
        self.assertFalse(model.health()["loading"])

    def test_nothing_works_reports_error(self):
        def photon(name, device):
            raise RuntimeError(f"broken on {device}")

        fake_torch = types.SimpleNamespace(
            __version__="2.8.0", cuda=types.SimpleNamespace(is_available=lambda: False), backends=types.SimpleNamespace())
        with mock.patch.dict(sys.modules, {"torch": fake_torch, "moondream": types.SimpleNamespace(photon=photon)}):
            model = parakeet_server.Model(PARAKEET_MODEL, "auto")
            model.load()
        self.assertIsNone(model.device)
        self.assertIn("broken on cpu", model.error)
        with self.assertRaises(RuntimeError):
            model.transcribe("x.wav")


class StartSession:
    def __init__(self, store, emit):
        from engine.profiles import MeetingStats

        self.store = store
        self.stats = MeetingStats()
        self.started = threading.Event()

    def state(self):
        return {"type": "state", "recording": False}

    def start(self, language=None):
        self.started.set()
        return {"ok": True}


class StartRecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        store = SettingsStore(Path(self.tmp.name) / "settings.json", env_file=Path(self.tmp.name) / "x.env")
        self.events = []
        hub = EventHub()
        hub.send = lambda msg, only=None: self.events.append(msg)
        self.engine = Engine(store, hub, session_factory=StartSession)

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_start_installs_parakeet_then_records(self):
        calls = []
        with mock.patch.object(parakeet, "health", side_effect=parakeet.ParakeetError("down")), \
                mock.patch.object(parakeet, "installed", return_value=False), \
                mock.patch.object(parakeet, "install", side_effect=lambda p: calls.append("install")), \
                mock.patch.object(parakeet, "ensure_running",
                                  side_effect=lambda *a: calls.append("run") or {"device": "cuda"}), \
                mock.patch.object(parakeet, "start_server", return_value=False):
            result = self.engine.handle("start", {"language": "lv"})
            self.assertTrue(result["ok"])
            self.assertTrue(self.engine.session.started.wait(5))
        self.assertEqual(calls, ["install", "run"])
        done = [e for e in self.events if e.get("type") == "llm" and e.get("state") == "done"]
        self.assertEqual(done[0]["role"], "transcribe")
        self.assertIn("NVIDIA GPU", done[0]["message"])

    def test_start_goes_straight_on_when_loaded(self):
        health = {"loaded": True, "model": PARAKEET_MODEL, "requested_device": "auto"}
        with mock.patch.object(parakeet, "health", return_value=health), \
                mock.patch.object(parakeet, "install") as install:
            self.engine.handle("start", {})
            self.assertTrue(self.engine.session.started.wait(5))
        install.assert_not_called()

    def test_openai_needs_no_setup(self):
        self.engine.store.update({"transcribe_api": "openai"})
        with mock.patch.object(parakeet, "install") as install:
            self.engine.handle("start", {})
            self.assertTrue(self.engine.session.started.wait(5))
        install.assert_not_called()

    def test_failed_setup_does_not_start(self):
        with mock.patch.object(parakeet, "health", side_effect=parakeet.ParakeetError("down")), \
                mock.patch.object(parakeet, "installed", return_value=True), \
                mock.patch.object(parakeet, "ensure_running", side_effect=parakeet.ParakeetError("no luck")), \
                mock.patch.object(parakeet, "start_server", return_value=False):
            self.engine.handle("start", {})
            deadline = time.time() + 5
            while self.engine._llm_current is not None or not any(e.get("state") == "error" for e in self.events):
                if time.time() > deadline:
                    break
                time.sleep(0.05)
        self.assertFalse(self.engine.session.started.is_set())
        errors = [e for e in self.events if e.get("state") == "error"]
        self.assertEqual(errors[0]["message"], "no luck")


class ServerScriptTests(unittest.TestCase):
    def test_server_script_is_next_to_the_module(self):
        self.assertTrue(parakeet.server_script().exists())

    def test_port_from_url(self):
        self.assertEqual(parakeet._port("http://127.0.0.1:8766/v1"), "8766")
        self.assertEqual(parakeet._port("http://127.0.0.1:9000"), "9000")
        self.assertEqual(parakeet._port("http://localhost/v1"), "8766")


if __name__ == "__main__":
    unittest.main()
