import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from engine import llm
from engine.answers import AnswerRunner, answer_messages, is_question
from engine.checks import Utterance, _context_block
from engine.server import Engine, EventHub
from engine.settings import SettingsStore, default_settings, normalize, role_settings


class FakeChat(BaseHTTPRequestHandler):
    """OpenAI-style server that remembers the auth header and answers every chat."""
    seen_auth: list = []

    def log_message(self, *args):
        pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        FakeChat.seen_auth.append(self.headers.get("Authorization"))
        self._send({"data": [{"id": "whisper-1"}, {"id": "gpt-4o-mini-transcribe"}]})

    def do_POST(self):
        FakeChat.seen_auth.append(self.headers.get("Authorization"))
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert "response_format" not in body
        self._send({"choices": [{"message": {"content": " Friday, after the tests pass. "}}]})


class AnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeChat)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def settings(self, **extra):
        values = default_settings()
        values.update({"answer_enabled": True, "answer_api": "openai", "answer_base_url": self.url,
                       "answer_model": "m", "answer_api_key_env": "MT_TEST_ANSWER_KEY",
                       "answer_context": "Release is planned for Friday."})
        values.update(extra)
        return normalize(values)

    def test_question_from_others_gets_an_answer_with_key_and_context(self):
        settings = self.settings()
        events = []
        runner = AnswerRunner(lambda: settings, events.append)
        u = Utterance("u1", "Anna", "When do we ship?", 1.0, False, ["Anna: hi"])
        FakeChat.seen_auth.clear()
        with mock.patch.dict(os.environ, {"MT_TEST_ANSWER_KEY": "sk-local"}):
            self.assertEqual(runner.answer(u), "Friday, after the tests pass.")
        self.assertEqual(FakeChat.seen_auth, ["Bearer sk-local"])
        self.assertEqual(events[0]["kind"], "answer")
        self.assertEqual(events[0]["result"]["note"], "Friday, after the tests pass.")

        system, _user = answer_messages(u, settings["answer_prompt"], settings["answer_context"])
        self.assertIn("Release is planned for Friday.", system)

    def test_only_questions_from_the_right_people_are_queued(self):
        settings = self.settings()
        runner = AnswerRunner(lambda: settings, lambda e: None)
        runner.submit(Utterance("u1", "Anna", "We ship on Friday.", 1.0, False))
        runner.submit(Utterance("u2", "Me", "When do we ship?", 1.0, True))
        self.assertEqual(len(runner._queue), 0)
        runner.submit(Utterance("u3", "Anna", "When do we ship?", 1.0, False))
        self.assertEqual(len(runner._queue), 1)
        self.assertTrue(is_question("what is the plan for today"))
        self.assertFalse(is_question("ok"))

    def test_role_settings_maps_answer_keys(self):
        view = role_settings(self.settings(), "answer")
        self.assertEqual(view["llm_api"], "openai")
        self.assertEqual(view["llm_base_url"], self.url)
        self.assertEqual(view["llm_api_key_env"], "MT_TEST_ANSWER_KEY")
        self.assertTrue(view["llm_enabled"])
        client = llm.LLMClient.from_settings(view)
        self.assertEqual(client.model, "m")

    def test_checks_context_goes_first(self):
        u = Utterance("u1", "Anna", "Hi", 1.0, False)
        self.assertTrue(_context_block(u, "Project Atlas").startswith("Background:\nProject Atlas"))
        self.assertTrue(_context_block(u).startswith("Earlier conversation"))


class EngineRoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.store = SettingsStore(d / "s.json", env_file=d / "none.env")
        self.engine = Engine(self.store, EventHub())

    def tearDown(self):
        self.engine.session.checks.stop()
        self.engine.session.answers.stop()
        self.tmp.cleanup()

    def test_answer_setup_buttons_save_to_answer_keys(self):
        info = self.engine.handle("llm_status", {"role": "answer", "llm_api": "openai",
                                                 "llm_base_url": "http://127.0.0.1:9/v1",
                                                 "llm_model": "qwen2.5:7b", "llm_api_key_env": "X_KEY"})
        self.assertFalse(info["running"])
        values = self.store.get()
        self.assertEqual(values["answer_api"], "openai")
        self.assertEqual(values["answer_model"], "qwen2.5:7b")
        self.assertEqual(values["answer_api_key_env"], "X_KEY")
        # The checks AI is untouched.
        self.assertEqual(values["llm_api"], "laya")
        self.assertEqual(values["llm_model"], "llama3.2:3b")

    def test_transcription_key_comes_from_its_env_variable(self):
        self.store.update({"transcribe_api_key_env": "MT_TEST_STT_KEY"})
        with mock.patch.dict(os.environ, {"MT_TEST_STT_KEY": "sk-stt"}):
            self.assertEqual(self.store.api_key(), "sk-stt")
            self.assertTrue(self.store.public()["openai_api_key_set"])
        self.store.update({"transcribe_api_key_env": ""})
        self.assertEqual(self.store.api_key(), "")

    def test_transcribe_check(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeChat)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/v1"
            result = self.engine.handle("transcribe_check", {"transcribe_base_url": url,
                                                             "transcript_model": "whisper-1"})
            self.assertTrue(result["ok"], result)
            self.assertIn("whisper-1 found", result["message"])
            missing = self.engine.handle("transcribe_check", {"transcribe_base_url": "",
                                                              "transcribe_api_key_env": "MT_NOT_SET_ANYWHERE"})
            self.assertFalse(missing["ok"])
            self.assertIn("MT_NOT_SET_ANYWHERE", missing["error"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
