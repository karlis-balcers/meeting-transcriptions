import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine import laya
from engine.checks import CheckRunner, Utterance
from engine.settings import SettingsStore


class FakeLaya(BaseHTTPRequestHandler):
    """Answers like laya-serve: one probability per question."""
    requests = []

    def log_message(self, *args):
        pass

    def _send(self, payload, code=200):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok", "loaded": [], "device": "cpu"})
        else:
            self._send({"detail": "nope"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLaya.requests.append(body)
        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "choice":
                answers[qid] = {"type": "choice", "choice": "frustrated", "answer_confidence": 0.81}
            elif q["type"] == "score":
                answers[qid] = {"type": "score", "score": 2.0}
            else:
                answers[qid] = {"type": "noul", "noul": 0.9 if "task" in q["instructions"] else 0.1}
        self._send({"answers": answers, "routing": {"model": "english"}})


class LayaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLaya)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_status_and_predict(self):
        info = laya.status(self.url)
        self.assertTrue(info["running"])
        self.assertTrue(info["model_ready"])
        answers = laya.predict(self.url, "hi", {"x": {"type": "noul", "instructions": "a task?"}})
        self.assertEqual(answers["x"]["noul"], 0.9)

    def test_unreachable(self):
        self.assertFalse(laya.status("http://127.0.0.1:9")["running"])
        with self.assertRaises(laya.LayaError):
            laya.predict("http://127.0.0.1:9", "hi", {}, timeout=1)

    def test_checks_run_as_one_call(self):
        settings = {
            "llm_enabled": True, "llm_api": "laya", "llm_base_url": self.url,
            "mood_enabled": True, "fact_check_enabled": True, "fact_check_applies_to": "everyone",
            "custom_checks": [
                {"id": "act", "name": "Action item", "prompt": "Is there a task?", "applies_to": "everyone",
                 "enabled": True, "color": "#fff"},
                {"id": "q", "name": "Question", "prompt": "Is it a question?", "applies_to": "everyone",
                 "enabled": True},
                {"id": "off", "name": "Off", "prompt": "a task", "applies_to": "everyone", "enabled": False},
            ],
        }
        FakeLaya.requests.clear()
        events, moods, hits = [], [], []
        runner = CheckRunner(lambda: settings, events.append, on_mood=lambda *a: moods.append(a),
                             on_hit=lambda *a: hits.append(a))
        runner.run_checks(Utterance("u1", "Anna", "Send the report by Friday", 5.0, False, ["Me: hi"]))
        self.assertEqual(len(FakeLaya.requests), 1)
        sent = FakeLaya.requests[0]
        self.assertEqual(set(sent["questions"]), {"mood", "intensity", "fact", "custom:act", "custom:q"})
        self.assertIn("Me: hi", sent["state"])
        self.assertEqual([e["kind"] for e in events], ["mood", "custom"])  # fact 10% and Question 10%: no hit
        self.assertEqual(events[0]["result"]["mood"], "frustrated")
        self.assertEqual(events[0]["result"]["intensity"], 1.0)
        self.assertEqual(moods[0][:2], ("Anna", "frustrated"))
        self.assertEqual(hits, [("Anna", "Action item")])

    def test_unreachable_server_is_one_status(self):
        settings = {"llm_enabled": True, "llm_api": "laya", "llm_base_url": "http://127.0.0.1:9",
                    "llm_timeout_seconds": 1, "mood_enabled": True, "custom_checks": []}
        events = []
        runner = CheckRunner(lambda: settings, events.append)
        runner.run_checks(Utterance("u1", "A", "x", 0, False))
        runner.run_checks(Utterance("u2", "A", "y", 0, False))
        self.assertEqual([e["type"] for e in events], ["status"])


class LayaSettingsTests(unittest.TestCase):
    def test_server_url_follows_server_type(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            store = SettingsStore(Path(d) / "s.json", env_file=Path(d) / "none.env")
            self.assertEqual(store.get()["llm_api"], "laya")
            store.update({"llm_api": "ollama"})
            self.assertEqual(store.get()["llm_base_url"], "http://127.0.0.1:11434")
            store.update({"llm_api": "laya"})
            self.assertEqual(store.get()["llm_base_url"], "http://127.0.0.1:8765")
            store.update({"llm_api": "openai", "llm_base_url": "http://127.0.0.1:1234/v1"})
            store.update({"llm_api": "laya"})
            self.assertEqual(store.get()["llm_base_url"], "http://127.0.0.1:1234/v1")


if __name__ == "__main__":
    unittest.main()
