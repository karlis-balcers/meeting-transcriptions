import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine import laya, speedtest
from engine.checks import CheckRunner, Utterance, laya_state, sample_utterance
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
        try:
            self.wfile.write(body)
        except BrokenPipeError:  # the client gave up waiting (timeout test)
            pass

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok", "loaded": [], "device": "cpu"})
        else:
            self._send({"detail": "nope"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLaya.requests.append(body)
        if "SLOW" in str(body.get("state")):
            time.sleep(1.0)
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
        import tempfile
        from pathlib import Path
        from unittest import mock

        with tempfile.TemporaryDirectory() as d, mock.patch.object(laya, "home", lambda: Path(d)):
            info = laya.status(self.url)
            self.assertTrue(info["running"])
            self.assertFalse(info["model_ready"])  # running, but the checkpoints were never downloaded

            # laya-serve prints Hugging Face progress bars into its log while it downloads.
            (Path(d) / "laya-serve.log").write_text("old line\n")
            progress, lines = [], []

            def on_progress(message, fraction):
                progress.append((message, fraction))

            on_progress.log = lines.append
            original = laya.predict

            def slow_predict(*args, **kwargs):
                with open(Path(d) / "laya-serve.log", "a") as f:
                    f.write("model.safetensors:  40%|####      | 200M/500M\rmodel.safetensors: 100%|##########| 500M/500M\n")
                import time
                time.sleep(0.8)
                return original(*args, **kwargs)

            with mock.patch.object(laya, "predict", slow_predict):
                laya.warm_up(self.url, on_progress)
            self.assertTrue(laya.status(self.url)["model_ready"])
            self.assertIn(0.4, [f for _, f in progress])
            self.assertTrue(any("100%|" in line for line in lines))
            self.assertFalse(any("old line" in line for line in lines))
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

    def test_state_puts_the_latest_line_first_and_fits_the_window(self):
        u = Utterance("u", "Anna", "the newest line", 0, False, [f"line {i}: " + "x" * 300 for i in range(6)])
        state = laya_state(u, "We are the payments team.", max_chars=1000)
        self.assertTrue(state.startswith("LATEST utterance by Anna:\nthe newest line"))
        self.assertLessEqual(len(state), 1000)
        self.assertIn("line 5", state)  # the newest earlier lines stay...
        self.assertNotIn("line 0", state)  # ...the oldest go
        self.assertLess(state.index("line 4"), state.index("line 5"))  # still in the order they were said
        short = laya_state(Utterance("u", "A", "hi", 0, False, ["B: hello"]), "We are the payments team.")
        self.assertTrue(short.endswith("Background:\nWe are the payments team."))

    def test_slow_answer_is_a_timeout(self):
        with self.assertRaises(laya.LayaTimeout):
            laya.predict(self.url, "SLOW", {"x": {"type": "noul", "instructions": "a task?"}}, timeout=0.3)

    def test_timeout_skips_the_backlog(self):
        settings = {"llm_enabled": True, "llm_api": "laya", "mood_enabled": True, "custom_checks": []}

        def timing_out(*args, **kwargs):
            raise laya.LayaTimeout("Laya did not answer within 30s")

        events = []
        runner = CheckRunner(lambda: settings, events.append, laya_predict=timing_out)
        for i in range(5):
            runner.submit(Utterance(f"u{i}", "A", "x", 0, False))
        runner.run_checks(Utterance("now", "A", "x", 0, False))
        self.assertEqual([u.id for u in runner._queue], ["u4"])
        self.assertIn("skipped 4 lines", events[0]["message"])

    def test_speed_test_sends_short_and_full_context(self):
        from unittest import mock

        settings = {"llm_base_url": self.url, "llm_timeout_seconds": 30, "llm_context": "Payments team",
                    "mood_enabled": True, "custom_checks": [
                        {"id": "act", "name": "Action item", "prompt": "Is there a task?", "enabled": True}]}
        FakeLaya.requests.clear()
        with mock.patch.object(laya, "nvidia_gpu", lambda: True):
            message = speedtest.laya_speed(settings, lambda *a: None)
        self.assertEqual(len(FakeLaya.requests), 3)
        self.assertEqual(set(FakeLaya.requests[1]["questions"]), {"mood", "intensity", "custom:act"})
        self.assertIn(sample_utterance("en").context[-1], FakeLaya.requests[1]["state"])
        self.assertIn("Pēteris", FakeLaya.requests[2]["state"])
        self.assertIn("Laya on cpu: short line", message)
        self.assertIn("press Install to move it to the NVIDIA GPU", message)

    def test_unreachable_server_is_one_status(self):
        settings = {"llm_enabled": True, "llm_api": "laya", "llm_base_url": "http://127.0.0.1:9",
                    "llm_timeout_seconds": 1, "mood_enabled": True, "custom_checks": []}
        events = []
        runner = CheckRunner(lambda: settings, events.append)
        runner.run_checks(Utterance("u1", "A", "x", 0, False))
        runner.run_checks(Utterance("u2", "A", "y", 0, False))
        self.assertEqual([e["type"] for e in events], ["status"])


class ProcTests(unittest.TestCase):
    def test_run_logged_streams_lines_and_progress(self):
        import sys

        from engine.laya import _PipProgress
        from engine.proc import run_logged

        progress, lines = [], []

        def on_progress(message, fraction):
            progress.append((message, fraction))

        on_progress.log = lines.append
        script = ("import sys; print('Resolved 40 packages'); print('Downloading torch (110.0MiB)'); "
                  "print('Downloading numpy (12MiB)'); sys.stdout.write('Downloaded numpy\\r'); "
                  "print('Downloaded torch'); sys.exit(3)")
        code, tail = run_logged([sys.executable, "-c", script], on_progress, "Installing",
                                on_line=_PipProgress(on_progress, "Installing"))
        self.assertEqual(code, 3)
        self.assertIn("Downloaded numpy", lines)
        self.assertEqual(tail[-1], "Downloaded torch")
        self.assertEqual(progress[-1][1], 0.99)
        self.assertIn("1/2 big downloads: torch", progress[-2][0])


class LayaSettingsTests(unittest.TestCase):
    def test_old_ollama_settings_move_to_laya_once(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.json"
            path.write_text(json.dumps({"llm_api": "ollama", "llm_base_url": "http://127.0.0.1:11434"}))
            store = SettingsStore(path)
            self.assertEqual(store.get()["llm_api"], "laya")
            self.assertEqual(store.get()["llm_base_url"], "http://127.0.0.1:8765")
            store.update({"llm_api": "ollama"})  # picked again on purpose: stays
            self.assertEqual(SettingsStore(path).get()["llm_api"], "ollama")

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



class SetupButtonsTests(unittest.TestCase):
    def test_setup_commands_use_the_form_values(self):
        import tempfile
        from pathlib import Path

        from engine.server import Engine, EventHub

        with tempfile.TemporaryDirectory() as d:
            store = SettingsStore(Path(d) / "s.json", env_file=Path(d) / "none.env")
            engine = Engine(store, EventHub())
            info = engine.handle("llm_status", {"llm_api": "ollama", "llm_base_url": "http://127.0.0.1:8765",
                                                "llm_model": "qwen2.5:3b"})
            self.assertEqual(info["api"], "ollama")
            self.assertEqual(store.get()["llm_base_url"], "http://127.0.0.1:11434")
            self.assertEqual(store.get()["llm_model"], "qwen2.5:3b")


if __name__ == "__main__":
    unittest.main()
