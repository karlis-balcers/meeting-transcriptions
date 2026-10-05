import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine import llm


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload, lines=False):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if lines:
            for item in payload:
                self.wfile.write((json.dumps(item) + "\n").encode())
        else:
            self.wfile.write(json.dumps(payload).encode())

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": "llama3.2:3b"}]})
        elif self.path == "/v1/models":
            self._send({"data": [{"id": "local-model"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/chat":
            assert body["format"] == "json"
            self._send({"message": {"content": '{"mood": "calm", "intensity": 0.4}'}})
        elif self.path == "/v1/chat/completions":
            self._send({"choices": [{"message": {"content": 'Here: {"hit": true}'}}]})
        elif self.path == "/api/pull":
            self._send([{"status": "pulling", "total": 10, "completed": 5}, {"status": "success"}], lines=True)
        else:
            self.send_error(404)


class LLMClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_ollama_status_chat_and_pull(self):
        settings = {"llm_api": "ollama", "llm_base_url": self.url, "llm_model": "llama3.2:3b", "llm_enabled": True}
        info = llm.status(settings)
        self.assertTrue(info["running"])
        self.assertTrue(info["model_ready"])
        client = llm.LLMClient.from_settings(settings)
        self.assertEqual(client.chat_json("sys", "user")["mood"], "calm")
        progress = []
        client.pull(lambda status, fraction: progress.append((status, fraction)))
        self.assertEqual(progress[0], ("pulling", 0.5))

    def test_missing_model_is_reported(self):
        info = llm.status({"llm_api": "ollama", "llm_base_url": self.url, "llm_model": "qwen2.5:3b"})
        self.assertTrue(info["running"])
        self.assertFalse(info["model_ready"])

    def test_openai_compatible_server(self):
        client = llm.LLMClient("openai", self.url, "local-model")
        self.assertEqual(client.list_models(), ["local-model"])
        self.assertEqual(client.chat_json("sys", "user"), {"hit": True})

    def test_unreachable_server(self):
        client = llm.LLMClient("ollama", "http://127.0.0.1:9", "x", timeout=1)
        with self.assertRaises(llm.LLMError):
            client.chat_json("a", "b")


if __name__ == "__main__":
    unittest.main()
