import os
import time
import unittest

from assistant import Assistant


class AssistantConcurrencySmokeTests(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        self.assistant = Assistant(vector_store_id=None, your_name="Tester")

    def tearDown(self):
        self.assistant.stop()

    def test_messages_are_buffered_by_background_thread(self):
        count = 20
        start = time.time()
        for i in range(count):
            self.assistant.add_message(start + i, f"msg-{i}")

        deadline = time.time() + 3.0
        while time.time() < deadline:
            with self.assistant.messages_lock:
                if len(self.assistant.messages) >= count:
                    break
            time.sleep(0.02)

        with self.assistant.messages_lock:
            buffered = len(self.assistant.messages)

        self.assertGreaterEqual(buffered, count, f"Expected at least {count} messages, got {buffered}")


if __name__ == "__main__":
    unittest.main()
