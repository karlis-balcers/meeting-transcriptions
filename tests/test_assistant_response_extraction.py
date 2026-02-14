import unittest

from assistant import Assistant


class AssistantResponseExtractionTests(unittest.TestCase):
    def setUp(self):
        # Avoid constructor side-effects (thread/env dependency)
        self.assistant = Assistant.__new__(Assistant)

    def test_extracts_top_level_output_text(self):
        data = {"output_text": " Hello world "}
        self.assertEqual(self.assistant._extract_text_from_response(data), "Hello world")

    def test_extracts_structured_message_chunks(self):
        data = {
            "output": [
                {"type": "file_search_call", "status": "completed"},
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Part 1"},
                        {"type": "text", "text": {"value": "Part 2"}},
                    ],
                },
            ]
        }
        self.assertEqual(self.assistant._extract_text_from_response(data), "Part 1\nPart 2")

    def test_returns_none_when_no_text(self):
        data = {"output": [{"type": "message", "content": [{"type": "tool_result"}]}]}
        self.assertIsNone(self.assistant._extract_text_from_response(data))


if __name__ == "__main__":
    unittest.main()
