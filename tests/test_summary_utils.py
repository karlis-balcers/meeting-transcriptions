import unittest

from summary_utils import sanitize_title_for_filename


class SummaryUtilsTests(unittest.TestCase):
    def test_sanitizes_and_normalizes_title(self):
        title = "Roadmap / Decisions: Q1!"
        self.assertEqual(sanitize_title_for_filename(title), "roadmap__decisions_q1")

    def test_applies_max_length(self):
        title = "A" * 100
        self.assertEqual(len(sanitize_title_for_filename(title, max_length=10)), 10)

    def test_uses_default_when_empty(self):
        self.assertEqual(sanitize_title_for_filename("!!!", default="fallback"), "fallback")


if __name__ == "__main__":
    unittest.main()
