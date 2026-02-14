import unittest

from transcript_filter import TranscriptFilter


class TranscriptFilterTests(unittest.TestCase):
    def test_filters_known_artifact_phrase(self):
        f = TranscriptFilter()
        should_filter, reason = f.should_filter("Thanks for watching everyone")
        self.assertTrue(should_filter)
        self.assertEqual(reason, "prefix-rule")

    def test_filters_url_artifact(self):
        f = TranscriptFilter()
        should_filter, reason = f.should_filter("Visit www.example.com for details")
        self.assertTrue(should_filter)
        self.assertEqual(reason, "regex-rule")

    def test_keeps_text_with_keywords(self):
        f = TranscriptFilter(keywords="Paymentology, Flexpay")
        should_filter, reason = f.should_filter("We discussed Flexpay integration timelines")
        self.assertFalse(should_filter)
        self.assertEqual(reason, "contains-keyword")

    def test_short_noise_heuristic(self):
        f = TranscriptFilter()
        should_filter, reason = f.should_filter("a")
        self.assertTrue(should_filter)
        self.assertEqual(reason, "short-heuristic")

    def test_accepts_normal_sentence(self):
        f = TranscriptFilter()
        should_filter, reason = f.should_filter("Let's review the release plan and owner assignments")
        self.assertFalse(should_filter)
        self.assertEqual(reason, "accepted")


if __name__ == "__main__":
    unittest.main()
