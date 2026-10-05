import unittest

from engine.transcript_filter import TranscriptFilter


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
        f = TranscriptFilter(keywords="Acme, WidgetPro")
        should_filter, reason = f.should_filter("We discussed WidgetPro integration timelines")
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

    def test_custom_rules_from_settings(self):
        f = TranscriptFilter.from_settings({"filter_exact": "okay then", "filter_regex": "^foo, [bad"})
        self.assertEqual(f.should_filter("Okay then"), (True, "exact-rule"))
        self.assertEqual(f.should_filter("foo bar baz"), (True, "regex-rule"))


if __name__ == "__main__":
    unittest.main()
