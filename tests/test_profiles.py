import tempfile
import unittest

from engine.profiles import MeetingStats, ProfileBook, count_questions, summarize_profile


class MeetingStatsTests(unittest.TestCase):
    def test_counts_words_questions_and_talk_share(self):
        stats = MeetingStats()
        stats.add_utterance("Me", "What is the plan for the release?", 100.0, 4.0, True)
        stats.add_utterance("Anna B", "We ship the release on Friday, um, probably.", 105.0, 6.0, False)
        snap = stats.snapshot()["speakers"]
        self.assertEqual(snap["Me"]["questions"], 1)
        self.assertEqual(snap["Anna B"]["words"], 8)
        self.assertEqual(snap["Anna B"]["fillers"], 1)
        self.assertAlmostEqual(snap["Me"]["talk_share"] + snap["Anna B"]["talk_share"], 1.0, places=2)
        self.assertIn("release", snap["Anna B"]["top_topics"])

    def test_interruption_and_transitions(self):
        stats = MeetingStats()
        stats.add_utterance("A", "a long explanation going on and on", 0.0, 10.0, False)
        stats.add_utterance("B", "sorry to cut in here", 5.0, 3.0, False)
        snap = stats.snapshot()
        self.assertEqual(snap["speakers"]["B"]["interruptions"], 1)
        self.assertEqual(snap["transitions"], [{"from": "A", "to": "B", "count": 1}])

    def test_rename_merges_into_existing_speaker(self):
        stats = MeetingStats()
        stats.add_utterance("Remote", "hello there everyone", 0.0, 3.0, False)
        stats.add_utterance("Anna", "good morning", 5.0, 2.0, False)
        stats.add_mood("Remote", "happy", 0.8, 1.0)
        stats.rename("Remote", "Anna")
        snap = stats.snapshot()["speakers"]
        self.assertNotIn("Remote", snap)
        self.assertEqual(snap["Anna"]["utterances"], 2)
        self.assertEqual(snap["Anna"]["mood_counts"], {"happy": 1})

    def test_question_without_mark(self):
        self.assertEqual(count_questions("can you share the screen"), 1)
        self.assertEqual(count_questions("I can share it"), 0)


class ProfileBookTests(unittest.TestCase):
    def test_merge_save_reload_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = MeetingStats()
            stats.add_utterance("Anna", "we should review the budget numbers today", 0.0, 4.0, False)
            stats.add_mood("Anna", "calm", 0.3, 1.0)
            book = ProfileBook(tmp)
            book.merge_meeting(stats.snapshot(), "Weekly sync")
            book.merge_meeting(stats.snapshot(), "Weekly sync 2")

            reloaded = ProfileBook(tmp).all()
            self.assertEqual(reloaded["Anna"]["meetings"], 2)
            self.assertEqual(len(reloaded["Anna"]["history"]), 2)
            summary = summarize_profile(reloaded["Anna"])
            self.assertEqual(summary["top_mood"], "calm")
            self.assertGreater(summary["wpm"], 0)

            ProfileBook(tmp).rename("Anna", "Anna B")
            self.assertIn("Anna B", ProfileBook(tmp).all())


if __name__ == "__main__":
    unittest.main()
