import unittest

from engine.checks import CheckRunner, Utterance, applies, parse_custom, parse_fact, parse_mood
from engine.llm import LLMError, extract_json


class FakeClient:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def chat_json(self, system, user):
        self.calls.append((system, user))
        for key, reply in self.replies.items():
            if key in system:
                if isinstance(reply, Exception):
                    raise reply
                return reply
        return {}


class ParserTests(unittest.TestCase):
    def test_extract_json_from_wrapped_reply(self):
        self.assertEqual(extract_json('Sure! ```json\n{"hit": true, "note": "a {b}"}\n```'), {"hit": True, "note": "a {b}"})
        self.assertEqual(extract_json('prefix {"a": 1} suffix'), {"a": 1})
        with self.assertRaises(LLMError):
            extract_json("no json here")

    def test_parse_mood(self):
        self.assertEqual(parse_mood({"mood": "Angry", "intensity": 1})["valence"], -0.8)
        self.assertIsNone(parse_mood({"mood": "sleepy"}))

    def test_parse_fact_skips_no_claim(self):
        self.assertIsNone(parse_fact({"verdict": "no_claim", "claim": "x"}))
        self.assertEqual(parse_fact({"verdict": "incorrect", "claim": "Paris is in Spain"})["verdict"], "incorrect")

    def test_parse_custom(self):
        self.assertIsNone(parse_custom({"hit": False}))
        self.assertEqual(parse_custom({"hit": "yes", "label": "Deadline"})["label"], "Deadline")

    def test_applies(self):
        self.assertTrue(applies("me", True))
        self.assertFalse(applies("me", False))
        self.assertTrue(applies("others", False))
        self.assertTrue(applies("everyone", True))


class RunnerTests(unittest.TestCase):
    def test_runs_enabled_checks_and_reports(self):
        settings = {
            "llm_enabled": True,
            "mood_enabled": True,
            "mood_prompt": "",
            "fact_check_enabled": True,
            "fact_check_prompt": "",
            "fact_check_applies_to": "others",
            "custom_checks": [
                {"id": "act", "name": "Action item", "prompt": "task?", "applies_to": "everyone", "enabled": True},
                {"id": "off", "name": "Disabled", "prompt": "x", "applies_to": "everyone", "enabled": False},
            ],
        }
        client = FakeClient({
            "emotional tone": {"mood": "happy", "intensity": 0.5},
            "fact-check": {"verdict": "incorrect", "claim": "The moon is cheese", "note": "It is rock."},
            "Action item": {"hit": True, "label": "Task", "note": "Anna will send notes"},
        })
        events, moods, facts, hits = [], [], [], []
        runner = CheckRunner(
            settings_getter=lambda: settings, emit=events.append,
            on_mood=lambda *a: moods.append(a), on_fact=lambda *a: facts.append(a), on_hit=lambda *a: hits.append(a),
            client_factory=lambda s: client,
        )
        runner.run_checks(Utterance("u1", "Anna", "The moon is cheese, I'll send notes", 10.0, False, ["Me: hi"]))
        kinds = [e["kind"] for e in events]
        self.assertEqual(kinds, ["mood", "fact", "custom"])
        self.assertEqual(moods[0][:2], ("Anna", "happy"))
        self.assertEqual(facts, [("Anna", "incorrect")])
        self.assertEqual(hits, [("Anna", "Action item")])
        self.assertEqual(len(client.calls), 3)
        self.assertIn("Me: hi", client.calls[0][1])

        # Fact check only applies to others, so my own line skips it.
        events.clear()
        runner.run_checks(Utterance("u2", "Me", "hello", 11.0, True))
        self.assertEqual([e["kind"] for e in events], ["mood", "custom"])

    def test_llm_errors_become_one_status(self):
        settings = {"llm_enabled": True, "mood_enabled": True, "custom_checks": []}
        events = []
        runner = CheckRunner(lambda: settings, events.append,
                             client_factory=lambda s: FakeClient({"emotional": LLMError("down")}))
        runner.run_checks(Utterance("u1", "A", "x", 0, False))
        runner.run_checks(Utterance("u2", "A", "y", 0, False))
        self.assertEqual([e["type"] for e in events], ["status"])

    def test_submit_ignored_when_disabled(self):
        runner = CheckRunner(lambda: {"llm_enabled": False}, lambda e: None)
        runner.submit(Utterance("u1", "A", "x", 0, False))
        self.assertEqual(len(runner._queue), 0)


if __name__ == "__main__":
    unittest.main()
