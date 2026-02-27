import unittest

from env_utils import env_str, env_int, env_float, env_bool, parse_language_candidates


class EnvUtilsTests(unittest.TestCase):
    def test_env_str_default_and_trim(self):
        env = {"NAME": "  Karlis  "}
        self.assertEqual(env_str(env, "NAME", "x"), "Karlis")
        self.assertEqual(env_str(env, "MISSING", "x"), "x")

    def test_env_int_parsing_and_warning(self):
        warnings = []
        env = {"N": "42", "BAD": "oops"}
        self.assertEqual(env_int(env, "N", 7, warnings), 42)
        self.assertEqual(env_int(env, "BAD", 7, warnings), 7)
        self.assertTrue(any("Invalid integer" in w for w in warnings))

    def test_env_float_parsing_and_warning(self):
        warnings = []
        env = {"F": "1.5", "BAD": "nope"}
        self.assertAlmostEqual(env_float(env, "F", 0.2, warnings), 1.5)
        self.assertAlmostEqual(env_float(env, "BAD", 0.2, warnings), 0.2)
        self.assertTrue(any("Invalid number" in w for w in warnings))

    def test_env_bool_truthy_falsy_and_warning(self):
        warnings = []
        env = {"A": "true", "B": "off", "C": "maybe"}
        self.assertTrue(env_bool(env, "A", False, warnings))
        self.assertFalse(env_bool(env, "B", True, warnings))
        self.assertTrue(env_bool(env, "C", True, warnings))
        self.assertTrue(any("Invalid boolean" in w for w in warnings))

    def test_parse_language_candidates_single(self):
        warnings = []
        langs = parse_language_candidates("en", ["en", "no", "lv"], warnings)
        self.assertEqual(langs, ["en"])
        self.assertEqual(warnings, [])

    def test_parse_language_candidates_multiple(self):
        warnings = []
        langs = parse_language_candidates("en,no", ["en", "no", "lv"], warnings)
        self.assertEqual(langs, ["en", "no"])
        self.assertEqual(warnings, [])

    def test_parse_language_candidates_with_invalid(self):
        warnings = []
        langs = parse_language_candidates("EN, no ,xx", ["en", "no", "lv"], warnings)
        self.assertEqual(langs, ["en", "no"])
        self.assertTrue(any("unsupported code 'xx'" in w for w in warnings))

    def test_parse_language_candidates_empty_fallback(self):
        warnings = []
        langs = parse_language_candidates(", ,", ["en", "no", "lv"], warnings)
        self.assertEqual(langs, ["en"])
        self.assertTrue(any("has no supported codes" in w for w in warnings))

    def test_parse_language_candidates_dedupes(self):
        warnings = []
        langs = parse_language_candidates("en,en,no", ["en", "no", "lv"], warnings)
        self.assertEqual(langs, ["en", "no"])
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
