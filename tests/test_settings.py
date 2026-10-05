import json
import tempfile
import unittest
from pathlib import Path

from engine.settings import SettingsStore, import_env, normalize, normalize_checks, parse_env_file


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_imports_old_env_file_on_first_run(self):
        env = self.dir / ".env"
        env.write_text(
            "YOUR_NAME=Karlis\nLANGUAGE=lv, en\nOUTPUT_DIR=/tmp/out\nSILENCE_THRESHOLD=70\n"
            "AUTO_START_TRANSCRIPTION=True\nKEYWORDS=\"Acme, Widget\"\n# comment\nOPENAI_API_KEY=sk-test\n",
            encoding="utf-8",
        )
        store = SettingsStore(self.dir / "settings.json", env_file=env)
        values = store.get()
        self.assertEqual(values["your_name"], "Karlis")
        self.assertEqual(values["languages"], "lv,en")
        self.assertEqual(values["output_dir"], "/tmp/out")
        self.assertEqual(values["silence_threshold"], 70.0)
        self.assertTrue(values["auto_start"])
        self.assertEqual(values["keywords"], "Acme, Widget")
        self.assertTrue((self.dir / "settings.json").exists())

    def test_public_settings_hide_api_key(self):
        store = SettingsStore(self.dir / "settings.json", env_file=self.dir / "missing.env")
        store.update({"openai_api_key": "sk-secret"})
        public = store.public()
        self.assertNotIn("openai_api_key", public)
        self.assertTrue(public["openai_api_key_set"])
        self.assertNotIn("sk-secret", json.dumps(public))

    def test_update_persists_and_clamps(self):
        path = self.dir / "settings.json"
        store = SettingsStore(path, env_file=self.dir / "missing.env")
        store.update({"frame_duration_ms": 5, "record_seconds": "60", "llm_api": "weird", "languages": "xx"})
        reloaded = SettingsStore(path).get()
        self.assertEqual(reloaded["frame_duration_ms"], 10)
        self.assertEqual(reloaded["record_seconds"], 60)
        self.assertEqual(reloaded["llm_api"], "ollama")
        self.assertEqual(reloaded["languages"], "en")

    def test_normalize_checks_drops_invalid_and_dedupes_ids(self):
        checks = normalize_checks([
            {"id": "a", "name": "One", "prompt": "p1"},
            {"id": "a", "name": "Two", "prompt": "p2", "applies_to": "nobody"},
            {"name": "", "prompt": "missing name"},
            "junk",
        ])
        self.assertEqual(len(checks), 2)
        self.assertNotEqual(checks[0]["id"], checks[1]["id"])
        self.assertEqual(checks[1]["applies_to"], "everyone")

    def test_parse_env_file_and_import_ignore_bad_numbers(self):
        env = self.dir / ".env"
        env.write_text("RECORD_SECONDS=abc\nexport TEMP_DIR='/x y'\n", encoding="utf-8")
        values = import_env(parse_env_file(env))
        self.assertNotIn("record_seconds", values)
        self.assertEqual(values["temp_dir"], "/x y")

    def test_normalize_keeps_null_device_index(self):
        values = normalize({"input_device_index": "", "output_device_index": "4"})
        self.assertIsNone(values["input_device_index"])
        self.assertEqual(values["output_device_index"], 4)


if __name__ == "__main__":
    unittest.main()
