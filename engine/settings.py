"""Persistent engine settings.

Settings live in a JSON file in the per-user config directory. They mirror the
options the old Python/Tk app kept in `.env` (minus the OpenAI assistant), plus
the local LLM ("Ollama") checks. On first run an existing `.env` is imported so
people coming from the Python version keep their name, keywords, folders, etc.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger("settings")

APP_DIR_NAME = "MeetingTranscriptions"

SUPPORTED_LANGUAGES = [
    "en", "lv", "ru", "de", "fr", "es", "it", "pt", "nl", "pl",
    "sv", "fi", "et", "lt", "ja", "zh", "ko", "ar", "tr", "no",
]

DEFAULT_MOOD_PROMPT = (
    "Classify the emotional tone of the LATEST utterance of the speaker, using the earlier lines only as context."
)
DEFAULT_FACT_PROMPT = (
    "Find concrete factual claims in the LATEST utterance (numbers, dates, names, technical or historical facts) "
    "and judge whether they are correct based on your knowledge. Opinions, plans and small talk are not claims."
)

APPLIES_TO = ("everyone", "others", "me")
LLM_APIS = ("laya", "ollama", "openai")
LAYA_URL = "http://127.0.0.1:8765"
SETTINGS_VERSION = 2
OLLAMA_URL = "http://127.0.0.1:11434"


def default_config_dir() -> Path:
    if sys.platform == "win32":
        base = os.getenv("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    base = os.getenv("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "meeting-transcriptions"


def default_output_dir() -> str:
    docs = Path.home() / "Documents"
    return str((docs if docs.exists() else Path.home()) / "Meeting Transcriptions")


def default_settings() -> dict[str, Any]:
    return {
        # General
        "your_name": "Me",
        "languages": "en",
        "keywords": "",
        "auto_start": False,
        "teams_window_name": "Meeting compact view*",
        "remote_speaker_name": "Remote",
        # OpenAI transcription
        "openai_api_key": "",
        "transcript_model": "gpt-4o-mini-transcribe",
        "transcribe_timeout_seconds": 60.0,
        "transcribe_max_retries": 3,
        "transcribe_retry_base_seconds": 1.0,
        # Audio
        "input_device_index": None,
        "input_device_name": "",
        "output_device_index": None,
        "output_device_name": "",
        "record_seconds": 300,
        "silence_threshold": 50.0,
        "silence_duration": 1.0,
        "frame_duration_ms": 100,
        # Directories
        "output_dir": default_output_dir(),
        "temp_dir": str(Path(tempfile.gettempdir()) / "meeting-transcriptions"),
        # Transcript filtering
        "filter_min_chars": 2,
        "filter_exact": "",
        "filter_prefixes": "",
        "filter_contains": "",
        "filter_regex": "",
        # Logging
        "log_level": "INFO",
        "log_file_max_mb": 5,
        "log_file_backup_count": 5,
        # Local LLM (Ollama or any OpenAI-compatible local server)
        "llm_enabled": False,
        # "laya" (local decision model), "ollama", or "openai" (OpenAI-compatible, e.g. llama.cpp / LM Studio)
        "llm_api": "laya",
        "settings_version": SETTINGS_VERSION,
        "llm_base_url": LAYA_URL,
        "llm_model": "llama3.2:3b",
        "llm_timeout_seconds": 30.0,
        "llm_context_lines": 6,
        "mood_enabled": True,
        "mood_prompt": DEFAULT_MOOD_PROMPT,
        "fact_check_enabled": True,
        "fact_check_prompt": DEFAULT_FACT_PROMPT,
        "fact_check_applies_to": "everyone",
        "custom_checks": [
            {
                "id": "action-items",
                "name": "Action item",
                "prompt": "Does the latest utterance assign a task, deadline or promise someone will do something?",
                "applies_to": "everyone",
                "enabled": False,
                "color": "#f5a524",
            },
        ],
    }


# Old .env keys from the Python version -> settings keys.
_ENV_IMPORT = {
    "YOUR_NAME": ("your_name", str),
    "LANGUAGE": ("languages", str),
    "KEYWORDS": ("keywords", str),
    "AUTO_START_TRANSCRIPTION": ("auto_start", "bool"),
    "OPENAI_API_KEY": ("openai_api_key", str),
    "OPENAI_MODEL_FOR_TRANSCRIPT": ("transcript_model", str),
    "TRANSCRIBE_API_TIMEOUT_SECONDS": ("transcribe_timeout_seconds", float),
    "TRANSCRIBE_API_MAX_RETRIES": ("transcribe_max_retries", int),
    "TRANSCRIBE_API_RETRY_BASE_SECONDS": ("transcribe_retry_base_seconds", float),
    "AUDIO_INPUT_DEVICE_INDEX": ("input_device_index", int),
    "AUDIO_INPUT_DEVICE_NAME": ("input_device_name", str),
    "AUDIO_OUTPUT_DEVICE_INDEX": ("output_device_index", int),
    "AUDIO_OUTPUT_DEVICE_NAME": ("output_device_name", str),
    "RECORD_SECONDS": ("record_seconds", int),
    "SILENCE_THRESHOLD": ("silence_threshold", float),
    "SILENCE_DURATION": ("silence_duration", float),
    "FRAME_DURATION_MS": ("frame_duration_ms", int),
    "OUTPUT_DIR": ("output_dir", str),
    "TEMP_DIR": ("temp_dir", str),
    "TRANSCRIPT_FILTER_MIN_CHARS": ("filter_min_chars", int),
    "TRANSCRIPT_FILTER_EXACT": ("filter_exact", str),
    "TRANSCRIPT_FILTER_PREFIXES": ("filter_prefixes", str),
    "TRANSCRIPT_FILTER_CONTAINS": ("filter_contains", str),
    "TRANSCRIPT_FILTER_REGEX": ("filter_regex", str),
    "LOG_LEVEL": ("log_level", str),
    "LOG_FILE_MAX_MB": ("log_file_max_mb", float),
    "LOG_FILE_BACKUP_COUNT": ("log_file_backup_count", int),
}

_NUMERIC_LIMITS = {
    "record_seconds": (5, 3600),
    "silence_threshold": (0.0, 32767.0),
    "silence_duration": (0.1, 30.0),
    "frame_duration_ms": (10, 1000),
    "filter_min_chars": (1, 100),
    "transcribe_timeout_seconds": (1.0, 600.0),
    "transcribe_max_retries": (0, 10),
    "transcribe_retry_base_seconds": (0.0, 60.0),
    "llm_timeout_seconds": (1.0, 600.0),
    "llm_context_lines": (0, 50),
    "log_file_max_mb": (0.1, 1024.0),
    "log_file_backup_count": (0, 100),
}


def parse_env_file(path: Path) -> dict[str, str]:
    """Minimal .env parser (KEY=VALUE, # comments, optional quotes)."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def _to_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def import_env(env: dict[str, str]) -> dict[str, Any]:
    """Translate an old-style `.env` mapping into settings keys."""
    imported: dict[str, Any] = {}
    for env_key, (key, kind) in _ENV_IMPORT.items():
        raw = env.get(env_key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            if kind == "bool":
                imported[key] = _to_bool(raw)
            else:
                imported[key] = kind(str(raw).strip())
        except ValueError:
            logger.warning("Ignoring invalid .env value %s=%r", env_key, raw)
    return imported


def parse_language_candidates(raw: str | None, fallback: str = "en") -> list[str]:
    seen: list[str] = []
    for token in (raw or "").lower().split(","):
        code = token.strip()
        if code and code in SUPPORTED_LANGUAGES and code not in seen:
            seen.append(code)
    return seen or [fallback]


def _coerce(key: str, value: Any, default: Any) -> Any:
    """Coerce a value to the type of its default, falling back on bad input."""
    if key in ("input_device_index", "output_device_index"):
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if key == "custom_checks":
        return normalize_checks(value)
    if isinstance(default, bool):
        return _to_bool(value)
    if isinstance(default, int):
        try:
            result = int(float(value))
        except (TypeError, ValueError):
            return default
    elif isinstance(default, float):
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
    else:
        return default if value is None else str(value)
    limits = _NUMERIC_LIMITS.get(key)
    if limits:
        result = type(default)(min(max(result, limits[0]), limits[1]))
    return result


def normalize_checks(raw: Any) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not name or not prompt:
            continue
        check_id = str(item.get("id") or "").strip() or uuid.uuid4().hex[:8]
        while check_id in seen:
            check_id = uuid.uuid4().hex[:8]
        seen.add(check_id)
        applies_to = str(item.get("applies_to") or "everyone")
        checks.append({
            "id": check_id,
            "name": name[:60],
            "prompt": prompt,
            "applies_to": applies_to if applies_to in APPLIES_TO else "everyone",
            "enabled": _to_bool(item.get("enabled", True)),
            "color": str(item.get("color") or "#7c8cff"),
        })
    return checks


def normalize(values: dict[str, Any]) -> dict[str, Any]:
    defaults = default_settings()
    result = copy.deepcopy(defaults)
    for key, default in defaults.items():
        if key in values:
            result[key] = _coerce(key, values[key], default)
    for key in ("fact_check_applies_to",):
        if result[key] not in APPLIES_TO:
            result[key] = "everyone"
    if result["llm_api"] not in LLM_APIS:
        result["llm_api"] = "laya"
    # Switching server type with the other one's default URL still in place: use this one's default.
    url = result["llm_base_url"].strip().rstrip("/")
    if result["llm_api"] == "laya" and url in ("", OLLAMA_URL):
        result["llm_base_url"] = LAYA_URL
    elif result["llm_api"] == "ollama" and url in ("", LAYA_URL):
        result["llm_base_url"] = OLLAMA_URL
    result["languages"] = ",".join(parse_language_candidates(result["languages"]))
    for key in ("output_dir", "temp_dir"):
        result[key] = os.path.expanduser(str(result[key]).strip()) or defaults[key]
    result["your_name"] = result["your_name"].strip() or defaults["your_name"]
    result["remote_speaker_name"] = result["remote_speaker_name"].strip() or defaults["remote_speaker_name"]
    return result


def migrate(data: dict[str, Any]) -> bool:
    """Bring a settings file from an older version up to date, in place. True if anything changed."""
    version = int(data.get("settings_version") or 1)
    if version >= SETTINGS_VERSION:
        return False
    if version < 2 and data.get("llm_api") == "ollama" and data.get("llm_base_url", OLLAMA_URL) == OLLAMA_URL:
        # v2.0.0-2.0.3 defaulted to Ollama; Laya is what the local AI was meant to be.
        data["llm_api"] = "laya"
        data["llm_base_url"] = LAYA_URL
        logger.info("Settings: local AI switched from Ollama to Laya (pick Ollama again in Settings if you want it)")
    data["settings_version"] = SETTINGS_VERSION
    return True


class SettingsStore:
    """Thread-safe settings file wrapper."""

    def __init__(self, path: Path | None = None, env_file: Path | None = None):
        self.path = Path(path) if path else default_config_dir() / "settings.json"
        self._lock = Lock()
        self._values = self._load(env_file if env_file is not None else Path.cwd() / ".env")

    def _load(self, env_file: Path) -> dict[str, Any]:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    migrated = migrate(data)
                    result = normalize(data)
                    if migrated:
                        self._write(result)
                    return result
            except (OSError, ValueError) as e:
                logger.warning("Could not read settings %s (%s); using defaults.", self.path, e)
            return normalize({})

        values: dict[str, Any] = {}
        if env_file and env_file.exists():
            values = import_env(parse_env_file(env_file))
            if values:
                logger.info("Imported %s settings from %s", len(values), env_file)
        result = normalize(values)
        self._write(result)
        return result

    def _write(self, values: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(values, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            logger.error("Could not save settings to %s: %s", self.path, e)

    def get(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._values)

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            merged = dict(self._values)
            merged.update(changes or {})
            self._values = normalize(merged)
            self._write(self._values)
            return copy.deepcopy(self._values)

    def api_key(self) -> str:
        with self._lock:
            return self._values.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")

    def public(self) -> dict[str, Any]:
        """Settings as sent to the UI: the API key is replaced by a flag."""
        values = self.get()
        values["openai_api_key_set"] = bool(values.pop("openai_api_key", "") or os.getenv("OPENAI_API_KEY"))
        values["config_path"] = str(self.path)
        values["supported_languages"] = list(SUPPORTED_LANGUAGES)
        return values
