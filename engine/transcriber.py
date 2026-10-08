"""OpenAI audio transcription, restored from the Python version (openai_transcribe.py).

With a base URL it talks to any OpenAI-compatible speech-to-text server instead
(a local faster-whisper / Speaches server, LocalAI, Groq...)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger("transcriber")

# Sentinel words in the prompt. If they show up in the result the model echoed the
# prompt back instead of transcribing (it happens on silence), so the chunk is dropped.
FAKE_KEYWORD = "LAMPA"
FAKE_KEYWORD_2 = "MEMMEE"


@dataclass
class Segment:
    start: float
    end: float
    text: str


def build_prompt(keywords: Optional[str]) -> str:
    return f"This discussion might mention {FAKE_KEYWORD}, {keywords}, {FAKE_KEYWORD_2}." if keywords else ""


def is_prompt_echo(text: str) -> bool:
    upper = text.upper()
    return (
        FAKE_KEYWORD in upper
        or FAKE_KEYWORD_2 in upper
        or "###" in text
        or "context/instructions" in text
    )


class OpenAITranscribe:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini-transcribe",
        base_url: Optional[str] = None,
        language: str = "en",
        keywords: Optional[str] = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_base_seconds: float = 1.0,
        status_callback: Optional[Callable[[str, str], None]] = None,
    ):
        from openai import OpenAI  # imported lazily so tests don't need the SDK

        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_base_seconds = float(retry_base_seconds)
        # The SDK refuses to start without a key; local servers ignore it, so give them a placeholder.
        self.client = OpenAI(api_key=api_key or ("not-needed" if base_url else None), base_url=base_url or None,
                             timeout=self.timeout_seconds, max_retries=0)
        self.model = model
        self.language = language
        self.keywords = keywords
        self.initial_prompt = build_prompt(keywords)
        self.status_callback = status_callback

    def _emit_status(self, message: str, level: str = "info") -> None:
        if self.status_callback:
            try:
                self.status_callback(message, level)
            except Exception as e:
                logger.debug("Transcribe status callback failed: %s", e)

    def _backoff(self, attempt: int) -> float:
        return self.retry_base_seconds * (2 ** max(0, attempt - 1))

    def transcribe(self, audio_file_path: str) -> list[Segment]:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

        transcription = None
        max_attempts = max(1, self.max_retries + 1)

        for attempt in range(1, max_attempts + 1):
            try:
                with open(audio_file_path, "rb") as audio_file:
                    transcription = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        prompt=self.initial_prompt,
                        language=self.language,
                    )
                logger.debug("Transcription response: %s", transcription)
                break
            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                if attempt < max_attempts:
                    delay = self._backoff(attempt)
                    logger.warning(
                        "Transcription transient error on attempt %s/%s: %s. Retrying in %.2fs.",
                        attempt, max_attempts, e, delay,
                    )
                    self._emit_status("Transcription transient API issue, retrying...", "warning")
                    time.sleep(delay)
                    continue
                logger.error("Transcription failed after %s attempts: %s", max_attempts, e)
                self._emit_status("Transcription failed after retries.", "error")
                return []
            except APIStatusError as e:
                status_code = e.status_code
                if status_code in (401, 403):
                    logger.error("Transcription hard failure %s (auth/permission): %s", status_code, e)
                    self._emit_status("Transcription failed: check the transcription API key.", "error")
                    return []
                transient = status_code in (408, 409, 429) or (status_code is not None and status_code >= 500)
                if transient and attempt < max_attempts:
                    delay = self._backoff(attempt)
                    logger.warning(
                        "Transcription transient API status %s on attempt %s/%s. Retrying in %.2fs.",
                        status_code, attempt, max_attempts, delay,
                    )
                    self._emit_status(f"Transcription service busy ({status_code}), retrying...", "warning")
                    time.sleep(delay)
                    continue
                logger.error("Transcription failed with status %s: %s", status_code, e)
                self._emit_status(f"Transcription failed ({status_code}).", "error")
                return []
            except Exception as e:
                logger.error("Transcription unexpected error: %s", e)
                self._emit_status("Transcription failed due to unexpected error.", "error")
                return []

        if transcription is None:
            self._emit_status("Transcription failed: no response.", "error")
            return []

        text = getattr(transcription, "text", None)
        if not text:
            logger.debug("No transcription text in response: %s", transcription)
            return []
        if is_prompt_echo(text):
            logger.debug("Dropping prompt echo: %s", text)
            return []
        return [Segment(start=0, end=1.0, text=text)]
