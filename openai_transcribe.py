from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, APIStatusError, RateLimitError
from segment import Segment
from typing import Iterable
import logging
import os
import time

logger = logging.getLogger("openai_transcribe")

FAKE_KEYWORD = "LAMPA"
FAKE_KEYWORD_2 = "MEMMEE"

class OpenAITranscribe:
    def __init__(self, model: str = "gpt-4o-mini-transcribe", language="en", keywords: str = None, status_callback=None):
        self.timeout_seconds = float(os.getenv("TRANSCRIBE_API_TIMEOUT_SECONDS", "60"))
        self.max_retries = int(os.getenv("TRANSCRIBE_API_MAX_RETRIES", "3"))
        self.retry_base_seconds = float(os.getenv("TRANSCRIBE_API_RETRY_BASE_SECONDS", "1.0"))
        self.client = OpenAI(timeout=self.timeout_seconds, max_retries=0)
        self.model = model
        self.language = language
        self.keywords = keywords
        self.initial_prompt = f"This discussion might mention {FAKE_KEYWORD}, {keywords}, {FAKE_KEYWORD_2}." if keywords else ""
        self.status_callback = status_callback

    def _emit_status(self, message: str, level: str = "info") -> None:
        if self.status_callback:
            try:
                self.status_callback(message, level)
            except Exception as e:
                logger.debug("Transcribe status callback failed: %s", e)

    def _backoff(self, attempt: int) -> float:
        return self.retry_base_seconds * (2 ** max(0, attempt - 1))

    def transcribe(self, audio_file_path: str) -> Iterable[Segment]:
        """
        Transcribe the audio file at `audio_file_path` using the OpenAI API.
        """
        segments = []
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
                        attempt,
                        max_attempts,
                        e,
                        delay,
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
                    self._emit_status("Transcription failed: authentication/permissions issue.", "error")
                    return []

                transient_status = status_code in (408, 409, 429) or (status_code is not None and status_code >= 500)
                if transient_status and attempt < max_attempts:
                    delay = self._backoff(attempt)
                    logger.warning(
                        "Transcription transient API status %s on attempt %s/%s. Retrying in %.2fs.",
                        status_code,
                        attempt,
                        max_attempts,
                        delay,
                    )
                    self._emit_status(f"Transcription service busy ({status_code}), retrying...", "warning")
                    time.sleep(delay)
                    continue

                if transient_status:
                    logger.error("Transcription failed after retries with status %s: %s", status_code, e)
                    self._emit_status(f"Transcription failed after retries ({status_code}).", "error")
                else:
                    logger.error("Transcription hard failure status %s: %s", status_code, e)
                    self._emit_status(f"Transcription failed ({status_code}).", "error")
                return []
            except Exception as e:
                logger.error("Transcription unexpected error: %s", e)
                self._emit_status("Transcription failed due to unexpected error.", "error")
                return []

        if transcription is None:
            self._emit_status("Transcription failed: no response.", "error")
            return []

        json_response = transcription.to_dict()
        if "text" not in json_response:
            logger.warning("No transcription text found in response: %s", json_response)
            self._emit_status("Transcription response missing text.", "warning")
        else:
            if transcription.text.upper().find(FAKE_KEYWORD) < 0 and transcription.text.upper().find(FAKE_KEYWORD_2) < 0 and transcription.text.find('###') < 0 and transcription.text.find('context/instructions') < 0:
                segment = Segment(start=0, end=1.0, text=transcription.text)
                segments.append(segment)
                self._emit_status("Transcription completed.", "info")
        return segments

