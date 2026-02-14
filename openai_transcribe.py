from openai import OpenAI
from segment import Segment
from typing import Iterable
import logging

logger = logging.getLogger("openai_transcribe")

FAKE_KEYWORD = "LAMPA"
FAKE_KEYWORD_2 = "MEMMEE"

class OpenAITranscribe:
    def __init__(self, model: str = "gpt-4o-mini-transcribe", language="en", keywords: str = None):
        self.client = OpenAI()
        self.model = model
        self.language = language
        self.keywords = keywords
        self.initial_prompt = f"This discussion might mention {FAKE_KEYWORD}, {keywords}, {FAKE_KEYWORD_2}." if keywords else ""

    def transcribe(self, audio_file_path: str) -> Iterable[Segment]:
        """
        Transcribe the audio file at `audio_file_path` using the OpenAI API.
        """
        segments = []
        with open(audio_file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                prompt=self.initial_prompt,
                language=self.language,
            )
            logger.debug("Transcription response: %s", transcription)
        json_response = transcription.to_dict()
        if "text" not in json_response:
            logger.warning("No transcription text found in response: %s", json_response)
        else:
            if transcription.text.upper().find(FAKE_KEYWORD) < 0 and transcription.text.upper().find(FAKE_KEYWORD_2) < 0 and transcription.text.find('###') < 0 and transcription.text.find('context/instructions') < 0:
                segment = Segment(start=0, end=1.0, text=transcription.text)
                segments.append(segment)
        return segments

