from openai import OpenAI
from segment import Segment
from typing import Iterable

class OpenAITranscribe:
    def __init__(self, model: str = "gpt-4o-mini-transcribe", language="en", keywords: str = None):
        self.client = OpenAI()
        self.model = model
        self.language = language
        self.keywords = keywords
        self.initial_prompt = f"This discussion might mention {keywords}." if keywords else ""

    def transcribe(self, audio_file_path: str) -> Iterable[Segment]:
        """
        Transcribe the audio file at `audio_file_path` using the OpenAI API.
        """
        with open(audio_file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                prompt=self.initial_prompt,
                language=self.language,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        if not transcription.segments:
            print(f"Error: No transcription segments found for {audio_file_path}")
            return []

        segments = [Segment(start=seg['start'], end=seg['end'], text=seg['text']) for seg in transcription.segments]
        return segments

