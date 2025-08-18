from openai import OpenAI
from segment import Segment
from typing import Iterable

FAKE_KEYWORD = "LAMPA"

class OpenAITranscribe:
    def __init__(self, model: str = "gpt-4o-mini-transcribe", language="en", keywords: str = None):
        self.client = OpenAI()
        self.model = model
        self.language = language
        self.keywords = keywords
        self.initial_prompt = f"This discussion might mention {FAKE_KEYWORD}, {keywords}." if keywords else ""

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
            print(f"Transcription response: {transcription}")
        json_response = transcription.to_dict()
        if "text" not in json_response:
            print(f"Error: No transcription found: {json_response}")
        else:
            if transcription.text.find(FAKE_KEYWORD) < 0 and transcription.text.find('###') < 0 and transcription.text.find('context/instructions') < 0:
                segment = Segment(start=0, end=1.0, text=transcription.text)
                segments.append(segment)
        return segments

