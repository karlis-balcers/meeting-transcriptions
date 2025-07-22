from faster_whisper import WhisperModel
from segment import Segment
from typing import Iterable

class FastWhisperTranscribe:
    # Expected one of: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, distil-small.en
    def __init__(self, model_name="large-v3", device="cuda", language="en", keywords: str = None):
        self.model = WhisperModel(model_name, device=device, compute_type="float16")
        self.language = language
        self.vad_filter = True
        self.initial_prompt = f"This discussion might mention {keywords}" if keywords else None


    def transcribe(self, audio_file_path: str) ->  Iterable[Segment]:
        whisper_segments, _ = self.model.transcribe(
            audio_file_path, beam_size=5, language=self.language, vad_filter=self.vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=True,
            initial_prompt=self.initial_prompt
        )
        if not whisper_segments:
            print(f"Error: No transcription found for {audio_file_path}")
            return []
        segments = [Segment(start=seg.start, end=seg.end, text=seg.text) for seg in whisper_segments]
        return segments