
class Segment:
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text

    def __repr__(self):
        return f"Segment(start={self.start}, end={self.end}, text='{self.text}')"

    def to_dict(self):
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text
        }