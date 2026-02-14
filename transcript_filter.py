from __future__ import annotations

import os
import re
from typing import Optional


def _parse_list(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    items = []
    for part in raw.replace(";", ",").split(","):
        value = part.strip()
        if value:
            items.append(value)
    return items


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


class TranscriptFilter:
    """Configurable transcript artifact filter with light heuristics."""

    DEFAULT_EXACT = {
        "bye.",
        "um",
        "you",
        ".",
        "thank you.",
        "paldies!",
    }

    DEFAULT_PREFIXES = [
        "thanks for watching",
        "thank you for watching",
        "thanks for listening",
        "thank you for joining",
        "thank you very much",
        "thank you for tuning in",
        "thank you guys.",
    ]

    DEFAULT_CONTAINS = [
        "please subscribe to my channel",
        "www.northstarit.co.uk",
        "amara.org",
        "i'll see you in the next video",
        "brandhagen10.com",
        "www.abercap.com",
    ]

    DEFAULT_REGEX = [
        r"\b(?:https?://|www\.)\S+\b",
        r"\b(?:like|share|subscribe)\b.*\bchannel\b",
        r"^[\W_]+$",
    ]

    def __init__(self, keywords: Optional[str] = None):
        self.min_chars = max(1, int(os.getenv("TRANSCRIPT_FILTER_MIN_CHARS", "2")))

        exact_extra = {_normalize(v) for v in _parse_list(os.getenv("TRANSCRIPT_FILTER_EXACT"))}
        prefixes_extra = [_normalize(v) for v in _parse_list(os.getenv("TRANSCRIPT_FILTER_PREFIXES"))]
        contains_extra = [_normalize(v) for v in _parse_list(os.getenv("TRANSCRIPT_FILTER_CONTAINS"))]
        regex_extra = _parse_list(os.getenv("TRANSCRIPT_FILTER_REGEX"))

        self.exact = set(self.DEFAULT_EXACT) | {v for v in exact_extra if v}
        self.prefixes = [p for p in (self.DEFAULT_PREFIXES + prefixes_extra) if p]
        self.contains = [c for c in (self.DEFAULT_CONTAINS + contains_extra) if c]
        self.regexes = [re.compile(r, re.IGNORECASE) for r in (self.DEFAULT_REGEX + regex_extra) if r]

        self.keywords = [k.strip().lower() for k in _parse_list(keywords) if k.strip()]

    def _has_keyword(self, normalized_text: str) -> bool:
        if not self.keywords:
            return False
        return any(k in normalized_text for k in self.keywords)

    def should_filter(self, text: str) -> tuple[bool, str]:
        normalized = _normalize(text)
        if not normalized:
            return True, "empty"

        if self._has_keyword(normalized):
            return False, "contains-keyword"

        if normalized in self.exact:
            return True, "exact-rule"

        if any(normalized.startswith(prefix) for prefix in self.prefixes):
            return True, "prefix-rule"

        if any(fragment in normalized for fragment in self.contains):
            return True, "contains-rule"

        if any(rx.search(normalized) for rx in self.regexes):
            return True, "regex-rule"

        # Lightweight heuristic: very short alphabetic utterances are often noise
        letters_only = re.sub(r"[^a-z]", "", normalized)
        if 0 < len(letters_only) < self.min_chars:
            return True, "short-heuristic"

        return False, "accepted"
