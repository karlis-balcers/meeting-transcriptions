from __future__ import annotations

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

    def __init__(
        self,
        keywords: Optional[str] = None,
        min_chars: int = 2,
        exact: Optional[str] = None,
        prefixes: Optional[str] = None,
        contains: Optional[str] = None,
        regex: Optional[str] = None,
    ):
        self.min_chars = max(1, int(min_chars))

        exact_extra = {_normalize(v) for v in _parse_list(exact)}
        prefixes_extra = [_normalize(v) for v in _parse_list(prefixes)]
        contains_extra = [_normalize(v) for v in _parse_list(contains)]
        regex_extra = _parse_list(regex)

        self.exact = set(self.DEFAULT_EXACT) | {v for v in exact_extra if v}
        self.prefixes = [p for p in (self.DEFAULT_PREFIXES + prefixes_extra) if p]
        self.contains = [c for c in (self.DEFAULT_CONTAINS + contains_extra) if c]
        self.regexes = []
        for pattern in self.DEFAULT_REGEX + regex_extra:
            try:
                self.regexes.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                continue

        self.keywords = [k.strip().lower() for k in _parse_list(keywords) if k.strip()]

    @classmethod
    def from_settings(cls, settings: dict) -> "TranscriptFilter":
        return cls(
            keywords=settings.get("keywords"),
            min_chars=settings.get("filter_min_chars", 2),
            exact=settings.get("filter_exact"),
            prefixes=settings.get("filter_prefixes"),
            contains=settings.get("filter_contains"),
            regex=settings.get("filter_regex"),
        )

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
