from __future__ import annotations


def sanitize_title_for_filename(title: str, max_length: int = 50, default: str = "meeting_summary") -> str:
    safe_title = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in (title or ""))
    safe_title = safe_title.replace(" ", "_").lower().strip("_")
    if max_length > 0:
        safe_title = safe_title[:max_length]
    return safe_title or default
