---
updated_at: 2026-05-29T10:50:00Z
focus_area: Windows output-capture validation remains blocked on DirectShow-only ffmpeg capture; UI polish and win64 validation are complete
active_issues: []
---

# What We're Focused On

Windows loopback/output capture still falls back to microphone-only on this host because the current ffmpeg build only supports DirectShow capture; synthesized `Speakers/Headphones [Loopback]` devices are display-only.

Completed this session: green active shortcut styling in the TUI footer, `R Resume` renamed to `R Record`, Windows amd64 binary rebuilt successfully, and tests passed.
