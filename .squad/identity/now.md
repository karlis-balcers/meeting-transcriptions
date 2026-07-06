---
updated_at: 2026-07-06
focus_area: Root-level Go application maintenance and Windows WASAPI sidecar validation
active_issues: []
---

# What We're Focused On

The project is now a root-level Go application: `go.mod` is at the repository root, entrypoints live under `cmd/`, core packages under `internal/`, docs under `docs/`, and Windows build artifacts are produced by `build_transcribe_win64.bat` into `build/windows-amd64/`.

Active focus is maintaining the Go CLI/TUI transcription app and validating the pure-Go Windows WASAPI loopback sidecar. The old `transcribe/` module path is obsolete for current navigation except for stale artifacts under `transcribe/build/` if present.
