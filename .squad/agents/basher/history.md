# Basher — History

## Core Context

- **Project:** A Python meeting transcription application that will be migrated from a Windows GUI to a TUI/CLI experience.
- **Role:** QA Engineer
- **Joined:** 2026-05-27T15:26:08.967Z

## Learnings

<!-- Append learnings below -->
- 2026-05-27: Current Python tests cover utility seams (env parsing, device selection, transcript filtering, assistant response extraction), but the Go TUI rebuild needs fakeable ports for audio, OpenAI, device discovery, filesystem/clock/stdout, and the TUI model from day one.
- 2026-05-27: Silent-mode QA must assert stdout is exactly empty; logs/status/errors belong on stderr or in log files unless a machine-readable stdout mode is explicitly specified.
- 2026-05-27: Go TUI review added workflow tests for silent stdout, API-key fail-fast, and P/R/M/U/S plus settings M/O shortcuts; build/test/vet/cross-compile passed, but live capture semantics still need independent revision.
- 2026-05-27: Danny's Go TUI revision fixed the live-control rejection with cancellable short capture chunks, RMS-derived meter events, and real Windows Teams title polling; full Go validation matrix passed on Linux amd64.
- 2026-05-27: Coordinator independently confirmed Basher's approval with clean editor diagnostics plus Go tests/vet/race/native build/cross-compile checks from `transcribe/`.
- 2026-05-27: Windows DirectShow regressions need tests for stderr device parsing, audio-vs-video section boundaries, alternative-name aliases, default-to-concrete selection, and recorder validation before ffmpeg can receive `audio=default`.
- 2026-05-27: Windows DirectShow audio fix is approved; regression coverage now verifies dshow parsing, aliases, default resolution, ffmpeg argument behavior, and recorder validation.
