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
- 2026-05-28: Windows DirectShow `--list-devices` needs row-level audio/video classification, not just a later audio-section header; a video-first listing can still surface real microphones and loopback devices if the parser honors explicit `(audio)/(video)` markers and audio-like aliases.
- 2026-05-28: WSL2 Ubuntu validation for `transcribe/` used native Go 1.22.2 after installing the Ubuntu `golang-go` package; run `go test ./...` and `go build -o transcribe ./cmd/transcribe` from `transcribe/` when checking Linux audio behavior.
- 2026-05-28: PulseAudio discovery now preserves the numeric column from `pactl list short sources` as a device alias, and alias-aware matching lets `--mic`, `--output`, `AUDIO_INPUT_DEVICE_INDEX`, and `AUDIO_OUTPUT_DEVICE_INDEX` target the exact source row in multi-device Linux setups.
- 2026-05-28: The TUI device picker in `internal/tui/model.go` is source-filtered and must keep listing/navigation coverage for multiple microphones and output devices so `j`/`k` + Enter can select the second or later device, not just the first row.
- 2026-05-28: `--logging 1` writes `transcribe.log` in the current working directory; use it to capture device discovery, capture, and transcription breadcrumbs without polluting stdout.
- 2026-05-28: When a chunk is transcribed but filtered out, keep logging that path so the session does not look silently stuck or looping.
- 📌 Team update (2026-05-28T09:49:25Z): DirectShow microphone defaults now skip loopback candidates so Windows mic defaults prefer real microphones — decided by Rusty.
- 2026-05-28: Filtered-but-nonempty transcription chunks should still be logged, and `--logging 1` remains the shared diagnostics switch for following chunk capture and OpenAI transcription progress.
