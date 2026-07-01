# Rusty — History

## Core Context

- **Project:** A Python meeting transcription application that will be migrated from a Windows GUI to a TUI/CLI experience.
- **Role:** Python Developer
- **Joined:** 2026-05-27T15:26:08.956Z

## Learnings

<!-- Append learnings below -->
- 2026-05-27: Python `Start` is the true session boundary: it drains queues, creates a fresh transcript file, clears transcript state, and starts six workers for mic/output capture, mic/output WAV+transcribe, Teams detection, and transcript consumption.
- 2026-05-27: Audio chunking uses 100 ms PCM frames by default, RMS silence threshold/duration, max-duration flush, residual stop flush, and a special output-only speaker-change split with a two-second overlap.
- 2026-05-27: Transcript queueing has a shutdown-drain risk: the display/store consumer stops on the shared stop event while transcription workers may still enqueue final entries. Go should stop capture, drain chunks/transcribers, then close the transcript sink.
- 2026-05-27: Go migration should preserve `.env` compatibility and transcript/filter semantics, but can drop Tkinter UI details, startup empty transcript files, global any-letter split shortcuts, and core coupling to Teams UI scraping.
- 2026-05-27: Final Go TUI implementation is approved; the Python behavior map informed retained capture/transcript/filter essentials while GUI, assistant, summaries, and local-Whisper scope stayed out.
- 2026-05-28: WSL2 Ubuntu validation for `transcribe/` ran against native Go 1.22.2 after installing the Ubuntu `golang-go` package; the reliable smoke checks are `go test ./...` and `go build -o transcribe ./cmd/transcribe` from `transcribe/`.
- 2026-05-28: DirectShow audio enumeration should not treat the first listed row as the microphone default; prefer the first non-loopback device so loopback/virtual output rows do not steal the mic default when several devices are present.
- 📌 Team update (2026-05-28T09:49:25Z): PulseAudio source indices are now preserved as aliases, so Linux device selection can target exact pactl rows in multi-device setups — decided by Basher.
- 2026-05-28: Windows `--list-devices` now falls back to PowerShell `Get-PnpDevice -Class AudioEndpoint` names when ffmpeg DirectShow returns no audio rows; the fallback keeps `FriendlyName` as the capture ID, preserves `InstanceId` as an alias, and was validated with `go test ./...`, `go build -o transcribe ./cmd/transcribe`, and targeted audio-discovery tests in WSL2 Ubuntu.
- 2026-05-28: The Go `transcribe/` app should not allow mic/output capture to resolve to the same physical device; if startup selection would duplicate the mic, prefer a distinct output candidate when available and otherwise fail with a clear output-capture error.
- 2026-05-28: `--logging 1` now writes `transcribe.log` in the current working directory, and the fail-fast device-selection rule keeps the same physical headset from being silently used for both mic and output capture.
