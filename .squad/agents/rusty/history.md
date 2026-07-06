# Rusty — History

## Core Context

- **Project:** A root-level Go meeting transcription CLI/TUI application with Windows WASAPI sidecar validation in progress.
- **Role:** Go/Core Developer
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
- 2026-07-03 (Danny §4a — ffmpeg unify): Deleted `transcribe/internal/audio/windows_helper.go` entirely (Python WASAPI layer) and unified Windows output capture on the ffmpeg DirectShow path (`-f dshow -i audio=<id>`), identical to the mic.
	- Key gotcha: `windows_helper.go` did NOT only hold the Python layer — it also defined `commandForRequest()` and `platform()`, which `ffmpeg.go` and `ffmpeg_test.go` call. So "delete the file" required relocating cleaned copies of those two functions into `ffmpeg.go` (with the Windows-output Python branch stripped). It also contained `resolveExistingFile`/`normalizeExecutableCandidate`/`windowsPythonCandidates`, which were used only inside that file.
	- `ffmpeg.go`: removed `PythonPath`/`HelperScriptPath` from the `ExternalRecorder` struct (their assignments in `transcribe/internal/cli/root.go` were removed by Livingston in parallel — that file is NOT mine), removed the `backendName = "Windows output helper"` error branch in `RecordChunk`, and dropped the now-redundant Windows-output `commandForRequest` probe in `Validate`.
	- `discover.go`: refined `windowsOutputCaptureWarning` so it names the concrete fix (enable Stereo Mix, or install VB-CABLE/Voicemeeter/virtual-audio-capturer, then re-run `transcribe --list-devices`). Kept the literal phrases `loopback` and `output-capture` because existing `discover_test.go` asserts on them.
	- `audio.go`: added a selection-time warning in `SelectDevices` — on Windows, if the selected output device does not look like a true loopback (via `likelyDShowLoopback` over ID/Name/aliases), emit one actionable warning. This covers the case discovery's enumeration-time warning misses: a user explicitly picking a render-only `Speakers/Headphones` endpoint that ffmpeg DirectShow exposes as display-only.
	- `ffmpeg_test.go`: replaced the 3 Python-helper tests with `TestWindowsOutputRecorderUsesFFmpegDirectShow` (asserts `-f dshow` + `audio=<raw output name>`, no `audio_capture.py`) and `TestWindowsOutputRecorderValidationRequiresFFmpegOnly`.
	- Test-environment gotcha (for future Rusty): a "Validate must fail when ffmpeg missing" assertion is fragile because `Validate` only calls `r.ffmpeg()`, which trusts any non-empty `FFmpegPath` without stat-ing the binary, and `exec.LookPath("ffmpeg")` succeeds on hosts with ffmpeg installed (e.g. WSL where I validate). Don't write that assertion; instead pin behavior via a configured `FFmpegPath`.
- 2026-07-03: Full validation in WSL2 Ubuntu go1.22.2: `go vet ./internal/audio/...` clean, `go test ./internal/audio/...` all pass, and `go build ./...` is GREEN — Livingston's matching `root.go` call-site removal had already landed, so no struct-field coordination gap remains. Team note dropped at `.squad/decisions/inbox/rusty-audio-unify-ffmpeg.md`.
- 2026-07-06: Core work now starts from the repo root Go module: `go.mod` is at root, app entrypoints are in `cmd/`, shared logic is in `internal/`, docs are in `docs/`, Windows builds use `build_transcribe_win64.bat`, and outputs land in `build/windows-amd64/`; do not navigate to old `transcribe/` except to recognize stale `transcribe/build/` artifacts.
