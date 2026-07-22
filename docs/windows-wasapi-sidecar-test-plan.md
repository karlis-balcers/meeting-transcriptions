# Windows WASAPI Recorder Sidecar Test Plan

**Status:** Implementation landed; manual hardware QA still pending — Livingston, 2026-07-06  
**Scope:** Windows recorder-sidecar / recorder-helper architecture for `transcribe/`  
**Directive under test:** Windows recording moves behind recorder executable(s). OS-specific recording abstractions live only in the recording part. Audio-file processing remains OS-independent.

This plan supersedes the Windows-output portions of the older `cross-platform-test-plan.md` Path-C/DirectShow plan. The tree now includes a pure-Go Windows `wasapi-loopback-recorder.exe` sidecar for render-endpoint loopback capture plus hardware-free tests for the helper protocol, WAV writer, format conversion, and main-app sidecar dispatch. The manual matrix below still needs confirmation on real Windows audio hardware.

## Quality bars

- Windows system-output capture must work on a clean stock Windows setup with **no Stereo Mix**, **no VB-CABLE**, and **no Voicemeeter** by using WASAPI loopback, equivalent to Audacity's Windows WASAPI loopback recording mode.
- Built-in Realtek speakers/headphones must be capturable as output without installing virtual audio devices.
- USB and Bluetooth render devices must be discoverable/selectable, including default-device switching during a session.
- Helper cancellation must be prompt and clean: pause, stop, device switch, and Ctrl+C cancel active helper work without orphan recorder processes or leaked temp WAV files.
- The app core must consume only recorder-produced WAV files and `audio.Chunk` metadata. RMS, filtering, transcription, transcript ordering, and storage must not care which OS produced the WAV.
- User messages must not tell stock Windows users to install Stereo Mix/VB-CABLE as the primary path once WASAPI loopback is available. Those remain optional fallback/remediation tools only if the helper cannot use WASAPI.

## Proposed architecture seams to test

The exact implementation belongs to Livingston, but QA should push for these seams because they make the helper testable without real Windows audio hardware:

| Seam | Why QA needs it |
| --- | --- |
| Recorder command builder / process runner interface | Unit-test helper invocation, arguments/JSON payload, stderr mapping, and cancellation without launching a real executable. |
| Structured helper protocol fixture | Unit-test device discovery and capture result parsing from captured stdout/stderr examples. JSON lines are preferred over localized text. |
| Recorder executable path resolver | Unit-test bundled helper lookup, explicit override, missing-helper error text, and build packaging. |
| Clock/temp-dir/process cleanup hooks | Unit-test timeout, cancellation, and temp-file cleanup deterministically. |
| OS-neutral WAV contract | Reuse existing WAV/RMS/transcriber tests against helper-produced fixture files. |

## Automated unit coverage Livingston should add with the sidecar

### Recorder-sidecar invocation

- `TestWindowsOutputRecorderUsesWASAPISidecarNotDShowOutput`
  - Given a Windows output `ChunkRequest` for `Speakers (Realtek(R) Audio)`.
  - Expect the output recorder to invoke the sidecar executable/protocol, not `ffmpeg -f dshow -i audio=Speakers (...)`.
  - Expect no references to `Stereo Mix`, `VB-CABLE`, or `Voicemeeter` in the happy-path command or validation text.

- `TestWindowsMicRecorderRemainsSeparateFromWASAPIOutput`
  - If mic capture stays on ffmpeg/DirectShow, assert that only the output path goes through WASAPI sidecar.
  - If mic also moves behind a recorder executable, assert that mic/output are explicit modes in the same recording layer and do not leak into app/session/transcript packages.

- `TestRecorderSidecarPathResolution`
  - Bundled helper next to `transcribe.exe` is preferred.
  - Explicit config/env/flag override wins, if Livingston adds one.
  - Missing helper fails with a recorder-helper-specific error and a packaging hint.

### Helper protocol parsing

- `TestParseWindowsSidecarDeviceList`
  - Fixture includes a built-in Realtek microphone, built-in Realtek speakers, a USB headset, a Bluetooth speaker/headset, and default markers.
  - Expect microphone devices and output loopback devices to be distinct, stable, and selectable by ID and display name.
  - Current implementation provides `wasapi-loopback-recorder.exe list --json` with active render endpoints/default marker; the main app still supplements discovery from PowerShell render endpoints.

- `TestParseWindowsSidecarDefaultRenderLoopback`
  - Fixture exposes the current default render endpoint as an output loopback candidate.
  - Expect `--output default` / default config to resolve to a real WASAPI loopback target, not a display-only DirectShow render endpoint.

- `TestSidecarErrorMappingIsActionable`
  - Fixture for helper JSON error / exit code maps to a clear app warning/error.
  - Missing default render endpoint, access denied, unsupported format, and helper protocol version mismatch should each produce a distinct message.

### Cancellation and lifecycle

- `TestWindowsSidecarRecordChunkCancelsProcessOnContextCancel`
  - Fake process blocks until context cancellation.
  - Expect `RecordChunk` returns `context.Canceled`, the child process is killed, and no successful `Chunk` is emitted.

- `TestPauseStopAndDeviceSwitchCancelSidecarOutputChunk`
  - Extend the existing session-level cancellation pattern from `TestPauseCancelsActiveRecorderChunks` to assert sidecar-backed output capture observes cancellation for pause, stop, and output-device changes.

- `TestSidecarTimeoutDoesNotPoisonNextChunk`
  - First helper call times out/fails; the next call with the same selected device can still start cleanly.
  - Expect no stale temp file path or stale process handle is reused.

### OS-boundary guardrails

- `TestWindowsSpecificRecordingCodeIsIsolated`
  - A static/package-boundary test should fail if Windows WASAPI imports/types appear in app/session/transcript/filter/openai/TUI packages.
  - Allowed locations should be limited to recorder executable code and its narrow adapter package.

- `TestAudioFileProcessingAcceptsSidecarWAVFixture`
  - Feed a short PCM WAV produced by the sidecar fixture into the existing RMS/transcription pipeline seams.
  - Expect RMS parsing, filter decisions, transcript storage, and stdout/stderr silent-mode contracts to remain unchanged.
  - Current helper writes PCM16 16 kHz mono WAV output after native mix-format conversion, so existing RMS/OpenAI paths consume the same format as ffmpeg chunks.

### Build/package checks

- `TestWindowsBuildIncludesRecorderHelperManifest`
  - Build packaging must place the recorder helper beside `transcribe.exe` or in the documented helper directory.
  - `transcribe.cmd` stays present for double-click TTY behavior.

- `TestNonWindowsBuildsDoNotRequireWindowsHelper`
  - Linux/macOS builds and tests must not require the Windows helper executable.
  - Cross-compiles should still pass even if the helper itself is Windows-only; document if helper packaging is a separate Windows build artifact.

## Coverage Basher can validate now without helper internals

The following existing tests already exercise contracts that should survive the sidecar pivot:

- `internal/app/app_test.go::TestPauseCancelsActiveRecorderChunks` — session cancellation reaches both source recorders.
- `internal/app/app_test.go::TestRunSilentWritesOnlyFinalTranscriptToStdout` — audio-file processing output contract stays UI-free and OS-independent.
- `internal/audio/wav_test.go` — RMS parsing consumes WAV bytes, independent of capture backend.
- `internal/transcript`, `internal/filter`, and `internal/openai` tests — downstream processing should not gain Windows-specific behavior.

Basher should not add sidecar command/parser tests until Livingston exposes the helper API or protocol fixture; otherwise those tests would either lock in the wrong API or assert against unimplemented internals.

## Clean Windows manual validation matrix

Run these on a real Windows host after the sidecar lands. Use a clean user profile where possible. Capture `transcribe --list-devices`, `transcribe.log` with `--logging 1`, and the resulting transcript/WAV diagnostics for each case.

| Case | Setup | Steps | Expected result |
| --- | --- | --- | --- |
| Clean stock Windows / Realtek | Disable Stereo Mix. Do not install VB-CABLE/Voicemeeter. Use built-in Realtek speakers and built-in mic. | Play a known audio sample through speakers. Run `transcribe --list-devices`, then record with default mic/output. | Output capture is available via WASAPI loopback. No primary remediation tells user to install Stereo Mix/VB-CABLE. Transcript includes microphone and system-output chunks when audio is audible. |
| Audacity-equivalent WASAPI loopback | Same as stock setup. Audacity would show Windows WASAPI loopback for the render device. | Compare a short known audio sample captured by Audacity loopback and by `transcribe` sidecar. | `transcribe` captures the same render endpoint class Audacity can capture; levels are non-zero during playback and near-zero during silence. |
| Built-in Realtek speakers | Realtek speakers selected as Windows default output. | Start recording, play audio, stop. | Output chunks are valid WAV files and show non-zero RMS while playback is active. |
| USB headset/speaker | Plug in USB audio device and make it default output. | List devices, select USB output, record playback. | USB render endpoint appears as output loopback candidate and captures playback. |
| Bluetooth speaker/headset | Pair Bluetooth device; make it default output. | List devices, select Bluetooth output, record playback. | Bluetooth render endpoint appears as output loopback candidate. Capture works or fails with a helper-specific limitation message, not a DirectShow/Stereo-Mix message. |
| Default-device switch | Start with Realtek default output, then switch Windows default to USB/Bluetooth during the session. | Record with default-output selection. Switch default while playback continues. | If selection is default-following, the next chunk follows the new default. If selection is concrete, capture stays on the selected endpoint. The chosen contract must be documented and tested. |
| Output-device switch in TUI | Start recording, open settings, switch output device. | Switch Realtek → USB/Bluetooth and back. | Active helper chunk is cancelled; next output chunk uses the new device; mic continues. |
| Pause/resume | Start recording while playback runs. Press `P`, wait, press `R`. | Inspect logs and transcript timing. | Helper is cancelled on pause, no orphan process remains, resume starts a fresh helper capture. |
| Ctrl+C / stop | Start recording with playback. Press Ctrl+C / `Q`. | Inspect process list/temp dir/log. | Helper exits promptly; temp WAVs are cleaned or finalized; transcript file is flushed. |
| No playback silence | Start recording with no system audio playback. | Record a short session. | Helper produces valid silent/near-silent WAV chunks or intentionally suppresses them per contract; it must not report device failure just because audio is silent. |

## Regression watch list

- Old DirectShow-only assertions have been narrowed to DirectShow fallback behavior. WASAPI sidecar endpoints are runtime-capable and may appear in the fallback chain; synthesized DirectShow render endpoints remain display-only and must not become ffmpeg targets.
- `README.md` Windows requirements now state that WASAPI sidecar captures normal render endpoints; virtual loopback tools are fallback/diagnostic options only.
- Build artifacts under `transcribe/build/windows-amd64/.venv/` are stale historical leftovers; helper packaging must not accidentally revive the old Python venv path unless that is the explicit chosen sidecar technology.

## Open questions for Livingston

1. Is the recorder helper a Go executable, a small native Windows executable, or a Python/PyAudioWPatch executable? QA only needs a stable process/protocol contract, but packaging tests depend on the answer.
2. Does `default` output mean "resolve once at session start" or "follow Windows default render endpoint per chunk"?
3. Should helper output failure be non-fatal mic-only continuation, fatal session error, or configurable? Current synthesized-only DirectShow behavior is non-fatal mic-only continuation.
4. What helper protocol/version field should tests pin so an old helper beside a new `transcribe.exe` fails clearly?
5. Where exactly is Windows-specific recorder code allowed to live? QA recommends a narrow recorder package plus `cmd/<helper>` only.
