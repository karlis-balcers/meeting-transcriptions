# Go TUI QA and Validation Strategy

**By:** Basher
**Date:** 2026-05-27T15:57:08Z
**Scope:** QA strategy only. No implementation in this pass.

## Quality position

The Go TUI rebuild should be test-first around ports/adapters. CI must not need a real microphone, loopback device, Teams window, OpenAI credential, terminal, or user keystrokes. Anything external gets faked; anything deterministic gets unit tested; workflow behavior gets driven through the CLI/TUI model.

## Testable units

### Config loading and validation

- Parse env/config/flags into one immutable runtime config.
- Preserve current defaults and warnings for:
  - `YOUR_NAME`, `LANGUAGE`, `AUTO_START_TRANSCRIPTION`, `AUTO_SUMMARIZE_ON_STOP`
  - OpenAI models, vector store, web search toggle, live assistant panel toggles
  - audio segmentation: `RECORD_SECONDS`, `SILENCE_THRESHOLD`, `SILENCE_DURATION`, `FRAME_DURATION_MS`
  - output paths: `OUTPUT_DIR`, `TEMP_DIR`, `SUMMARIES_DIR`, `CONTEXT_FILE`
  - API retry/timeout settings
  - transcript filter overrides
- Cover invalid values: bad bools, negative durations, zero frame duration, unsupported languages, duplicate language codes, empty output dirs.
- Assert precedence once CLI exists: flags > config file > environment > defaults.
- Assert `env.sample` and config schema stay in sync.

### Device discovery and selection

- Normalize device names and match loopback variants.
- Split microphone vs output-capture devices by platform:
  - Windows: loopback input devices are output capture; non-loopback input devices are microphones.
  - Linux/macOS: inputs and outputs are selected from their channel capabilities.
- Preferred selection order: saved index, saved name, default device, first available.
- Edge cases: no mic, no output capture, stale saved index/name, duplicate names, device enumeration errors.

### Audio segmentation

- Silence threshold/duration split behavior.
- Max recording duration split behavior.
- Manual split behavior from keyboard shortcut/input message.
- Speaker-change flush behavior, including the current 2-second output rollback concept.
- Mic mute behavior: mic frames are skipped while output capture can continue.
- Stop behavior: buffered frames are flushed exactly once.
- Overflow/read errors surface status and stop the stream cleanly.

### WAV/temp-file handling

- Creates temp directory if missing.
- Writes correct channel count, sample width, and sample rate.
- Invokes transcription callback with the expected source (`mic`/`output`) and speaker token.
- Removes temp WAV after transcription success or handled failure.
- Keeps errors observable without crashing the workflow.

### Transcription/OpenAI adapter

- Constructs transcription request with model, language, and keyword prompt.
- Retries transient failures with bounded exponential backoff.
- Does not retry auth/permission failures.
- Suppresses fake-keyword/control-artifact responses.
- Handles missing text and malformed responses.
- Uses fake client/`httptest`; no live OpenAI calls in unit or CI tests.

### Transcript filtering

- Preserve exact/prefix/contains/regex artifact filters.
- Preserve keyword bypass.
- Preserve short-noise heuristic.
- Cover invalid regex config as a Go-specific robustness case.

### Transcript store and output persistence

- Thread-safe add/snapshot ordering by start time.
- Normalize escaped newlines into real newlines.
- Do not append assistant/agent messages to transcript files.
- Generate a fresh transcript file on every Start.
- Clear in-memory transcript view on Start.
- Keep Markdown header format stable.
- Surface file-write errors in status/logs.

### Speaker detection

- Parse Teams control dumps into speaker names.
- Normalize Teams window title into meeting title.
- Non-Windows detector is a no-op stub that compiles and returns no speaker.
- Windows detector is isolated behind build tags and can be tested with supplied control-dump fixtures.

### Assistant/summary seams

- Fake assistant client for custom prompts, live panels, and summaries.
- Verify transcript passed to summary contains ordered `speaker: text` lines.
- Verify summary file naming sanitizes generated titles.
- Verify auto-summary on stop only runs when enabled and transcript has content.

### TUI model/state

- State transitions: ready → recording → stopping → ready; ready/error paths for failed device init.
- Start is idempotent while already recording.
- Stop is idempotent while already stopped.
- Settings/config controls are disabled or rejected while recording.
- Prompt entry only accepts input when recording and assistant is available.
- Status line severity maps to stable state/color tokens.
- Live assistant panels render waiting, success, empty (`---`), and failure states.

## Required integration seams

Build these ports before wiring real implementations:

- `DeviceDiscoverer`: enumerate devices and provide platform defaults.
- `AudioRecorder` / `FrameSource`: start, stop, and emit deterministic audio frames.
- `Segmenter`: convert frames and split events into recording chunks.
- `Transcriber`: convert audio chunks into transcript segments.
- `AssistantClient`: custom prompt, live answer modes, summary generation.
- `SpeakerDetector`: current speaker and meeting title snapshots.
- `TranscriptSink`: in-memory store plus file append behavior.
- `FileSystem`: temp dirs, transcript files, summary files.
- `Clock`: timestamps for filenames and transcript ordering.
- `StdIO`: stdout/stderr writers for CLI and silent-mode assertions.
- `TUIProgram`: injectable model driver/message loop for keyboard/state tests.
- `Logger`: separate user-visible output from logs.

Hard rule: core workflow packages should not call real audio APIs, OpenAI, `os.Stdout`, `time.Now`, or `os.Getenv` directly. Keep those in adapters.

## Workflow/integration tests

### Happy path with fakes

Drive fake mic/output frames through the segmenter, fake transcriber, transcript store, and TUI model. Assert:

- Start creates a new transcript file.
- Mic output is labeled as the configured user.
- Output-device transcription is labeled with detected speaker or fallback person token.
- Transcript file contains only non-agent transcript lines.
- Stop flushes buffered frames and returns to ready.

### Start/stop session isolation

- Start, receive transcript, stop.
- Start again.
- Assert a second transcript file is created, memory is cleared, stale queue messages are ignored, and assistant thread/context is reset.

### Device failure paths

- No devices: start fails, no recording goroutines remain, clear error status.
- Mic only/output only: start fails unless product explicitly chooses degraded mode.
- Device disappears during recording: stream exits, status is warning/error, stop remains safe.

### API failure paths

- Transient OpenAI failures retry then succeed.
- Retry exhaustion leaves no transcript segment and emits error status.
- Auth failure exits the adapter path without retry.
- Assistant failures do not break recording/transcription.

### Auto-start and auto-summary

- `AUTO_START_TRANSCRIPTION=True` starts with first configured language and no prompt.
- Auto-summary on stop runs only when enabled, assistant exists, and transcript snapshot is non-empty.

## CLI behavior tests

Use process-level tests for the built binary plus package-level tests for command parsing.

Minimum cases:

- `--help` exits 0 and lists supported flags/shortcuts.
- `--version` exits 0 and prints version only.
- `--list-devices` uses fake device discovery and prints stable, parseable device rows.
- Missing required config exits non-zero with actionable stderr and no panic.
- Invalid config path exits non-zero with actionable stderr.
- `--config <file>` loads only the requested file and does not read `.env` in tests.
- `--output-dir`, `--temp-dir`, and `--language` override config.
- `--no-assistant` or equivalent disables assistant calls while allowing transcription.
- Non-interactive/`--once` style mode, if implemented, can process fixture audio and exit deterministically.

Keep all command tests isolated with temp dirs, `t.Setenv`, fake adapters, and captured stdout/stderr.

## Silent-mode stdout behavior

Silent mode needs exact-output tests. No excuses.

Expected contract:

- TUI disabled.
- No progress bars, ANSI escapes, status lines, timestamps, logs, or warnings on stdout.
- Logs/status go to stderr or log files only.
- Transcript and summary files are still written.
- On success, stdout is exactly empty unless a documented machine-readable output flag is used.
- On failure, stdout is exactly empty and stderr contains the actionable error.
- `NO_COLOR=1` and non-TTY output disable color in stderr/log console output.

Tests should capture `bytes.Buffer` stdout/stderr in package tests and run the binary for at least one process-level regression.

## Transcript file output checks

Golden-file assertions should cover:

- Filename pattern: `transcription-YYYYMMDD_HHMMSS.md` using fake clock.
- Header:
  - `# Transcription Log`
  - `**Created:** YYYY-MM-DD HH:MM:SS`
- Entry format: `Speaker: text` followed by a blank line.
- Entries are ordered by start timestamp, not arrival order.
- Agent/assistant messages are excluded.
- Escaped newline text is decoded before writing.
- A fresh file is created on each Start.
- Empty session still has the header only.
- Output directory is created if absent.

## Keyboard shortcut and TUI state tests

If using Bubble Tea or a similar Elm-style TUI, test the model `Update` function directly.

Required shortcut/state cases:

- Start/stop shortcut toggles recording.
- Quit shortcut while stopped exits cleanly.
- Quit while recording triggers stop/flush path before exit.
- Mute shortcut toggles mic mute and updates state.
- Manual split shortcut emits a split event only when not typing in prompt input.
- Prompt text input consumes normal character keys and Enter sends prompt.
- Escape cancels prompt/settings overlays without stopping recording.
- Device list navigation changes selection only when stopped.
- Help overlay lists the same shortcuts implemented by the model.
- Window resize keeps transcript viewport and panels consistent.
- Error state can be acknowledged/retried without restarting the program.

Use model-level event sequences for fast unit tests and a small set of golden render snapshots for regressions.

## Cross-platform build checks

CI matrix should cover Linux, macOS, and Windows.

Required gates:

- `go test ./...` on all three OSes.
- `go test -race ./...` at least on Linux.
- `go vet ./...` on all OSes.
- Cross-build smoke checks for:
  - `GOOS=linux GOARCH=amd64`
  - `GOOS=darwin GOARCH=amd64`
  - `GOOS=darwin GOARCH=arm64`
  - `GOOS=windows GOARCH=amd64`
- Platform adapters compile behind build tags:
  - Windows-only audio/Teams implementations.
  - Non-Windows stubs for loopback and Teams detection.
- CI fails if unit tests require audio hardware, GUI terminal support, or OpenAI credentials.

## Test priority

### P0 before first usable Go TUI build

- Config parser and precedence tests.
- Fakeable ports for audio, OpenAI, device discovery, filesystem, clock, stdout/stderr.
- Transcript store/file golden tests.
- Silent-mode exact stdout tests.
- TUI start/stop state tests.
- Cross-platform compile checks with non-Windows stubs.

### P1 before feature parity

- Full fake happy-path workflow.
- Device failure workflows.
- OpenAI retry/error workflows.
- Keyboard shortcut focus/overlay tests.
- Auto-start and auto-summary workflows.

### P2 hardening

- Race tests around transcript store and workflow goroutines.
- Golden TUI render snapshots.
- Larger fake audio segmentation scenarios.
- Windows Teams parser fixture suite.

## Exit criteria

The Go TUI rebuild is not QA-ready until it passes deterministic unit tests, fake integration workflows, exact silent-mode stdout checks, transcript golden tests, TUI state/shortcut tests, and cross-platform build gates without real hardware or live network access.
