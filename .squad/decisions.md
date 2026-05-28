# Squad Decisions

## Active Decisions

### 2026-05-27: Go TUI migration design
**By:** Danny
**What:** Rebuild the Python window transcription app as a focused Go TUI/CLI under `transcribe/`, producing an executable named `transcribe`. Keep live mic + output transcription, device selection, Teams best-effort speaker detection, transcript persistence, filtering, OpenAI transcription retries, pause/resume/mute, and loudness feedback. Drop Tkinter, assistant panels, custom prompts, summaries, vector stores, local Whisper/faster-whisper, `.env` app config, context files, and agent features.
**Why:** The requested product is a single-purpose recorder/transcriber with terminal UX and pipe-friendly silent mode. Keeping assistant/summary code would increase migration risk and conflict with the requirement to support only OpenAI transcription and no assistant/agents.

#### Go module layout
- `transcribe/go.mod` — standalone module for the new app.
- `transcribe/cmd/transcribe/main.go` — CLI parsing, config load, mode selection, top-level exit codes.
- `transcribe/internal/app` — lifecycle orchestration, signal handling, state machine.
- `transcribe/internal/config` — `~/.transcribe/config.yaml` load/save, validation, defaults; never stores API keys.
- `transcribe/internal/audio` — cross-platform device enumeration, default/preferred device selection, capture, segmentation, loudness samples.
- `transcribe/internal/openai` — multipart transcription client, retry/backoff, timeout handling.
- `transcribe/internal/transcript` — ordered aggregation, Markdown persistence, silent-mode final stdout rendering.
- `transcribe/internal/filter` — carry forward artifact filtering and keyword-preserve behavior.
- `transcribe/internal/speaker` — `speaker_windows.go` best-effort Teams UIA integration; non-Windows stubs warn once and return `Person_?`.
- `transcribe/internal/tui` — Bubble Tea TUI, shortcuts, meters, device/settings overlays.

#### CLI contract
- `transcribe` starts recording/transcribing immediately after validation.
- `-s`, `--silent` disables TUI, records until interrupt, writes only final complete transcript to stdout; all status/errors go to stderr.
- `-o`, `--output-dir <dir>` overrides configured transcript directory for this run.
- `--config <path>` optional override for advanced/testing use; default remains `~/.transcribe/config.yaml`.
- `--list-devices` prints detected mic/output candidates and exits.
- Optional run overrides: `--mic <id-or-name>`, `--output <id-or-name>`, `--language <code>`, `--model <name>`; persisted only through runtime settings, not CLI flags.

#### Config schema
- Path: `~/.transcribe/config.yaml`; create parent directory with user-only permissions where supported.
- API key: `OPENAI_API_KEY` environment variable only. Do not read/write API keys in YAML or `.env`.
- Suggested keys:
	- `user_name: "You"`
	- `language: "en"`
	- `keywords: []`
	- `openai.model: "gpt-4o-mini-transcribe"`, `openai.timeout_seconds: 60`, `openai.max_retries: 3`, `openai.retry_base_seconds: 1`
	- `audio.mic_device_id/name`, `audio.output_device_id/name`, `audio.frame_duration_ms: 100`, `audio.silence_threshold: 50`, `audio.silence_duration: "1s"`, `audio.max_segment_duration: "300s"`
	- `paths.output_dir: ""` meaning current working directory, `paths.temp_dir: ""` meaning OS temp/cache location.
	- `teams.enabled: true`, `tui.show_loudness_meters: true`
	- `filter.min_chars`, `filter.exact/prefixes/contains/regex`

#### Audio/transcription flow
1. Validate `OPENAI_API_KEY`, config, output directory, temp directory, and audio devices before starting capture; fail fast with actionable errors.
2. Select configured devices if present and healthy; otherwise use OS default microphone and default output-capture device. Show both selected devices at TUI start and on stderr in silent mode.
3. Start mic and output capture goroutines. Output capture is required unless the user explicitly selects a valid output-capture device; if unsupported or unavailable, fail with clear platform guidance rather than silently recording only mic.
4. Segment by silence and max segment duration, write temporary WAV chunks under temp storage, send to OpenAI transcription, delete chunks after successful/failed processing.
5. Aggregate segments by capture timestamp. Mic segments use `user_name`; output segments use Teams speaker when available, otherwise `Person_?`.
6. Preserve transcript artifact filtering and keyword prompts from the Python implementation.
7. Pause (`P`) stops capture ingestion without losing already queued transcription work; resume (`R`) restarts capture; mute (`M`) suppresses mic stream only; unmute (`U`) resumes mic stream.

#### TUI states and shortcuts
- States: `Starting`, `Recording`, `Paused`, `Settings`, `DeviceSelectMic`, `DeviceSelectOutput`, `Stopping`, `Error`.
- Bottom shortcut bar: `P Pause`, `R Resume`, `M Mute mic`, `U Unmute mic`, `S Settings`, `Q Quit`.
- `S` opens settings overlay; then `M` selects microphone, `O` selects output device. Selection persists to YAML and restarts the affected stream if recording.
- Main screen shows selected devices, session timer, status/errors, last transcript entries, Teams speaker status, and horizontal loudness meters for mic and output.
- Professional TUI defaults: no stdout noise outside silent mode transcript output, resize-aware layout, accessible contrast, clear degraded-mode warnings, graceful terminal restore on panic/signals.

#### File output strategy
- Default transcript directory: `paths.output_dir` if set; otherwise current working directory; CLI `--output-dir` wins for the run.
- Filename: `transcription-YYYYMMDD_HHMMSS.md`; include created timestamp, selected devices, language, and model metadata.
- Append accepted segments incrementally in chronological order; keep in-memory ordered transcript for TUI and silent final stdout.
- Silent mode still saves the Markdown transcript by default and prints the complete final transcript to stdout only after interrupt/drain.

#### Platform risks
- Output/system audio capture is the highest risk. Windows WASAPI loopback is feasible; macOS/Linux may need OS permissions, PulseAudio/PipeWire monitor sources, BlackHole/Soundflower-style virtual devices, or equivalent. Treat missing output capture as a clear startup error.
- Go audio library choice should favor a cross-platform capture stack with loopback support where possible; validate build requirements early because CGo can complicate Windows/Linux/macOS ARM+Intel releases.
- Teams speaker recognition is Windows-only and fragile. Implement behind build tags and warn/fallback cleanly everywhere else.
- OpenAI file limits require bounded chunks; keep max segment duration conservative and surface API retry status in the TUI.
- macOS microphone/screen/audio permissions and terminal raw mode restoration need explicit test coverage.

#### Implementation/testing responsibilities
- Lead Engineer: own migration architecture, package boundaries, acceptance criteria, and cut/drop decisions.
- Audio/platform implementer: device enumeration, default selection, loopback capture, loudness meters, pause/mute semantics, cross-platform build matrix.
- TUI implementer: Bubble Tea state model, shortcuts, settings/device overlays, silent mode behavior, terminal safety.
- API/transcript implementer: OpenAI transcription client, chunk lifecycle, filtering, ordered persistence, stdout contract.
- Windows/platform specialist: Teams UIA speaker detection and WASAPI validation.
- QA/release: unit tests for config/device selection/filtering/aggregation, integration smoke tests with fake audio/OpenAI client, manual OS matrix checks, release build scripts for Windows, Linux, macOS ARM, and macOS Intel.

### 2026-05-27: Python behavior map decisions for Go core
**By:** Rusty
**What:** Preserve the core capture -> chunk -> transcribe -> transcript-store semantics for the Go TUI rebuild, but do not line-copy Tkinter/global-key/empty-file side effects. The Go core should create transcript files only when a session starts, allow mic-only operation when output capture is unavailable, expose speaker detection as an optional event source, and replace any-letter manual split behavior with explicit split/speaker controls.
**Why:** These choices keep user-visible transcript behavior while removing brittle GUI, PyAudio, and Windows UI Automation coupling from the core. They also fix shutdown/drain risks in the Python thread model without changing the intended session semantics.

### 2026-05-27: Go TUI/CLI stack for transcription rebuild
**By:** Livingston
**What:** Use a Bubble Tea-based TUI with Cobra commands, typed YAML config at `~/.transcribe/config.yaml`, an OpenAI-only transcription adapter, and platform-isolated audio/speaker packages behind build tags. Prefer native release-matrix builds for audio-enabled binaries instead of promising frictionless single-host cross-compilation.
**Why:** CLI/TUI, config, silent stdout mode, and OpenAI transcription are straightforward in Go, but default mic plus system-output capture is inherently platform-specific. Windows should use WASAPI loopback, Linux should use PulseAudio/PipeWire monitor sources, and macOS should start with virtual loopback-device support or a native ScreenCaptureKit/CoreAudio helper path before claiming full built-in system-output capture.

Recommended libraries:
- TUI: `github.com/charmbracelet/bubbletea`, `github.com/charmbracelet/bubbles`, `github.com/charmbracelet/lipgloss`
- CLI/config: `github.com/spf13/cobra`, typed config loaded with `gopkg.in/yaml.v3` or `github.com/goccy/go-yaml`; keep Viper optional rather than central
- OpenAI transcription: `github.com/openai/openai-go` behind an internal interface; use a thin standard-library multipart fallback if the audio API surface is insufficient
- Audio/WAV: platform-specific capture under `internal/audio` plus `github.com/go-audio/wav` for PCM WAV chunk writing
- Windows audio/speaker: WASAPI loopback and Windows UI Automation behind `//go:build windows`
- Linux audio: PulseAudio/PipeWire monitor source capture, preferably through the Pulse protocol/client path
- macOS audio: microphone capture plus detected virtual loopback devices for v1; evaluate ScreenCaptureKit/CoreAudio process tap helper for later built-in system audio

Suggested module layout:
- `cmd/transcribe/` for the binary entry point
- `internal/cli/` for Cobra commands and silent-mode routing
- `internal/tui/` for Bubble Tea models/views/update loop
- `internal/config/` for defaults, validation, atomic saves, and `~/.transcribe/config.yaml`
- `internal/app/` for orchestration/session lifecycle
- `internal/audio/` for capture interfaces, segmenting, WAV chunking, and platform build-tag implementations
- `internal/transcribe/openai/` for the OpenAI transcription client
- `internal/transcript/` for transcript events, ordering, Markdown/stdout sinks
- `internal/speaker/` for Teams best-effort detection plus no-op fallbacks
- `internal/logging/` for stderr/file logging that never pollutes silent stdout

### 2026-05-27: Go TUI tests require dependency-injected external seams
**By:** Basher
**What:** The Go TUI rebuild must keep audio capture, OpenAI/transcription, device discovery, Teams speaker detection, filesystem/clock/stdout, and TUI driver behavior behind injectable interfaces from the first implementation.
**Why:** Regression and workflow tests need deterministic fakes across Linux, macOS, and Windows. CI must validate silent-mode stdout, transcript files, keyboard/TUI state, and failure paths without real audio hardware, Teams, OpenAI credentials, or live network access.

Full QA strategy artifact: `.squad/agents/basher/go-tui-qa-validation-strategy.md`.

### 2026-05-27: Go TUI implementation boundaries
**By:** Livingston
**What:** Implemented the new `transcribe/` app as a standalone pure-Go module with Cobra CLI, Bubble Tea TUI, YAML config, ffmpeg external recording backend, PulseAudio/PipeWire `pactl` discovery on Linux, manual/default fallback devices on Windows/macOS, OpenAI multipart transcription via `net/http`, ordered Markdown transcript persistence, and non-Windows Teams speaker fallback to `Person_?`.
**Why:** This satisfies the requested shippable CLI/TUI contract while avoiding cgo/native audio bindings that would break cross-platform builds. Real loopback capture remains delegated to OS audio setup plus `ffmpeg`, so unsupported/missing devices fail before recording with actionable guidance instead of pretending capture works.

### 2026-05-27: Go TUI QA review rejected pending live workflow fixes
**By:** Basher
**What:** Reject the current Go TUI implementation as production-ready despite passing tests, vet, native build, and linux/darwin/windows amd64/arm64 cross-compile smoke checks. I added tests for silent-mode stdout, API-key fail-fast behavior, and TUI shortcuts/settings, but remaining runtime gaps need another owner.
**Why:** The ffmpeg chunk recorder runs bounded chunks up to the configured max duration, so pause, mute, device changes, and loudness feedback are delayed/fake rather than live workflow controls. Windows Teams detection is also explicitly stubbed instead of best-effort UI Automation. Revision should be owned by a new audio/platform specialist or Danny, not Livingston, per reviewer protocol.

### 2026-05-27: Go TUI revision addresses live-control rejection
**By:** Danny
**What:** Revised the Go `transcribe` implementation to use configurable short ffmpeg capture chunks (`audio.capture_chunk_duration`, default `2s`), cancel active per-source recorder contexts on pause/mute/device changes, persist settings-driven device changes before restarting the affected stream, compute TUI loudness from captured WAV/PCM RMS levels, and replace the Windows Teams stub with best-effort PowerShell window-title polling behind the Windows build tag.
**Why:** Basher's rejection was correct: long max-duration chunks delayed runtime controls, fake meters hid capture reality, and an explicit Windows Teams stub failed the migration contract. This keeps the pure-Go/cross-compile boundary while making controls responsive and telemetry real enough for the TUI revision.

### 2026-05-27: Go TUI re-review approved
**By:** Basher
**What:** APPROVED Danny's Go TUI revision after re-reviewing the original rejection points and the broader CLI/TUI/silent/config/OpenAI/transcript contract. Runtime controls now cancel active recorder contexts and use short configurable capture chunks, loudness meters are computed from captured WAV/PCM RMS levels, Windows Teams detection makes a genuine best-effort title-polling attempt, and non-Windows fallback is explicit.
**Why:** Validation passed on Linux amd64 with Go 1.26.3: `go test ./...`, `go vet ./...`, native build, cross-compiles for linux/darwin/windows amd64/arm64, `go test -race ./...`, formatting/module/whitespace/final-newline hygiene, plus CLI smoke checks for help/version/list-devices and missing API-key stderr/no-stdout behavior.

### 2026-05-27: Windows DirectShow devices must be concrete enumerated choices
**By:** Livingston
**What:** Windows audio discovery now parses ffmpeg DirectShow device-list output, stores DirectShow alternative names as aliases/selectable IDs, suppresses old synthetic `default`/`virtual-audio-capturer` placeholders unless actually enumerated, and exposes all DirectShow audio capture devices in both microphone and output-capture settings with clear output-candidate labeling.
**Why:** Passing `audio=default` to DirectShow failed on real Windows hosts and hid most available microphones. ffmpeg can list concrete DirectShow audio devices without cgo, preserving pure-Go cross-compilation while giving users actionable names/IDs and documenting that DirectShow alone cannot reliably classify system-output loopbacks.

### 2026-05-27: Windows DirectShow audio fix review approved
**By:** Basher
**What:** APPROVED Livingston's Windows audio discovery fix after code review and validation. The Go `transcribe/` app now parses ffmpeg DirectShow audio sections from stderr, preserves alternative-name aliases, avoids treating video devices as audio, exposes every parsed DirectShow audio candidate for both microphone and output-capture settings, suppresses unresolved synthetic `default`/old placeholder selections when concrete devices exist, and provides list-devices guidance when defaults cannot resolve.
**Why:** Regression coverage now pins the reported failure mode: multiple DirectShow audio devices and aliases are parsed, formatted device lists expose aliases instead of synthetic `audio=default`, default preferences fall back to concrete enumerated devices with actionable warnings, and recorder validation rejects unresolved DirectShow defaults before capture. Validation passed from `transcribe/`: `go test ./...`, `go vet ./...`, native `go build -o transcribe ./cmd/transcribe` with binary removal, linux/darwin/windows amd64+arm64 cross-compiles, and `go test -race ./...`.

## Governance

- All meaningful changes require team consensus
- Document architectural decisions here
- Keep history focused on work, decisions focused on direction
