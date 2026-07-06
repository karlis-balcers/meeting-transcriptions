# transcribe

`transcribe` is a focused Go TUI/CLI recorder for meeting transcription. It records a microphone plus a system-output capture device, sends WAV chunks to OpenAI audio transcription, and writes a Markdown transcript.

This is intentionally not a port of the old assistant/summarizer stack: no summaries, no assistant panels, no vector stores, no agents, and no local Whisper.

## Requirements

- Go 1.22+
- `ffmpeg` available on `PATH`
- `OPENAI_API_KEY` set in the environment
- A selectable microphone device
- A selectable system-output/loopback capture device
- Windows: `ffmpeg` on PATH for microphone/DirectShow fallback capture, plus the bundled `wasapi-loopback-recorder.exe` sidecar beside `transcribe.exe` for built-in speaker/headphone capture via WASAPI loopback. DirectShow loopback tools such as Stereo Mix, VB-CABLE, Voicemeeter, or `virtual-audio-capturer` remain optional fallback/diagnostic devices.

The OpenAI API key is never read from or written to the YAML config file.

## Build

Run build and test commands from the repository root:

```sh
go test ./...
go build -o transcribe ./cmd/transcribe
```

Cross-compile smoke checks:

```sh
GOOS=linux GOARCH=amd64 go build -o /tmp/transcribe-linux-amd64 ./cmd/transcribe
GOOS=linux GOARCH=arm64 go build -o /tmp/transcribe-linux-arm64 ./cmd/transcribe
GOOS=darwin GOARCH=amd64 go build -o /tmp/transcribe-darwin-amd64 ./cmd/transcribe
GOOS=darwin GOARCH=arm64 go build -o /tmp/transcribe-darwin-arm64 ./cmd/transcribe
GOOS=windows GOARCH=amd64 go build -o /tmp/transcribe-windows-amd64.exe ./cmd/transcribe
GOOS=windows GOARCH=arm64 go build -o /tmp/transcribe-windows-arm64.exe ./cmd/transcribe
```

On Windows, run `build_transcribe_win64.bat` from the repository root. It cross-compiles the pure-Go main binary via WSL, builds the sidecar from `./cmd/wasapi-loopback-recorder`, and emits `transcribe.exe`, `wasapi-loopback-recorder.exe`, and a `transcribe.cmd` launcher to the root-level `build/windows-amd64/` package directory. The launcher keeps a persistent `cmd.exe` attached so the Bubble Tea TUI has a real TTY when launched by double-click.

## Project layout

- `cmd/transcribe` — main CLI/TUI entry point and Windows console-launch glue.
- `cmd/wasapi-loopback-recorder` — Windows WASAPI loopback sidecar executable for built-in speaker/headphone capture.
- `internal/app` — session orchestration, recorder/transcriber wiring, and runtime lifecycle.
- `internal/audio` — device discovery, recorder backends, WAV/RMS helpers, and ffmpeg/WASAPI dispatch.
- `internal/cli` — Cobra command setup, config loading, logging flags, and command execution.
- `internal/config` — YAML config model, defaults, environment compatibility, and validation.
- `internal/openai` — OpenAI audio transcription client.
- `internal/speaker` — best-effort speaker naming helpers, including Windows Teams title probing.
- `internal/transcript` — ordered Markdown transcript storage.
- `internal/tui` — Bubble Tea model and interaction flow.
- `docs` — cross-platform QA and Windows WASAPI sidecar test plans.
- `build/windows-amd64` — generated Windows package artifacts from `build_transcribe_win64.bat`.

## Configuration

Default config path:

```text
~/.transcribe/config.yaml
```

Use `--config <path>` to override it for tests or alternate profiles.

Example:

```yaml
user_name: "You"
language: "en"
keywords:
  - "Paymentology"
  - "Banking.Live"

openai:
  model: "gpt-4o-mini-transcribe"
  timeout: "60s"
  max_retries: 3
  retry_base: "1s"
  retry_max_interval: "8s"

audio:
  mic_device_id: ""
  mic_device_name: ""
  output_device_id: "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
  output_device_name: ""
  capture_chunk_duration: "2s"
  frame_duration_ms: 100
  silence_threshold: 50
  silence_duration: "1s"
  max_segment_duration: "300s"

paths:
  output_dir: ""
  temp_dir: ""

teams:
  enabled: true

tui:
  show_loudness_meters: true

filter:
  min_chars: 2
  exact: ["LAMPA", "MEMMEE"]
  prefixes: ["[Music]", "(Music)", "♪"]
  contains: ["thank you for watching"]
  regex: []
```

If `paths.output_dir` is empty, transcripts are written to the current working directory. `--output-dir <dir>` overrides it for one run.

## Environment

Required:

```sh
export OPENAI_API_KEY="<your key>"
```

Optional compatibility environment variables are also accepted for non-secret settings, including `YOUR_NAME`, `LANGUAGE`, `KEYWORDS`, `OPENAI_MODEL_FOR_TRANSCRIPT`, `OUTPUT_DIR`, `TEMP_DIR`, `CAPTURE_CHUNK_DURATION`, `FRAME_DURATION_MS`, `SILENCE_THRESHOLD`, `SILENCE_DURATION`, `RECORD_SECONDS`, and transcription retry settings.

Config file values override those compatibility environment values. CLI flags override both.

## Running

Start immediately with the TUI:

```sh
./transcribe
```

Enable runtime logs in the current directory:

```sh
./transcribe --logging 1
```

When enabled, the app writes `transcribe.log` beside the directory you launched it from and records device selection, chunk capture, and transcription progress there.

List detected devices:

```sh
./transcribe --list-devices
```

Override devices for one run:

```sh
./transcribe --mic default --output alsa_output.pci-0000_00_1f.3.analog-stereo.monitor
```

On Windows, prefer a concrete device shown by `--list-devices` instead of `default`. DirectShow device IDs may be long `@device_...` alternative names; those are safe to paste into `--mic`, `--output`, `audio.mic_device_id`, or `audio.output_device_id`.

## TUI shortcuts

- `P` pause
- `R` resume
- `M` mute microphone
- `U` unmute microphone
- `S` settings
- In settings: `M` choose microphone, `O` choose output capture device
- Device list: `↑`/`↓` or `k`/`j`, `Enter` selects, `Esc` backs out
- `Q` or `Ctrl+C` quits safely

The TUI shows selected devices, current status/error, session timer, transcript viewport, degraded warnings, and horizontal loudness meters.

After choosing a device in the picker and pressing `Enter`, the TUI returns to the settings screen and shows the updated mic/output selection summary.

Runtime controls are applied between short external-recorder chunks and also cancel the active recorder context when possible. `capture_chunk_duration` controls that responsiveness for the ffmpeg backend and is capped by `max_segment_duration`; the default is `2s` so pause, mute, and device changes do not wait for a long transcription segment. Loudness meters are computed from captured WAV/PCM chunks by parsing sample RMS levels, so unsupported or malformed chunks show a warning rather than fake meter data.

## Silent mode

Silent mode disables the TUI and records until interrupted:

```sh
./transcribe --silent > transcript.txt
```

Contract:

- final complete transcript goes to stdout only after interrupt and drain
- status, selected devices, warnings, and errors go to stderr
- no ANSI UI, progress, logs, or diagnostics are written to stdout
- Markdown transcript files are still written

## OS audio notes

### Linux

Device discovery parses `pactl info` and `pactl list short sources` when PulseAudio/PipeWire is available. System output capture expects a monitor source such as:

```text
alsa_output.<device>.monitor
```

If no monitor source exists, create/enable one in your audio stack or set `audio.output_device_id` to a valid ffmpeg/Pulse input.

The numeric index column from `pactl list short sources` is preserved as a selectable alias, so `--mic`, `--output`, `AUDIO_INPUT_DEVICE_INDEX`, and `AUDIO_OUTPUT_DEVICE_INDEX` can target the exact listed row in multi-device setups.

### Windows

The main app is pure Go and cross-compiles. Mic capture still uses an ffmpeg external recorder with DirectShow device names from:

```sh
ffmpeg -hide_banner -list_devices true -f dshow -i dummy
```

`transcribe --list-devices` parses that output, shows all DirectShow audio capture devices as microphone candidates, and includes DirectShow `Alternative name` values as selectable IDs/aliases. A configured `default` or old synthetic placeholder is resolved to a concrete enumerated DirectShow audio device before capture; the app should not start ffmpeg with `audio=default` unless DirectShow actually listed a device with that exact name.

DirectShow does not reliably label which audio capture devices are system-output loopbacks. The output-device settings therefore surface concrete DirectShow loopback/virtual devices such as `Stereo Mix`, `virtual-audio-capturer`, VB-CABLE, Voicemeeter, BlackHole, or Soundflower when present, and also supplement normal Windows render endpoints from PowerShell as `wasapi-loopback` output candidates.

Windows render endpoints discovered from `Get-PnpDevice -Class AudioEndpoint` are labeled with the `wasapi-loopback` backend and routed through the bundled sidecar protocol, keeping Windows-specific recording concerns out of the main TUI/CLI. The helper is a pure-Go, no-cgo Windows executable that uses WASAPI loopback through COM, captures the endpoint's native mix format, downmixes/resamples to PCM16 16 kHz mono, and writes a normal WAV chunk for the shared RMS/OpenAI path. This is the stock Windows path for built-in speakers/headphones, equivalent in intent to Audacity's Windows WASAPI loopback recording mode.

DirectShow fallback output capture remains available when a real capture-able loopback source exists: enable **Stereo Mix** in the Windows audio control panel, or install a virtual loopback device such as **VB-CABLE**, **Voicemeeter**, or `virtual-audio-capturer`. These are fallback/remediation tools if the WASAPI sidecar is missing or a specific endpoint cannot be captured, not the primary stock-Windows requirement.

If system audio capture fails or records the wrong source, run `transcribe --list-devices` and choose a displayed `wasapi-loopback` render endpoint such as `Speakers (...) [Loopback]` or `Headphones (...) [Loopback]` for `audio.output_device_id` or `--output`. For helper diagnostics, `wasapi-loopback-recorder.exe list --json` lists active WASAPI render endpoints and marks the current default endpoint. If the bundled helper is missing, set `TRANSCRIBE_WINDOWS_WASAPI_HELPER` to its path or rebuild the Windows package.

MS Teams speaker recognition is best-effort on Windows via PowerShell process/window-title polling. It avoids native UI Automation/cgo for cross-compilation safety, updates output speaker labels when a recognizable Teams title is found, and otherwise falls back to `Person_?` with a warning.

### macOS

Microphone capture uses an AVFoundation-style default. System output capture requires a virtual loopback device such as BlackHole/Soundflower or a future native ScreenCaptureKit/CoreAudio helper. Configure `audio.output_device_id` to the ffmpeg AVFoundation input for that virtual device.

MS Teams speaker recognition is not available on macOS and falls back to `Person_?`.

## Limitations

- Chunking is short bounded-duration for the external ffmpeg backend; config keeps frame/silence fields for the future native segmenter.
- Windows WASAPI render-device capture is implemented in the bundled sidecar helper; unusual endpoint formats/devices may still need manual validation on real hardware.
- Other system-output capture depends on OS audio setup and `ffmpeg` support.
- Windows Teams speaker detection is best-effort title probing, not full UI Automation.
- OpenAI transcription is the only transcription provider.
- No summaries, assistant, vector store, agents, or local Whisper are included.
