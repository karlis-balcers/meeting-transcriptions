# Livingston — History

## Core Context

- **Project:** A Python meeting transcription application that will be migrated from a Windows GUI to a TUI/CLI experience.
- **Role:** Platform Developer
- **Joined:** 2026-05-27T15:26:08.961Z

## Learnings

<!-- Append learnings below -->
- 2026-05-27: Go TUI/CLI rebuild is low risk for Bubble Tea/Cobra/YAML config/OpenAI transcription; keep these behind internal interfaces for testability.
- 2026-05-27: Built-in system-output capture is the platform risk: Windows WASAPI and Linux Pulse/PipeWire monitor sources are viable, while macOS needs virtual loopback support first or a native ScreenCaptureKit/CoreAudio helper before claiming full capture parity.
- 2026-05-27: Implemented `transcribe/` as a standalone pure-Go module with Cobra/Bubble Tea/YAML, ffmpeg-backed chunk capture, OpenAI multipart transcription, ordered transcript persistence, and hardware-free tests.
- 2026-05-27: A nested `.gitignore` cannot safely ignore a build artifact named the same as its parent directory (`transcribe/transcribe`) without hiding the source tree; remove local binaries after validation or ignore from a parent scope later.
- 2026-05-27: Initial implementation was revised by Danny after Basher's rejection and is now approved; the ffmpeg-backed architecture remains, with short cancellable chunks and a green validation matrix.
- 2026-05-27: Windows DirectShow should never rely on synthetic `audio=default`; parse ffmpeg's dshow list, prefer concrete alternative-name IDs, and surface all dshow audio capture candidates because loopback-vs-mic classification is heuristic.
- 2026-05-27: Basher approved the Windows DirectShow device enumeration/capture fix and Coordinator validation passed; concrete enumerated dshow devices are now the accepted Windows capture path.
- 2026-05-28: Windows build wrappers should pass repo paths into Ubuntu WSL with `WSLENV=TRANSCRIBE_DIR/p` instead of manual `wslpath` argument juggling; the translated `TRANSCRIBE_DIR` keeps spaces safe and avoids batch quoting bugs.
- 2026-05-28: Ubuntu WSL2 on this machine needed `golang-go` installed before the Windows amd64 build would run; the validated toolchain is `go version go1.22.2 linux/amd64`.
- 2026-05-28: Windows amd64 build wrapper validated from PowerShell; `WSLENV=TRANSCRIBE_DIR/p` safely translates the repo path into Ubuntu and the pure-Go build lands at `transcribe/build/windows-amd64/transcribe.exe`.
- 2026-05-28: The device-picker Enter flow now applies the selected mic/output device and returns to Settings immediately, and `--logging 1` writes `transcribe.log` in the current working directory for session/chunk/OpenAI diagnostics.
