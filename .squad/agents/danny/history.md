# Danny — History

## Core Context

- **Project:** A Python meeting transcription application that will be migrated from a Windows GUI to a TUI/CLI experience.
- **Role:** Lead Engineer
- **Joined:** 2026-05-27T15:26:08.950Z

## Learnings

<!-- Append learnings below -->
- 2026-05-27: Existing Python app splits cleanly into audio capture, OpenAI transcription, Teams speaker detection, transcript filtering/storage, and Tkinter/assistant layers; Go migration should keep the first four and cut assistant/summary/Tkinter scope.
- 2026-05-27: System output capture is the main migration risk; Windows WASAPI loopback is feasible, while macOS/Linux require explicit validation and clear failure guidance when no monitor/loopback device exists.
- 2026-05-27: New `transcribe` TUI needs strict stdout discipline: silent mode prints only the final transcript to stdout, with all status/errors routed to stderr.
- 2026-05-27: External ffmpeg capture must be treated as cancellable short chunks, not long recording sessions; live controls cancel/restart source loops and loudness comes from parsed WAV samples.
- 2026-05-27: Coordinator validation approved the Danny-led revision: editor diagnostics, Go tests/vet/race, native build, and linux/darwin/windows amd64/arm64 cross-compiles all passed from `transcribe/`.
