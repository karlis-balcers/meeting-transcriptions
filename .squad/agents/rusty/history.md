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
