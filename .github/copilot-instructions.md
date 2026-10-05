# Meeting Transcriptions - Copilot Instructions

## Overview

Live transcription of a meeting from the microphone and the system output (the remote side of a call), shown in a Godot 2D UI with every speaker on a circle around the live transcript. Builds per-speaker stats and profiles across meetings. Optional live checks (mood, fact check, user-defined checks) run on a local LLM (Ollama or any OpenAI-compatible local server). There is no OpenAI assistant or summary feature anymore.

Targets Windows and macOS. Linux works for development.

## Architecture

- `engine/` - Python 3.11+ package, run as `python -m engine`. Restored from the last Python version (commit `c1d41bc`) for audio capture (`audio_capture.py`), Teams speaker detection (`speaker_detection.py`, Windows only), OpenAI transcription (`transcriber.py`) and the transcript filter. New: `session.py` (pipeline orchestration), `profiles.py` (stats and saved profiles), `checks.py` + `llm.py` (local AI), `server.py` (socket protocol), `settings.py` (JSON settings, imports old `.env`), `demo.py` (scripted meeting).
- `ui/` - Godot 4.4 project, GDScript only, UI built in code. `engine_client.gd` connects to the engine and starts it when it isn't running. `stage.gd` draws the speaker circle.
- Protocol: newline-delimited JSON over TCP on 127.0.0.1 (default port 47321). UI sends `{"id", "cmd", ...}`, engine replies `{"type": "response", "id", "ok", "data"|"error"}` and pushes events (`transcript`, `level`, `stats`, `check`, `state`, `status`, `llm`, `profiles`, `settings`...). See the docstring in `engine/server.py`.

## Threading (engine)

Per recording: one capture thread and one store/transcribe thread per source (mic, output), a Teams detection thread on Windows, and one check worker for the local LLM (drops old lines when it falls behind). Socket commands that block (start/stop, Ollama install, model pull) run on background threads; results come back as events.

## Conventions

- Keep the engine importable without audio libraries; `pyaudio` / `pyaudiowpatch` / `pywinauto` / `openai` are imported lazily.
- The OpenAI API key is never sent to the UI (`SettingsStore.public()` replaces it with a flag).
- Engine tests use `unittest` and fake audio/transcription: `python -m unittest discover -s tests -t .`.
- UI smoke test: `godot --headless --path ui -s res://tests/smoke.gd` against a demo engine (`MT_ENGINE_DEMO=1`).
- GDScript: typed variables where the type can be inferred, `:=` only with typed values, no `class_name` collisions with Godot globals.
