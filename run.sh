#!/bin/bash
# Run from source on macOS (or Linux): sets up .venv on first run, then opens the Godot UI
# (the UI starts the Python engine itself).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "First run: creating .venv and installing the engine..."
    command -v brew >/dev/null 2>&1 && brew list portaudio >/dev/null 2>&1 || {
        command -v brew >/dev/null 2>&1 && brew install portaudio || echo "Install portaudio (brew install portaudio) if pyaudio fails to build."
    }
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi

GODOT_BIN="${GODOT:-}"
if [ -z "$GODOT_BIN" ]; then
    for candidate in godot godot4 /Applications/Godot.app/Contents/MacOS/Godot "$HOME/Applications/Godot.app/Contents/MacOS/Godot"; do
        if command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ]; then GODOT_BIN="$candidate"; break; fi
    done
fi
if [ -z "$GODOT_BIN" ]; then
    echo "Godot 4.4+ not found. Install it (brew install --cask godot) or set GODOT=/path/to/Godot."
    exit 1
fi
exec "$GODOT_BIN" --path ui "$@"
