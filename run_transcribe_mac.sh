#!/bin/bash
set -u

# Run from repo root regardless of caller's current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "==============================================="
echo "Starting transcribe application"
echo "==============================================="
echo

# Check if Poetry is available
if ! command -v poetry >/dev/null 2>&1; then
    echo "ERROR: Poetry is not installed or not in PATH."
    echo "Please run ./first_time_install_mac.sh to set up the project."
    exit 1
fi

# Check if Poetry has an environment configured
if ! poetry env info >/dev/null 2>&1; then
    echo "ERROR: No Poetry virtual environment found."
    echo "Please run ./first_time_install_mac.sh to set up the project."
    exit 1
fi

# Optional warning if local .venv does not exist
if [ ! -d ".venv" ]; then
    echo "Warning: .venv not found in project directory."
    echo "Poetry may be using an external virtual environment."
    echo
fi

echo "Starting application..."
poetry run python transcribe.py
status=$?

if [ $status -ne 0 ]; then
    echo
    echo "ERROR: Application failed to start."
    echo "Please check the error messages above."
    exit $status
fi