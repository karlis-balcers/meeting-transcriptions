#!/bin/bash

echo "==============================================="
echo "Starting transcribe-in-textbox application"
echo "==============================================="
echo

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "ERROR: Poetry is not installed or not in PATH."
    echo "Please run ./first_time_install_linux.sh to set up the project."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found in project directory."
    echo "Checking if Poetry has a virtual environment configured..."
    if ! poetry env info >/dev/null 2>&1; then
        echo "ERROR: No virtual environment found."
        echo "Please run ./first_time_install_linux.sh to set up the project."
        exit 1
    else
        echo "Found Poetry virtual environment outside project directory."
        echo "This will still work, but consider running ./first_time_install_linux.sh"
        echo "to create a local .venv folder for easier management."
        echo
    fi
fi

echo "Starting application..."
poetry run python transcribe.py

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Application failed to start."
    echo "Please check the error messages above."
    exit 1
fi
