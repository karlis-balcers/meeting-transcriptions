#!/bin/bash

echo "==============================================="
echo "First-time setup for transcribe-in-textbox"
echo "Linux Installation Script"
echo "==============================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python 3.10 or higher using your distribution's package manager:"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip python3-venv"
    echo "  Fedora/RHEL:   sudo dnf install python3 python3-pip"
    echo "  Arch:          sudo pacman -S python python-pip"
    exit 1
fi

echo "Python found. Checking version..."
python3 --version

# Check Python version
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo "ERROR: Python 3.10 or higher is required."
    echo "Please upgrade your Python installation."
    exit 1
fi

echo
echo "Checking Poetry installation..."

# Check if Poetry is already installed
if command -v poetry &> /dev/null; then
    echo "Poetry is already installed."
    poetry --version
else
    echo "Poetry not found. Installing Poetry..."
    echo "Trying official installer first..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    if [ $? -ne 0 ]; then
        echo "Official installer failed. Trying pip installation..."
        python3 -m pip install --user poetry
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to install Poetry with both methods."
            echo "Please visit https://python-poetry.org/docs/#installation for manual installation."
            exit 1
        fi
    fi
    
    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    echo
    echo "Poetry installed successfully!"
    echo "NOTE: Poetry has been added to ~/.local/bin"
    echo "You may need to restart your terminal or run: source ~/.bashrc"
fi

# Make sure Poetry is in PATH
if ! command -v poetry &> /dev/null; then
    if [ -f "$HOME/.local/bin/poetry" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "ERROR: Poetry installation not found in expected location."
        echo "Please check your Poetry installation or add it to your PATH manually."
        exit 1
    fi
fi

echo
echo "Checking project setup..."

# Check if virtual environment already exists and has dependencies
if [ -d ".venv" ]; then
    echo "Virtual environment found. Checking if dependencies are installed..."
    if poetry run python -c "import openai, faster_whisper, pyaudio" 2>/dev/null; then
        echo "Dependencies are already installed and working!"
        echo "Checking for updates..."
        poetry update
        echo
        echo "Setup verification complete!"
        exit 0
    else
        echo "Some dependencies are missing or broken. Reinstalling..."
    fi
else
    echo "No virtual environment found. Setting up new environment..."
fi

echo
echo "Installing system dependencies (if needed)..."

# Check for common audio dependencies
if command -v apt-get &> /dev/null; then
    echo "Detected Debian/Ubuntu system. Installing audio dependencies..."
    sudo apt-get update
    sudo apt-get install -y python3-dev portaudio19-dev build-essential
elif command -v dnf &> /dev/null; then
    echo "Detected Fedora/RHEL system. Installing audio dependencies..."
    sudo dnf install -y python3-devel portaudio-devel gcc gcc-c++
elif command -v pacman &> /dev/null; then
    echo "Detected Arch system. Installing audio dependencies..."
    sudo pacman -S --needed python portaudio base-devel
else
    echo "Unknown distribution. You may need to install audio development libraries manually."
    echo "Required packages: python3-dev, portaudio-dev, build tools"
fi

echo
echo "Setting up virtual environment and installing dependencies..."
echo "Configuring Poetry to create virtual environment in project directory..."
poetry config virtualenvs.in-project true --local
echo "Checking Poetry configuration..."
poetry config --list | grep virtualenvs
echo "Updating lock file to match pyproject.toml changes..."
poetry lock
poetry install

if [ $? -ne 0 ]; then
    echo
    echo "Installation failed. Trying to regenerate lock file and install again..."
    poetry lock
    poetry install --no-root
    if [ $? -ne 0 ]; then
        echo
        echo "ERROR: Failed to install dependencies with both methods."
        echo "This might be due to:"
        echo "- Missing system dependencies (especially for PyAudio)"
        echo "- Network connectivity issues"
        echo "- Compilation errors"
        echo
        echo "Please check the error messages above and install any missing system packages."
        exit 1
    fi
fi

echo
echo "Updating dependencies to latest versions..."
poetry update

echo
echo "==============================================="
echo "Setup completed successfully!"
echo "==============================================="
echo
echo
