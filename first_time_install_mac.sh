#!/bin/bash

echo "==============================================="
echo "First-time setup for transcribe-in-textbox"
echo "macOS Installation Script"
echo "==============================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python 3.10 or higher:"
    echo "  Option 1: Download from https://python.org"
    echo "  Option 2: Install via Homebrew: brew install python"
    echo "  Option 3: Install via pyenv: pyenv install 3.10.0"
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
echo "Checking for Homebrew (recommended for system dependencies)..."
if command -v brew &> /dev/null; then
    echo "Homebrew found."
    HAS_BREW=true
else
    echo "Homebrew not found. Some dependencies may need manual installation."
    echo "Consider installing Homebrew from https://brew.sh for easier dependency management."
    HAS_BREW=false
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
    echo "You may need to restart your terminal or run: source ~/.zshrc (or ~/.bash_profile)"
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
echo "Installing system dependencies..."

# Install audio dependencies
if [ "$HAS_BREW" = true ]; then
    echo "Installing audio dependencies via Homebrew..."
    brew install portaudio
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to install portaudio via Homebrew."
        echo "You may need to install it manually or use a different method."
    fi
else
    echo "Without Homebrew, you'll need to install portaudio manually:"
    echo "  Option 1: Install Homebrew and run: brew install portaudio"
    echo "  Option 2: Install MacPorts and run: sudo port install portaudio"
    echo "  Option 3: Compile from source"
    echo
    echo "Continuing with Poetry installation..."
fi

# Check for Xcode command line tools (needed for compilation)
if ! xcode-select -p &> /dev/null; then
    echo "Xcode command line tools not found. Installing..."
    xcode-select --install
    echo "Please complete the Xcode command line tools installation and run this script again."
    exit 1
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
        echo "- Missing system dependencies (especially portaudio for PyAudio)"
        echo "- Network connectivity issues"
        echo "- Compilation errors"
        echo
        echo "Common solutions:"
        echo "1. Install portaudio: brew install portaudio"
        echo "2. Install Xcode command line tools: xcode-select --install"
        echo "3. For PyAudio issues, try: LDFLAGS=\"-L$(brew --prefix portaudio)/lib\" CPPFLAGS=\"-I$(brew --prefix portaudio)/include\" poetry install"
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
echo "Note: If you encounter audio-related issues, make sure you have:"
echo "- Granted microphone permissions to Terminal/iTerm2 in System Preferences > Security & Privacy"
echo "- Installed portaudio via Homebrew: brew install portaudio"
echo
