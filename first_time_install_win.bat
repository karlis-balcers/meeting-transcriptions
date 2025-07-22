@echo off
echo ===============================================
echo First-time setup for transcribe-in-textbox
echo Windows Installation Script
echo ===============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10 or higher from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Python found. Checking version...
python -c "import sys; print(f'Python {sys.version}')"
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"
if %errorlevel% neq 0 (
    echo ERROR: Python 3.10 or higher is required.
    echo Please upgrade your Python installation.
    pause
    exit /b 1
)

echo.
echo Checking Poetry installation...
REM Check if Poetry is already installed
poetry --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Poetry is already installed.
    poetry --version
) else (
    echo Poetry not found. Installing Poetry...
    echo Trying official installer first...
    curl -sSL https://install.python-poetry.org | python -
    if %errorlevel% neq 0 (
        echo Official installer failed. Trying pip installation...
        python -m pip install --user poetry
        if %errorlevel% neq 0 (
            echo ERROR: Failed to install Poetry with both methods.
            echo Please visit https://python-poetry.org/docs/#installation for manual installation.
            pause
            exit /b 1
        )
    )
    
    REM Add Poetry to PATH for current session
    set "PATH=%APPDATA%\Python\Scripts;%PATH%"
    
    echo.
    echo Poetry installed successfully!
    echo NOTE: You may need to restart your command prompt or add Poetry to your PATH manually.
    echo Poetry is typically installed at: %APPDATA%\Python\Scripts\poetry.exe
)

echo.
echo Checking project setup...

REM Check if virtual environment already exists and has dependencies
if exist ".venv" (
    echo Virtual environment found. Checking if dependencies are installed...
    poetry run python -c "import openai, faster_whisper, pyaudio" >nul 2>&1
    if %errorlevel% equ 0 (
        echo Dependencies are already installed and working!
        echo Checking for updates...
        poetry update
        goto :setup_complete
    ) else (
        echo Some dependencies are missing or broken. Reinstalling...
    )
) else (
    echo No virtual environment found. Setting up new environment...
)

echo Setting up virtual environment and installing dependencies...
echo Configuring Poetry to create virtual environment in project directory...
poetry config virtualenvs.in-project true --local
echo Checking Poetry configuration...
poetry config --list | findstr virtualenvs
echo Updating lock file to match pyproject.toml changes...
poetry lock
poetry install

if %errorlevel% neq 0 (
    echo.
    echo Installation failed. Trying to regenerate lock file and install again...
    poetry lock
    poetry install --no-root
    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Failed to install dependencies with both methods.
        echo This might be due to:
        echo - Missing system dependencies (especially for PyAudio)
        echo - Network connectivity issues
        echo - Poetry not being in PATH
        echo.
        echo For PyAudio issues on Windows, you may need to install Visual C++ Build Tools
        echo or download a pre-compiled wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
        pause
        exit /b 1
    )
)

echo.
echo Updating dependencies to latest versions...
poetry update

echo.
echo Checking virtual environment location...
if exist ".venv" (
    echo Virtual environment found in project directory: .venv
) else (
    echo Virtual environment not found in project directory.
    echo Checking Poetry environment info...
    poetry env info --path
    echo.
    echo Forcing Poetry to create virtual environment in project directory...
    echo Removing existing virtual environment...
    poetry env remove --all >nul 2>&1
    echo Setting configuration to create venv in project...
    poetry config virtualenvs.in-project true --local
    echo Creating virtual environment using Poetry env use...
    poetry env use python
    echo Installing dependencies...
    poetry install
    if exist ".venv" (
        echo SUCCESS: Virtual environment created in project directory!
    ) else (
        echo WARNING: Local virtual environment creation failed.
    )
)

:setup_complete

echo.
echo ===============================================
echo Setup completed successfully!
echo ===============================================
echo.
echo.
pause
