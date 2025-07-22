@echo off
echo ===============================================
echo Starting transcribe-in-textbox application
echo ===============================================
echo.

REM Check if Poetry is available
poetry --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Poetry is not installed or not in PATH.
    echo Please run first_time_install_win.bat to set up the project.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist ".venv" (
    echo Local virtual environment found: .venv
    echo Starting with local environment...
    .venv\Scripts\python.exe transcribe.py
    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Application failed to start with local environment.
        echo Falling back to Poetry...
        goto :poetry_run
    )
    goto :end
) else (
    echo Virtual environment not found in project directory.
    echo Checking if Poetry has a virtual environment configured...
    poetry env info >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: No virtual environment found.
        echo Please run first_time_install_win.bat to set up the project.
        echo Or run create_venv_manual_win.bat for a manual setup.
        pause
        exit /b 1
    ) else (
        echo Found Poetry virtual environment outside project directory.
        echo.
    )
)

:poetry_run
echo Starting application with Poetry...
poetry run python transcribe.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Application failed to start.
    echo Please check the error messages above.
    pause
    exit /b 1
)

:end
