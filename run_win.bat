@echo off
REM Run from source on Windows: sets up .venv on first run, then opens the Godot UI
REM (the UI starts the Python engine itself).
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: creating .venv and installing the engine...
    py -3 -m venv .venv || python -m venv .venv
    if errorlevel 1 (
        echo Python 3.11+ is needed: https://www.python.org/downloads/
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

set "GODOT_BIN=%GODOT%"
if "%GODOT_BIN%"=="" (
    where godot >nul 2>&1 && set "GODOT_BIN=godot"
)
if "%GODOT_BIN%"=="" (
    where godot4 >nul 2>&1 && set "GODOT_BIN=godot4"
)
if "%GODOT_BIN%"=="" (
    echo Godot 4.4+ not found. Install it ^(winget install GodotEngine.GodotEngine^) or set GODOT=C:\path\to\Godot.exe
    pause
    exit /b 1
)
"%GODOT_BIN%" --path ui %*
