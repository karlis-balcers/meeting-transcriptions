@echo off
setlocal EnableExtensions

echo ===============================================
echo Building transcribe for Windows 64-bit
echo ===============================================
echo.

set "TRANSCRIBE_DIR=%~dp0transcribe"
set "BUILD_DIR=%TRANSCRIBE_DIR%\build\windows-amd64"
set "BUILD_VENV=%BUILD_DIR%\.venv"
set "WSLENV=TRANSCRIBE_DIR/p"

where wsl.exe >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: WSL2 is not available or wsl.exe is not on PATH.
    pause
    exit /b 1
)

wsl.exe -d Ubuntu -- bash -lc "command -v go >/dev/null 2>&1" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Ubuntu WSL is not available or Go is not installed there.
    pause
    exit /b 1
)

wsl.exe -d Ubuntu -- bash -lc "set -euo pipefail; cd \"$TRANSCRIBE_DIR\"; mkdir -p build/windows-amd64; CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -o build/windows-amd64/transcribe.exe ./cmd/transcribe"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Windows amd64 build failed in Ubuntu WSL2.
    pause
    exit /b 1
)

if not exist "%BUILD_DIR%\transcribe.exe" (
    echo.
    echo ERROR: Build finished but the expected binary was not found.
    pause
    exit /b 1
)

if not exist "%~dp0audio_capture.py" (
    echo.
    echo ERROR: Windows WASAPI helper audio_capture.py was not found next to the repository root.
    pause
    exit /b 1
)

copy /Y "%~dp0audio_capture.py" "%BUILD_DIR%\audio_capture.py" >nul
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to stage audio_capture.py beside the Windows binary.
    pause
    exit /b 1
)

set "PYTHON_BOOTSTRAP=%TRANSCRIBE_DIR%\.venv\Scripts\python.exe"
if not exist "%PYTHON_BOOTSTRAP%" set "PYTHON_BOOTSTRAP=python"

"%PYTHON_BOOTSTRAP%" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Unable to run a Python interpreter for helper packaging.
    echo        Expected a repo-local .venv or python on PATH.
    pause
    exit /b 1
)

if exist "%BUILD_VENV%" (
    rmdir /s /q "%BUILD_VENV%" >nul 2>&1
)

echo Creating build-local Python venv for the WASAPI helper...
"%PYTHON_BOOTSTRAP%" -m venv "%BUILD_VENV%"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to create build-local Python venv at %BUILD_VENV%.
    pause
    exit /b 1
)

set "BUILD_VENV_PYTHON=%BUILD_VENV%\Scripts\python.exe"
if not exist "%BUILD_VENV_PYTHON%" (
    echo.
    echo ERROR: The build-local Python venv did not create Scripts\python.exe.
    pause
    exit /b 1
)

echo Installing WASAPI helper dependencies into the build venv...
"%BUILD_VENV_PYTHON%" -m pip install --quiet --disable-pip-version-check pyaudiowpatch numpy
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install pyaudiowpatch and numpy into the build venv.
    pause
    exit /b 1
)

echo.
echo Build succeeded: %BUILD_DIR%\transcribe.exe
echo Helper staged:   %BUILD_DIR%\audio_capture.py
echo Python venv:     %BUILD_VENV%
echo Python runtime:  %BUILD_VENV_PYTHON%

endlocal
exit /b 0