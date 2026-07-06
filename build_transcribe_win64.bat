@echo off
setlocal EnableExtensions

echo ===============================================
echo Building transcribe for Windows 64-bit
echo ===============================================
echo.

set "TRANSCRIBE_DIR=."
set "BUILD_DIR=%TRANSCRIBE_DIR%\build\windows-amd64"
set "WSLENV=TRANSCRIBE_DIR"

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

wsl.exe -d Ubuntu -- bash -lc "set -euo pipefail; cd \"$TRANSCRIBE_DIR\"; mkdir -p build/windows-amd64; CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -o build/windows-amd64/transcribe.exe ./cmd/transcribe; CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -o build/windows-amd64/wasapi-loopback-recorder.exe ./cmd/wasapi-loopback-recorder"
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

if not exist "%BUILD_DIR%\wasapi-loopback-recorder.exe" (
    echo.
    echo ERROR: Build finished but the expected WASAPI helper was not found.
    pause
    exit /b 1
)

REM Emit a transcribe.cmd launcher beside transcribe.exe. It keeps a persistent
REM cmd.exe attached so a double-clicked console-subsystem exe still gets a real
REM TTY for the Bubble Tea TUI and the transcript-on-quit stays visible.
> "%BUILD_DIR%\transcribe.cmd" echo @echo off
>> "%BUILD_DIR%\transcribe.cmd" echo cmd /k "%%~dp0transcribe.exe" %%*
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to write transcribe.cmd launcher beside transcribe.exe.
    pause
    exit /b 1
)

echo.
echo Build succeeded: %BUILD_DIR%\transcribe.exe
echo WASAPI helper:   %BUILD_DIR%\wasapi-loopback-recorder.exe
echo Launcher:        %BUILD_DIR%\transcribe.cmd

endlocal
exit /b 0