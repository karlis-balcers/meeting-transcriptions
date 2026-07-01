@echo off
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
set "TRANSCRIBE_DIR=%ROOT_DIR%transcribe"
set "TRANSCRIBE_EXE=%TRANSCRIBE_DIR%\build\windows-amd64\transcribe.exe"
set "TRANSCRIBE_HELPER_BUILT=%TRANSCRIBE_DIR%\build\windows-amd64\audio_capture.py"
set "TRANSCRIBE_HELPER_ROOT=%ROOT_DIR%audio_capture.py"
set "TRANSCRIBE_WINDOWS_AUDIO_HELPER=%TRANSCRIBE_HELPER_BUILT%"
if not exist "%TRANSCRIBE_WINDOWS_AUDIO_HELPER%" set "TRANSCRIBE_WINDOWS_AUDIO_HELPER=%TRANSCRIBE_HELPER_ROOT%"

set "TRANSCRIBE_WINDOWS_PYTHON=%ROOT_DIR%.venv\Scripts\python.exe"
if not exist "%TRANSCRIBE_WINDOWS_PYTHON%" set "TRANSCRIBE_WINDOWS_PYTHON=python"

echo ===============================================
echo Starting transcribe application
echo ===============================================
echo.

if not exist "%TRANSCRIBE_EXE%" (
    echo ERROR: Windows build not found.
    echo Run build_transcribe_win64.bat first.
    pause
    exit /b 1
)

echo Launching: %TRANSCRIBE_EXE%
echo Helper:    %TRANSCRIBE_WINDOWS_AUDIO_HELPER%
echo Python:    %TRANSCRIBE_WINDOWS_PYTHON%
echo.

"%TRANSCRIBE_EXE%" %*

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Application failed to start.
    echo Please check the error messages above.
    pause
    exit /b 1
)

endlocal
