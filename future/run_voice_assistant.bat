@echo off
REM One-click launcher for CNS Voice Assistant (Windows)

echo.
echo Starting CNS Voice Assistant...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from python.org
    pause
    exit /b 1
)

REM Check for Visual C++ Build Tools (needed for PyAudio on Windows)
echo Checking dependencies...
python -c "import sounddevice" >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Audio dependencies not found.
    echo.
    echo On Windows, you may need:
    echo 1. Visual C++ Build Tools (for PyAudio compilation)
    echo    Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
    echo 2. Or install pre-built PyAudio wheel:
    echo    pip install pipwin
    echo    pipwin install pyaudio
    echo.
    echo Installing Python packages...
    pip install sounddevice soundfile SpeechRecognition gTTS pygame pyttsx3 keyboard numpy
    echo.
    echo If errors occurred above, see VOICE_ASSISTANT_README.md for troubleshooting
    echo.
)

REM Run the assistant
python cns_voice_assistant.py

pause
