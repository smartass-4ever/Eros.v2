@echo off
REM Simple launcher for CNS Voice Assistant

echo.
echo Launching CNS Voice Assistant...
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from python.org
    pause
    exit /b 1
)

REM Install core deps if needed
python -c "import sounddevice" >nul 2>&1
if errorlevel 1 (
    echo Installing voice packages...
    pip install sounddevice soundfile SpeechRecognition gTTS pyttsx3 keyboard numpy
)

REM Run the assistant
python cns_voice_standalone.py

pause
