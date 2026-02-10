#!/bin/bash
# Simple launcher for CNS Voice Assistant

echo "Launching CNS Voice Assistant..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found"
    exit 1
fi

# Install deps if needed
python3 -c "import sounddevice" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing voice packages..."
    pip3 install sounddevice soundfile SpeechRecognition gTTS pyttsx3 keyboard numpy
fi

# Run
python3 cns_voice_standalone.py
