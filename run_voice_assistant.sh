#!/bin/bash
# One-click launcher for CNS Voice Assistant

echo "🚀 Starting CNS Voice Assistant..."
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check for native audio dependencies on Mac/Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! brew list portaudio &> /dev/null; then
        echo "⚠️  PortAudio not found. Installing via Homebrew..."
        brew install portaudio
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if ! ldconfig -p | grep -q libportaudio; then
        echo "⚠️  PortAudio not found. Install with:"
        echo "    sudo apt-get install portaudio19-dev python3-pyaudio"
        echo "    or: sudo yum install portaudio-devel"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Install Python dependencies if needed
if ! python3 -c "import sounddevice" 2>/dev/null; then
    echo "📦 Installing voice dependencies..."
    pip3 install sounddevice soundfile SpeechRecognition gTTS pygame pyttsx3 keyboard numpy PyAudio
    echo "✅ Dependencies installed"
    echo ""
fi

# Run the assistant
python3 cns_voice_assistant.py
