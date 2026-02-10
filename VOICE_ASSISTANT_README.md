# CNS Voice Assistant - Quick Start Guide

## What Is This?

A minimal voice interface for the CNS (Cognitive Neural System) brain. Talk to your AI assistant using your laptop's microphone and hear it respond with a natural male voice.

**Full CNS Intelligence:**
- 4-tier memory system (remembers everything)
- James Bond personality (witty, charming, direct)
- All 20 cognitive systems active
- Relationship building and emotional intelligence

---

## How to Run on Your Laptop

### 1. Download All Files

Download this entire project folder to your laptop.

### 2. Install Python (if needed)

**Requirements:** Python 3.9 or higher

- **Windows:** Download from [python.org](https://www.python.org/downloads/)
- **Mac:** Comes pre-installed, or use `brew install python3`
- **Linux:** Use `sudo apt install python3 python3-pip`

### 3. Launch the Voice Assistant

#### On Windows:
Double-click: `run_voice_assistant.bat`

#### On Mac/Linux:
```bash
./run_voice_assistant.sh
```

Or manually:
```bash
pip install -r voice_requirements.txt
python cns_voice_assistant.py
```

---

## How to Use

1. **Launch** - Run the script (it will install dependencies automatically on first run)
2. **Wait for prompt** - You'll see: `🎤 Press SPACEBAR to talk...`
3. **Press SPACEBAR** (or ENTER if keyboard module isn't available)
4. **Speak** - Talk clearly into your mic for ~5 seconds
5. **Listen** - CNS processes and responds with voice
6. **Repeat** - Press spacebar again to continue the conversation

**To exit:** Say "goodbye", "exit", or "quit", or press Ctrl+C

---

## What You Need

### Hardware:
- Laptop with microphone (built-in or external)
- Speakers or headphones

### Software (auto-installed):
- `sounddevice` - Captures microphone audio
- `soundfile` - Audio file handling
- `SpeechRecognition` - Converts your speech to text
- `gTTS` - Converts CNS responses to speech
- `pygame` - Plays audio
- `keyboard` - Detects spacebar press (optional)

### API Keys:
- `MISTRAL_API_KEY` or `TOGETHER_API_KEY` - For LLM processing

Set environment variable before running:
```bash
# Mac/Linux
export MISTRAL_API_KEY="your_key_here"

# Windows (Command Prompt)
set MISTRAL_API_KEY=your_key_here

# Windows (PowerShell)
$env:MISTRAL_API_KEY="your_key_here"
```

---

## How It Works

```
┌─────────────────────────────────────────┐
│  You press SPACEBAR                     │
│         ↓                               │
│  Microphone captures audio (5 sec)      │
│         ↓                               │
│  Speech → Text (Google Speech API)      │
│         ↓                               │
│  CNS Brain processes (full intelligence)│
│         ↓                               │
│  Text → Speech (gTTS male voice)        │
│         ↓                               │
│  Plays through speakers                 │
└─────────────────────────────────────────┘
```

---

## Troubleshooting

### "No module named sounddevice" or "No module named PyAudio"

**You need native audio libraries installed first:**

**Mac:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**Linux (Fedora/RHEL):**
```bash
sudo yum install portaudio-devel
pip install pyaudio
```

**Windows:**
PyAudio requires Visual C++ Build Tools. Two options:

Option 1 (easier):
```bash
pip install pipwin
pipwin install pyaudio
```

Option 2 (if that fails):
1. Install Visual C++ Build Tools from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Then: `pip install pyaudio`

### "Speech recognition failed"

The assistant will automatically fall back to text input if speech recognition fails. You can type your message instead.

**To fix permanently:**
- Check mic permissions in system settings
- Make sure mic isn't muted
- Test mic: `python -c "import sounddevice; print(sounddevice.query_devices())"`

### "Voice output unavailable"

If gTTS fails, the assistant falls back to pyttsx3 (local TTS). If that also fails, responses will be printed only (no voice).

**To fix:**
```bash
pip install pyttsx3 pygame
```

On Mac, if pygame fails:
```bash
brew install sdl2 sdl2_mixer
pip install pygame
```

### "API key not found"

Set the environment variable before running:
```bash
# Mac/Linux
export MISTRAL_API_KEY="your_key_here"

# Windows (Command Prompt)
set MISTRAL_API_KEY=your_key_here

# Windows (PowerShell)
$env:MISTRAL_API_KEY="your_key_here"
```

### "Keyboard module error"

The assistant will fall back to ENTER key instead of spacebar. This is normal and doesn't affect functionality.

### Voice sounds robotic

The free gTTS voice is simple but functional. For better quality:
- Use pyttsx3 voices (already included as fallback)
- Upgrade to premium TTS: ElevenLabs, Google Cloud TTS
- Or use local Coqui TTS (requires 2GB+ model download)

---

## Customization

### Change Voice
Edit `voice_output_module.py`:
- Line 10: Change `male_voice` to different voice name
- Or implement custom TTS service

### Change Recording Duration
Edit `cns_voice_assistant.py`:
- Line 82: Change `duration=5` to longer/shorter

### Change Wake Action
Edit `cns_voice_assistant.py`:
- Line 60-67: Modify `wait_for_spacebar()` method

---

## Files Overview

- `cns_voice_assistant.py` - Main voice loop
- `voice_input_module.py` - Microphone → text
- `voice_output_module.py` - Text → speech
- `voice_requirements.txt` - Package dependencies
- `run_voice_assistant.sh` - Mac/Linux launcher
- `run_voice_assistant.bat` - Windows launcher

---

## What Makes This Special

Unlike typical chatbots, CNS:
- **Remembers** - Recalls conversations from weeks ago
- **Evolves** - Adapts personality based on your relationship
- **Understands** - Detects subtle emotions and responds appropriately
- **Contributes** - Shares knowledge, opinions, memories (not just reactive)
- **Helps** - Proactively offers solutions and assistance

All of this works in voice mode with the same intelligence as the Discord bot!

---

## Next Steps

Once you've tested voice locally:
1. Customize the personality (edit `unified_cns_personality.py`)
2. Add custom voice cloning (use Coqui TTS)
3. Deploy as always-listening assistant
4. Integrate with home automation
5. Add wake word ("Hey CNS")

---

**Enjoy talking to your AI companion!** 🎤🧠
