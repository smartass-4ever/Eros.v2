"""
Minimal Voice Output Module - Converts text to natural male voice
"""

import os
import tempfile
from subprocess import call

class VoiceOutputHandler:
    def __init__(self, voice_type="male"):
        self.voice_type = voice_type
        self.male_voice = "en-US-GuyNeural"  # Azure male voice
        
    def text_to_speech_gtts(self, text):
        """Convert text to speech using gTTS (simple, works offline after download)"""
        try:
            from gtts import gTTS
            import pygame
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(tmp_file.name)
                
                pygame.mixer.init()
                pygame.mixer.music.load(tmp_file.name)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                pygame.mixer.quit()
                os.unlink(tmp_file.name)
                
                return True
                
        except ImportError:
            return self._text_to_speech_simple(text)
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return False
    
    def _text_to_speech_simple(self, text):
        """Fallback: Use system TTS or pyttsx3"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'male' in voice.name.lower() or 'david' in voice.name.lower() or 'zira' not in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            
            engine.setProperty('rate', 175)
            engine.say(text)
            engine.runAndWait()
            return True
            
        except ImportError:
            if os.name == 'posix':
                try:
                    call(['say', text])
                    return True
                except:
                    pass
            
            print(f"\n⚠️  Voice output unavailable. Response printed only.")
            return False
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            print(f"\n⚠️  Voice output unavailable. Response printed only.")
            return False
    
    def speak(self, text):
        """Main method: Convert text to speech"""
        print(f"🔊 CNS: {text}")
        return self.text_to_speech_gtts(text)


if __name__ == "__main__":
    handler = VoiceOutputHandler()
    handler.speak("Hello! I'm your CNS voice assistant. Ready to have a conversation?")
