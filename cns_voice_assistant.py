#!/usr/bin/env python3
"""
CNS Voice Assistant - Minimal voice interface for CNS brain
Press SPACEBAR to talk, CNS responds with voice
"""

import os
import sys
import asyncio
from voice_input_module import VoiceInputHandler
from voice_output_module import VoiceOutputHandler
from updated_cns_discord_bot import DiscordCNSCompanion

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("⚠️  keyboard module not available, using Enter key instead of spacebar")


class CNSVoiceAssistant:
    def __init__(self):
        print("🧠 Initializing CNS Brain...")
        self.cns = DiscordCNSCompanion()
        self.voice_input = VoiceInputHandler()
        self.voice_output = VoiceOutputHandler(voice_type="male")
        self.user_id = "voice_user_laptop"
        self.conversation_active = True
        print("✅ CNS Voice Assistant Ready!\n")
        
    async def process_message(self, user_input):
        """Send message to CNS brain and get response"""
        try:
            class MockMessage:
                def __init__(self, content, author_id):
                    self.content = content
                    self.author = type('obj', (object,), {'id': author_id})()
                    self.channel = type('obj', (object,), {'id': 'voice_channel'})()
            
            message = MockMessage(user_input, self.user_id)
            
            response = await self.cns.process_message_internal(
                message=message,
                channel_id='voice_channel',
                user_id=self.user_id
            )
            
            return response
            
        except Exception as e:
            print(f"❌ CNS processing error: {e}")
            return "I had a moment there. Could you say that again?"
    
    def wait_for_spacebar(self):
        """Wait for spacebar press"""
        if KEYBOARD_AVAILABLE:
            print("\n🎤 Press SPACEBAR to talk...")
            keyboard.wait('space')
            return True
        else:
            print("\n🎤 Press ENTER to talk...")
            input()
            return True
    
    async def conversation_loop(self):
        """Main conversation loop"""
        print("=" * 60)
        print("  CNS VOICE ASSISTANT")
        print("  Full Brain Intelligence + Natural Voice")
        print("=" * 60)
        print("\nReady to talk! I remember everything and evolve with each chat.\n")
        
        while self.conversation_active:
            try:
                self.wait_for_spacebar()
                
                user_text = self.voice_input.listen(duration=5)
                
                if not user_text:
                    print("❌ Couldn't hear you clearly. Try again?\n")
                    continue
                
                print(f"\n💬 You: {user_text}")
                
                if user_text.lower() in ['exit', 'quit', 'goodbye', 'bye']:
                    response = "See you later. It's been a pleasure."
                    self.voice_output.speak(response)
                    print("\n👋 CNS shutting down...")
                    break
                
                response = await self.process_message(user_text)
                
                self.voice_output.speak(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 CNS shutting down...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
    
    def run(self):
        """Start the voice assistant"""
        try:
            asyncio.run(self.conversation_loop())
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")


def main():
    """Launch CNS Voice Assistant"""
    print("\n" + "="*60)
    print("  Launching CNS Voice Assistant")
    print("="*60 + "\n")
    
    if not os.getenv('MISTRAL_API_KEY') and not os.getenv('TOGETHER_API_KEY'):
        print("⚠️  Warning: No API key found!")
        print("Set MISTRAL_API_KEY or TOGETHER_API_KEY environment variable")
        print("The assistant will work but may have limited capabilities\n")
    
    assistant = CNSVoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
