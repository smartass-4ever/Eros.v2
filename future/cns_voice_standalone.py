#!/usr/bin/env python3
"""
CNS Voice Assistant - Standalone (No Discord Dependencies)
Full cognitive brain with voice I/O and microphone diagnostics
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Import only core systems (no discord)
try:
    from simple_memory_system import SimpleMemory
    from voice_input_module import VoiceInputHandler
    from voice_output_module import VoiceOutputHandler
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all files are in the same directory")
    sys.exit(1)

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except:
    KEYBOARD_AVAILABLE = False


class StandaloneCNSBrain:
    """Minimal CNS brain without Discord dependencies"""
    
    def __init__(self):
        self.user_id = "voice_assistant_user"
        self.memory = IntelligentMemorySystem()
        self.conversation_history = []
        self.personality = {
            'charming': 0.9,
            'witty': 0.85,
            'confident': 0.95,
            'kind': 0.8,
            'flirtatious': 0.7
        }
        self.relationship_stage = "acquaintance"
        self.interaction_count = 0
        
    def process_input(self, user_input):
        """Process user input and generate response"""
        self.interaction_count += 1
        
        # Store in memory
        self.memory.store_interaction(
            user_id=self.user_id,
            content=user_input,
            emotion_context={'valence': 0.5, 'arousal': 0.5}
        )
        
        # Add to history
        self.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate response
        response = self._generate_response(user_input)
        
        # Store response in memory
        self.conversation_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def _generate_response(self, user_input):
        """Generate witty, personality-driven response"""
        
        # Simple response patterns (AI-like without LLM)
        user_lower = user_input.lower()
        
        # Emotion-aware responses
        if any(word in user_lower for word in ['sad', 'upset', 'depressed', 'hurt']):
            responses = [
                "That sounds rough. Want to talk about it?",
                "I hear you. Sometimes we all need to get things off our chest.",
                "Sounds like you're dealing with something real. I'm listening."
            ]
        elif any(word in user_lower for word in ['happy', 'excited', 'great', 'awesome']):
            responses = [
                "That's fantastic! What made your day?",
                "I love the energy - tell me more!",
                "Sounds like something good happened. I'm all ears."
            ]
        elif any(word in user_lower for word in ['how are you', 'whats up', 'how you doing']):
            responses = [
                "Doing well - more importantly, how are you?",
                "I'm here and ready to chat. What's on your mind?",
                "All good on my end. What brings you by?"
            ]
        elif any(word in user_lower for word in ['thank', 'thanks']):
            responses = [
                "Anytime - that's what I'm here for.",
                "Happy to help. Anything else on your mind?",
                "My pleasure. Feel free to ask anytime."
            ]
        else:
            # Generic curious responses
            responses = [
                "That's interesting - tell me more about that.",
                "I'm curious - what made you think of that?",
                "Fair point. Where are you going with this?",
                "I didn't expect that. Keep going.",
                "That's a good thought. What else?"
            ]
        
        import random
        return random.choice(responses)
    
    def get_context(self):
        """Get current conversation context for display"""
        if len(self.conversation_history) > 5:
            return self.conversation_history[-5:]
        return self.conversation_history


class VoiceAssistantUI:
    def __init__(self, device_id=None):
        self.brain = StandaloneCNSBrain()
        self.voice_in = VoiceInputHandler(device_id=device_id)
        self.voice_out = VoiceOutputHandler()
        
    def print_welcome(self):
        print("\n" + "="*60)
        print("  CNS VOICE ASSISTANT")
        print("  Full Cognitive Brain + Natural Voice")
        print("="*60)
        print("\n🧠 Brain Status: ACTIVE")
        print("💾 Memory System: READY")
        print("🎤 Voice I/O: READY")
        print("\n" + "="*60 + "\n")
    
    def print_header(self):
        print("\n🎤 Ready to talk! Press SPACEBAR to begin...")
        if not KEYBOARD_AVAILABLE:
            print("   (or press ENTER if spacebar doesn't work)\n")
    
    async def run_conversation_loop(self):
        """Main voice conversation loop"""
        self.print_welcome()
        
        while True:
            self.print_header()
            
            try:
                self.wait_for_input()
                
                # Get voice input
                user_text = self.voice_in.listen(duration=7)
                
                if not user_text:
                    print("❌ Couldn't hear that. Try again?\n")
                    continue
                
                print(f"\n💬 You: {user_text}")
                
                # Check for exit
                if user_text.lower() in ['exit', 'quit', 'goodbye', 'bye', 'stop']:
                    response = "It's been great talking with you. See you soon!"
                    self.voice_out.speak(response)
                    print("\n👋 Goodbye!\n")
                    break
                
                # Process and respond
                print("💭 Processing...")
                response = self.brain.process_input(user_text)
                
                print(f"🤖 CNS: {response}\n")
                self.voice_out.speak(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Shutting down...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
    
    def wait_for_input(self):
        """Wait for spacebar or enter"""
        if KEYBOARD_AVAILABLE:
            try:
                keyboard.wait('space')
            except:
                input()
        else:
            input()
    
    def run(self):
        """Start the assistant"""
        try:
            asyncio.run(self.run_conversation_loop())
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")


def setup_microphone():
    """Microphone setup wizard"""
    print("\n" + "="*60)
    print("  MICROPHONE SETUP")
    print("="*60 + "\n")
    
    # Select microphone
    device_id = VoiceInputHandler.select_microphone()
    if device_id is None:
        print("\n❌ No microphone selected.")
        print("The assistant will use the default microphone.\n")
        return None
    
    # Test microphone
    print("\n🎤 Let's test your microphone...\n")
    handler = VoiceInputHandler(device_id=device_id)
    
    if handler.test_microphone(duration=3):
        print("✅ Microphone working properly!\n")
        return device_id
    else:
        print("\n⚠️  Microphone test had issues, but we'll try to continue.")
        print("If you have problems, try selecting a different mic.\n")
        return device_id


def main():
    print("\n" + "="*60)
    print("  Launching CNS Voice Assistant (Standalone)")
    print("="*60 + "\n")
    
    # Setup microphone
    device_id = setup_microphone()
    
    # Launch assistant
    assistant = VoiceAssistantUI(device_id=device_id)
    assistant.run()


if __name__ == "__main__":
    main()
