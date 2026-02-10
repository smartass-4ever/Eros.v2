# LLM Fine-Tuning System for CNS
# Handles persona conditioning, Gen-Z dataset integration, and multi-turn conversation training

import json
import os
import time
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class TrainingExample:
    """Training example for persona fine-tuning"""
    persona: str
    context: List[Dict[str, str]]  # conversation history
    user_input: str
    ideal_response: str
    emotional_context: Dict[str, Any]
    quality_score: float  # 0.0-1.0

@dataclass
class PersonaTrainingData:
    """Structured training data for each persona"""
    persona_name: str
    examples: List[TrainingExample]
    style_markers: List[str]
    personality_traits: Dict[str, float]

class LLMFineTuningSystem:
    """Advanced LLM fine-tuning for persona consistency and Gen-Z natural language"""
    
    def __init__(self, mistral_api_key: Optional[str] = None):
        self.mistral_api_key = mistral_api_key or os.getenv('MISTRAL_API_KEY')
        self.training_data = {}
        self.persona_models = {}
        self.gen_z_patterns = self._load_gen_z_patterns()
        
        # Persona-specific training configurations
        self.persona_configs = {
            'supportive_partner': {
                'style_emphasis': ['warmth', 'empathy', 'encouragement'],
                'language_patterns': ['gentle', 'supportive', 'inclusive'],
                'emotional_resonance': 0.9
            },
            'witty_companion': {
                'style_emphasis': ['humor', 'cleverness', 'playfulness'],
                'language_patterns': ['casual', 'witty', 'observational'],
                'emotional_resonance': 0.7
            },
            'analytical_guide': {
                'style_emphasis': ['logic', 'precision', 'insight'],
                'language_patterns': ['structured', 'analytical', 'methodical'],
                'emotional_resonance': 0.6
            },
            'casual_friend': {
                'style_emphasis': ['relatability', 'authenticity', 'spontaneity'],
                'language_patterns': ['informal', 'genuine', 'conversational'],
                'emotional_resonance': 0.8
            }
        }
    
    def _load_gen_z_patterns(self) -> Dict[str, List[str]]:
        """Load Gen-Z language patterns for natural expression"""
        return {
            'expressions': [
                'honestly', 'ngl', 'fr', 'periodt', 'no cap', 'bet', 'say less',
                'it\'s giving', 'main character energy', 'that\'s so valid',
                'we love to see it', 'this is it', 'absolutely not'
            ],
            'conversational_markers': [
                'like', 'you know', 'I mean', 'literally', 'actually',
                'wait', 'okay but', 'also', 'tbh', 'imo'
            ],
            'emotional_intensifiers': [
                'absolutely', 'completely', 'totally', 'literally',
                'genuinely', 'actually', 'seriously'
            ],
            'casual_connectors': [
                'so like', 'and then', 'but also', 'or maybe',
                'I guess', 'kinda', 'sorta'
            ]
        }
    
    def generate_persona_prompt(self, persona: str, context: Dict[str, Any]) -> str:
        """Generate persona-conditioned prompt for LLM"""
        config = self.persona_configs.get(persona, self.persona_configs['supportive_partner'])
        
        # Base persona conditioning
        prompt = f"""You are responding as the {persona.replace('_', ' ')} persona of CNS, an AI with genuine personality.

PERSONA TRAITS:
- Style emphasis: {', '.join(config['style_emphasis'])}
- Language patterns: {', '.join(config['language_patterns'])}
- Emotional resonance level: {config['emotional_resonance']}

CONTEXT:
- User emotion: {context.get('emotion', 'neutral')}
- Conversation mood: {context.get('mood', 'balanced')}
- Relationship level: {context.get('relationship_level', 'new')}
- Recent topics: {context.get('recent_topics', [])}

GENERATION GUIDELINES:
1. Embody the persona naturally - don't announce it
2. Use appropriate Gen-Z language patterns when fitting
3. Match the user's emotional energy
4. Build on conversation history
5. Be authentic, not performative

Respond with genuine personality that feels human and relatable."""

        return prompt
    
    def prepare_training_data(self, conversation_logs: List[Dict], user_feedback: List[Dict]) -> Dict[str, PersonaTrainingData]:
        """Prepare structured training data from conversation logs and feedback"""
        training_data = defaultdict(lambda: PersonaTrainingData(
            persona_name="", examples=[], style_markers=[], personality_traits={}
        ))
        
        for log in conversation_logs:
            persona = log.get('persona', 'supportive_partner')
            
            # Create training example
            example = TrainingExample(
                persona=persona,
                context=log.get('conversation_history', []),
                user_input=log.get('user_input', ''),
                ideal_response=log.get('cns_response', ''),
                emotional_context=log.get('emotional_context', {}),
                quality_score=log.get('user_rating', 0.7)  # Default quality
            )
            
            training_data[persona].examples.append(example)
            training_data[persona].persona_name = persona
        
        # Enhance with user feedback
        for feedback in user_feedback:
            persona = feedback.get('persona', 'supportive_partner')
            if feedback.get('preference') == 'positive':
                # Boost quality score for preferred responses
                for example in training_data[persona].examples:
                    if example.user_input == feedback.get('user_input'):
                        example.quality_score = min(1.0, example.quality_score + 0.2)
        
        return dict(training_data)
    
    def generate_fine_tuning_dataset(self, persona: str, min_examples: int = 100) -> List[Dict]:
        """Generate fine-tuning dataset for specific persona"""
        if persona not in self.training_data:
            return []
        
        persona_data = self.training_data[persona]
        dataset = []
        
        for example in persona_data.examples:
            if example.quality_score >= 0.6:  # Only use good examples
                # Create prompt-completion pair
                prompt = self.generate_persona_prompt(persona, {
                    'emotion': example.emotional_context.get('emotion', 'neutral'),
                    'mood': example.emotional_context.get('mood', 'balanced'),
                    'recent_topics': [msg.get('content', '')[:50] for msg in example.context[-3:]]
                })
                
                completion = example.ideal_response
                
                dataset.append({
                    'prompt': prompt,
                    'completion': completion,
                    'metadata': {
                        'persona': persona,
                        'quality_score': example.quality_score,
                        'emotional_context': example.emotional_context
                    }
                })
        
        # If we don't have enough examples, generate synthetic ones
        if len(dataset) < min_examples:
            dataset.extend(self._generate_synthetic_examples(persona, min_examples - len(dataset)))
        
        return dataset
    
    def _generate_synthetic_examples(self, persona: str, count: int) -> List[Dict]:
        """Generate synthetic training examples for persona consistency"""
        config = self.persona_configs[persona]
        synthetic_examples = []
        
        # Common scenarios for each persona
        scenarios = {
            'supportive_partner': [
                "I'm feeling overwhelmed with work lately",
                "I'm not sure if I made the right decision",
                "I've been struggling with anxiety",
                "I feel like I'm not good enough"
            ],
            'witty_companion': [
                "What's the deal with people who don't use turn signals?",
                "I just watched the most ridiculous movie",
                "My roommate has some interesting habits",
                "Technology is getting weird these days"
            ],
            'analytical_guide': [
                "Help me understand this complex situation",
                "I need to make a difficult decision",
                "What are the pros and cons here?",
                "Can you break this down for me?"
            ],
            'casual_friend': [
                "What's up? How's your day going?",
                "Just had the weirdest experience",
                "I'm bored, what should I do?",
                "Tell me something interesting"
            ]
        }
        
        for i in range(min(count, len(scenarios.get(persona, [])))):
            user_input = scenarios[persona][i % len(scenarios[persona])]
            
            # Generate appropriate response based on persona
            response = self._generate_persona_response_template(persona, user_input, config)
            
            synthetic_examples.append({
                'prompt': self.generate_persona_prompt(persona, {'emotion': 'neutral'}),
                'completion': response,
                'metadata': {
                    'persona': persona,
                    'quality_score': 0.8,
                    'synthetic': True
                }
            })
        
        return synthetic_examples
    
    def _generate_persona_response_template(self, persona: str, user_input: str, config: Dict) -> str:
        """Generate template response that embodies the persona"""
        style_patterns = {
            'supportive_partner': "I hear you, and that sounds really challenging. {empathy} {support} You're not alone in feeling this way.",
            'witty_companion': "Okay, that's {observation}! {witty_comment} {playful_insight}",
            'analytical_guide': "Let me help you break this down. {analysis} {structured_approach} {logical_conclusion}",
            'casual_friend': "Oh {casual_reaction}! {relatable_comment} {friendly_continuation}"
        }
        
        return style_patterns.get(persona, "I understand what you're saying. Let me think about that.")
    
    async def fine_tune_persona_model(self, persona: str, dataset: List[Dict]) -> str:
        """Fine-tune LLM model for specific persona (placeholder for actual fine-tuning)"""
        # In a real implementation, this would call Mistral's fine-tuning API
        # For now, we'll simulate the process and return a model ID
        
        print(f"[FINE-TUNING] Starting fine-tuning for persona: {persona}")
        print(f"[FINE-TUNING] Dataset size: {len(dataset)} examples")
        
        # Simulate training process
        training_config = {
            'model': 'mistral-7b-instruct',
            'training_data': dataset,
            'epochs': 3,
            'learning_rate': 0.0001,
            'persona_emphasis': self.persona_configs[persona]['emotional_resonance']
        }
        
        # Store training configuration
        self.persona_models[persona] = {
            'model_id': f"cns-{persona}-{int(time.time())}",
            'config': training_config,
            'training_date': time.time(),
            'status': 'trained'
        }
        
        print(f"[FINE-TUNING] Completed training for {persona}")
        print(f"[FINE-TUNING] Model ID: {self.persona_models[persona]['model_id']}")
        
        return self.persona_models[persona]['model_id']
    
    def get_persona_conditioning_prompt(self, persona: str, context: Dict[str, Any], 
                                     conversation_history: List[Dict]) -> str:
        """Get persona-conditioned prompt for real-time inference"""
        base_prompt = self.generate_persona_prompt(persona, context)
        
        # Add conversation history
        if conversation_history:
            history_text = "\nCONVERSATION HISTORY:\n"
            for msg in conversation_history[-3:]:  # Last 3 exchanges
                role = "User" if msg.get('role') == 'user' else "CNS"
                history_text += f"{role}: {msg.get('content', '')}\n"
            base_prompt += history_text
        
        # Add Gen-Z language integration
        if context.get('use_gen_z_patterns', True):
            gen_z_guidance = "\nLANGUAGE STYLE: Incorporate natural Gen-Z expressions when appropriate, but don't force them. Use patterns like 'honestly', 'ngl', 'literally' organically."
            base_prompt += gen_z_guidance
        
        base_prompt += "\nUser: {user_input}\nCNS:"
        
        return base_prompt
    
    def save_training_data(self, filepath: str):
        """Save training data to file"""
        training_export = {
            'personas': {},
            'gen_z_patterns': self.gen_z_patterns,
            'persona_configs': self.persona_configs
        }
        
        for persona, data in self.training_data.items():
            training_export['personas'][persona] = {
                'persona_name': data.persona_name,
                'examples': [asdict(example) for example in data.examples],
                'style_markers': data.style_markers,
                'personality_traits': data.personality_traits
            }
        
        with open(filepath, 'w') as f:
            json.dump(training_export, f, indent=2)
        
        print(f"[TRAINING] Saved training data to {filepath}")
    
    def load_training_data(self, filepath: str):
        """Load training data from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.gen_z_patterns = data.get('gen_z_patterns', {})
            self.persona_configs = data.get('persona_configs', {})
            
            for persona, persona_data in data.get('personas', {}).items():
                examples = [TrainingExample(**ex) for ex in persona_data['examples']]
                self.training_data[persona] = PersonaTrainingData(
                    persona_name=persona_data['persona_name'],
                    examples=examples,
                    style_markers=persona_data['style_markers'],
                    personality_traits=persona_data['personality_traits']
                )
            
            print(f"[TRAINING] Loaded training data from {filepath}")