"""
LLM-Optimized Knowledge System
Leverages external LLM for knowledge instead of storing static facts
"""

import os
import time
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class KnowledgeQuery:
    """Represents a knowledge query with context"""
    question: str
    domain: str
    context: str
    urgency: float
    user_intent: str

class LLMOptimizedKnowledgeSystem:
    """Knowledge system that leverages LLM intelligence instead of static storage"""
    
    def __init__(self):
        self.query_cache = {}  # Cache recent queries for performance
        self.cache_duration = 300  # 5 minutes cache
        self.domain_specialists = {
            'geography': self._geography_specialist,
            'science': self._science_specialist,
            'logic': self._logic_specialist,
            'history': self._history_specialist,
            'general': self._general_specialist
        }
        
    def get_knowledge(self, query: str, context: str = "", domain: str = "general") -> Optional[str]:
        """Get knowledge using LLM with optimized prompting"""
        
        # Check cache first
        cache_key = f"{query.lower()}_{domain}"
        if cache_key in self.query_cache:
            cached = self.query_cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_duration:
                return cached['response']
        
        # Parse query into structured format
        knowledge_query = KnowledgeQuery(
            question=query,
            domain=self._detect_domain(query) if domain == "general" else domain,
            context=context,
            urgency=self._assess_urgency(query),
            user_intent=self._detect_intent(query)
        )
        
        # Route to domain specialist
        specialist = self.domain_specialists.get(knowledge_query.domain, self._general_specialist)
        response = specialist(knowledge_query)
        
        # Cache successful responses
        if response and len(response) > 10:
            self.query_cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }
        
        return response
    
    def _detect_domain(self, query: str) -> str:
        """Detect the domain of a query"""
        query_lower = query.lower()
        
        geography_keywords = ['capital', 'country', 'continent', 'river', 'mountain', 'desert', 'ocean']
        science_keywords = ['atomic', 'force', 'element', 'nuclear', 'chemical', 'physics', 'biology']
        logic_keywords = ['if', 'all', 'some', 'negation', 'implies', 'therefore', 'conclude']
        history_keywords = ['when', 'who', 'war', 'empire', 'ancient', 'century', 'historical']
        
        if any(keyword in query_lower for keyword in geography_keywords):
            return 'geography'
        elif any(keyword in query_lower for keyword in science_keywords):
            return 'science'
        elif any(keyword in query_lower for keyword in logic_keywords):
            return 'logic'
        elif any(keyword in query_lower for keyword in history_keywords):
            return 'history'
        else:
            return 'general'
    
    def _assess_urgency(self, query: str) -> float:
        """Assess the urgency/importance of a query"""
        urgent_indicators = ['urgent', 'quickly', 'asap', 'emergency', 'help']
        query_lower = query.lower()
        
        urgency = 0.5  # Base urgency
        if any(indicator in query_lower for indicator in urgent_indicators):
            urgency += 0.3
        if query.endswith('?'):
            urgency += 0.1
        
        return min(urgency, 1.0)
    
    def _detect_intent(self, query: str) -> str:
        """Detect user intent from query"""
        query_lower = query.lower()
        
        if query_lower.startswith(('what is', 'what are')):
            return 'definition'
        elif query_lower.startswith(('which', 'what')):
            return 'selection'
        elif query_lower.startswith(('how many', 'how much')):
            return 'quantification'
        elif query_lower.startswith('where'):
            return 'location'
        elif query_lower.startswith('when'):
            return 'temporal'
        elif query_lower.startswith('why'):
            return 'explanation'
        elif query_lower.startswith('how'):
            return 'process'
        else:
            return 'general_inquiry'
    
    def _geography_specialist(self, query: KnowledgeQuery) -> Optional[str]:
        """Specialized geography knowledge retrieval"""
        prompt = f"""You are a geography expert. Answer this question with precise, factual information.

Question: {query.question}
Context: {query.context}

Requirements:
- Provide accurate geographical facts
- Be specific with numbers, names, and locations
- Keep response concise (1-2 sentences)
- Focus on the most important information

Answer:"""
        
        return self._call_llm(prompt)
    
    def _science_specialist(self, query: KnowledgeQuery) -> Optional[str]:
        """Specialized science knowledge retrieval"""
        prompt = f"""You are a science expert. Answer this question with accurate scientific information.

Question: {query.question}
Context: {query.context}

Requirements:
- Provide precise scientific facts
- Include specific numbers, formulas, or measurements when relevant
- Use proper scientific terminology
- Keep response clear and concise (1-2 sentences)

Answer:"""
        
        return self._call_llm(prompt)
    
    def _logic_specialist(self, query: KnowledgeQuery) -> Optional[str]:
        """Specialized logic and reasoning knowledge retrieval"""
        prompt = f"""You are a logic and reasoning expert. Answer this question with precise logical analysis.

Question: {query.question}
Context: {query.context}

Requirements:
- Provide accurate logical reasoning
- Explain the logical principle clearly
- Be precise with logical terms and conclusions
- Keep response focused (1-2 sentences)

Answer:"""
        
        return self._call_llm(prompt)
    
    def _history_specialist(self, query: KnowledgeQuery) -> Optional[str]:
        """Specialized history knowledge retrieval"""
        prompt = f"""You are a history expert. Answer this question with accurate historical information.

Question: {query.question}
Context: {query.context}

Requirements:
- Provide accurate historical facts
- Include specific dates, names, and events when relevant
- Be precise and factual
- Keep response concise (1-2 sentences)

Answer:"""
        
        return self._call_llm(prompt)
    
    def _general_specialist(self, query: KnowledgeQuery) -> Optional[str]:
        """General knowledge retrieval for other domains"""
        prompt = f"""Answer this question with accurate, factual information.

Question: {query.question}
Context: {query.context}
Intent: {query.user_intent}

Requirements:
- Provide accurate, factual information
- Be specific and precise
- Keep response concise but complete
- Focus on the most relevant information

Answer:"""
        
        return self._call_llm(prompt)
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Make optimized LLM call for knowledge retrieval (sync version for compatibility)"""
        return self._call_llm_sync(prompt)
    
    def _call_llm_sync(self, prompt: str) -> Optional[str]:
        """Synchronous LLM call - use _call_llm_async in async contexts"""
        together_api_key = os.getenv("TOGETHER_API_KEY") or os.getenv("MISTRAL_API_KEY")
        
        if not together_api_key:
            return None
        
        try:
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {together_api_key}"},
                json={
                    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 200,
                    "top_p": 0.9
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                if content.lower().startswith('answer:'):
                    content = content[7:].strip()
                
                return content if len(content) > 5 else None
            else:
                print(f"LLM API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"LLM call error: {e}")
            return None
    
    async def _call_llm_async(self, prompt: str) -> Optional[str]:
        """Async-safe LLM call - runs sync HTTP in thread pool to avoid blocking event loop"""
        import asyncio
        return await asyncio.to_thread(self._call_llm_sync, prompt)
    
    async def get_knowledge_async(self, question: str, context: str = "", user_intent: str = "general") -> Optional[str]:
        """Async version of get_knowledge for use in Discord bot async context"""
        import asyncio
        cache_key = f"{question}_{context}_{user_intent}"
        
        if cache_key in self.query_cache:
            cached = self.query_cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_duration:
                return cached['response']
        
        query = KnowledgeQuery(
            question=question,
            domain=self._detect_domain(question),
            context=context,
            user_intent=user_intent,
            complexity=self._estimate_complexity(question)
        )
        
        if query.complexity == 'simple':
            return None
        
        specialist = self.specialists.get(query.domain, self._general_specialist)
        prompt = self._build_specialist_prompt(specialist, query)
        
        response = await self._call_llm_async(prompt)
        
        if response:
            self.query_cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }
        
        return response
    
    def _build_specialist_prompt(self, specialist, query: KnowledgeQuery) -> str:
        """Build prompt for specialist - extracted for async use"""
        if specialist == self._science_specialist:
            return f"""Answer this science question with accurate, factual information.

Question: {query.question}
Context: {query.context}

Requirements:
- Provide scientifically accurate information
- Include relevant scientific principles
- Be precise and factual
- Keep response concise (1-2 sentences)

Answer:"""
        elif specialist == self._history_specialist:
            return f"""Answer this history question with accurate, factual information.

Question: {query.question}
Context: {query.context}

Requirements:
- Provide accurate historical facts
- Include specific dates, names, and events when relevant
- Be precise and factual
- Keep response concise (1-2 sentences)

Answer:"""
        else:
            return f"""Answer this question with accurate, factual information.

Question: {query.question}
Context: {query.context}
Intent: {query.user_intent}

Requirements:
- Provide accurate, factual information
- Be specific and precise
- Keep response concise but complete
- Focus on the most relevant information

Answer:"""
    
    def clear_cache(self):
        """Clear the query cache"""
        self.query_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        valid_entries = sum(1 for entry in self.query_cache.values() 
                           if current_time - entry['timestamp'] < self.cache_duration)
        
        return {
            'total_cached': len(self.query_cache),
            'valid_entries': valid_entries,
            'cache_hit_potential': valid_entries / max(len(self.query_cache), 1)
        }

class MinimalWorldModel:
    """Minimal world model that only stores essential user-specific information"""
    
    def __init__(self):
        # Only store user-specific context, not general knowledge
        self.user_context = {}
        self.conversation_history = []
        self.personal_facts = {}  # User's personal information only
        
    def update_personal_fact(self, user_id: str, fact_type: str, content: str):
        """Store user-specific personal information"""
        if user_id not in self.personal_facts:
            self.personal_facts[user_id] = {}
        
        self.personal_facts[user_id][fact_type] = {
            'content': content,
            'timestamp': time.time()
        }
    
    def get_personal_context(self, user_id: str) -> Dict[str, Any]:
        """Get user-specific context for personalized responses"""
        return self.personal_facts.get(user_id, {})
    
    def update_conversation_context(self, user_input: str, bot_response: str):
        """Store recent conversation for context"""
        self.conversation_history.append({
            'user': user_input,
            'bot': bot_response,
            'timestamp': time.time()
        })
        
        # Keep only recent conversations
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

def test_llm_optimized_system():
    """Test the LLM-optimized knowledge system"""
    print("🧪 Testing LLM-Optimized Knowledge System...")
    
    knowledge_system = LLMOptimizedKnowledgeSystem()
    
    test_queries = [
        ("What is the capital of Australia?", "geography"),
        ("Which continent has the most countries?", "geography"),
        ("What is the atomic number of carbon?", "science"),
        ("If all roses are flowers, what can we conclude?", "logic"),
        ("Which force holds atomic nuclei together?", "science")
    ]
    
    print(f"\n🔬 Testing {len(test_queries)} queries...")
    
    successful_retrievals = 0
    
    for i, (query, expected_domain) in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {query} ---")
        print(f"Expected domain: {expected_domain}")
        
        # Detect domain
        detected_domain = knowledge_system._detect_domain(query)
        print(f"Detected domain: {detected_domain}")
        
        # Get knowledge
        response = knowledge_system.get_knowledge(query, domain=detected_domain)
        
        if response:
            print(f"✅ Success: {response}")
            successful_retrievals += 1
        else:
            print("❌ Failed to retrieve knowledge")
    
    success_rate = successful_retrievals / len(test_queries)
    print(f"\n🎯 LLM-Optimized System Results:")
    print(f"   Success rate: {successful_retrievals}/{len(test_queries)} ({success_rate*100:.1f}%)")
    
    # Cache stats
    cache_stats = knowledge_system.get_cache_stats()
    print(f"   Cache entries: {cache_stats['total_cached']}")
    
    return success_rate

if __name__ == "__main__":
    test_llm_optimized_system()