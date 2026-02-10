"""
Autonomous Research Agent - Researches solutions for user problems
Proactively looks up answers and offers help without being asked
"""

import os
import time
import json
import requests
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from user_intent_tracker import UserIntent


class ResearchResult:
    """Represents research findings for a user problem"""
    def __init__(self, intent: UserIntent, findings: str, confidence: float):
        self.intent_description = intent.description
        self.intent_type = intent.intent_type
        self.findings = findings
        self.confidence = confidence  # 0.0 - 1.0
        self.timestamp = time.time()
        self.completion_time = time.time()  # When research completed
        self.offered = False  # Legacy: True if offered anywhere
        self.offered_channels = []  # Track where offered: 'conversation', 'dm'
        self.user_response = None  # 'accepted', 'rejected', 'ignored'
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'intent_description': self.intent_description,
            'intent_type': self.intent_type,
            'findings': self.findings,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'completion_time': self.completion_time,
            'offered': self.offered,
            'offered_channels': self.offered_channels,
            'user_response': self.user_response,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat()
        }


class AutonomousResearchAgent:
    """
    Researches solutions for user problems autonomously
    Enables proactive help offering
    """
    
    def __init__(self, mistral_api_key: Optional[str] = None):
        self.mistral_api_key = mistral_api_key or os.environ.get("MISTRAL_API_KEY")
        self.mistral_api_url = "https://api.together.xyz/v1/chat/completions"
        
        # Storage: {user_id: List[ResearchResult]}
        self.research_results: Dict[str, List[ResearchResult]] = {}
        
        # Track ongoing research to avoid duplicates
        self.researching: Dict[str, set] = {}  # {user_id: set of problem descriptions}
        
        print("🔬 Autonomous research agent initialized - proactive solution finding active")
    
    async def research_problem(self, user_id: str, intent: UserIntent,
                               user_context: str = "") -> Optional[ResearchResult]:
        """
        Research a problem and find solutions
        Returns ResearchResult with findings
        """
        
        if not self.mistral_api_key:
            print("[RESEARCH] No API key - research disabled")
            return None
        
        # Check if already researching this
        if user_id not in self.researching:
            self.researching[user_id] = set()
        
        problem_key = intent.description.lower()[:100]
        if problem_key in self.researching[user_id]:
            print(f"[RESEARCH] Already researching: {intent.description[:50]}")
            return None
        
        # Mark as researching
        self.researching[user_id].add(problem_key)
        
        print(f"🔬 Researching solution for: {intent.description[:60]}...")
        
        try:
            # Build research prompt based on intent type
            if intent.intent_type == 'problem':
                findings = await self._research_problem_solution(intent.description, user_context)
            elif intent.intent_type == 'task':
                findings = await self._research_task_approach(intent.description, user_context)
            elif intent.intent_type == 'goal':
                findings = await self._research_goal_strategy(intent.description, user_context)
            else:
                findings = await self._research_general_help(intent.description, user_context)
            
            if findings:
                # Calculate confidence based on response quality
                confidence = self._calculate_confidence(findings)
                
                result = ResearchResult(
                    intent=intent,
                    findings=findings,
                    confidence=confidence
                )
                
                # Store result
                if user_id not in self.research_results:
                    self.research_results[user_id] = []
                
                self.research_results[user_id].append(result)
                
                # Keep only last 20 results per user
                self.research_results[user_id] = self.research_results[user_id][-20:]
                
                print(f"✅ Research complete: {len(findings)} chars, confidence: {confidence:.2f}")
                return result
            
        except Exception as e:
            print(f"[RESEARCH] Error researching problem: {e}")
        
        finally:
            # Remove from researching set
            self.researching[user_id].discard(problem_key)
        
        return None
    
    async def _research_problem_solution(self, problem: str, context: str) -> str:
        """Research solutions for a specific problem"""
        
        prompt = f"""The user is struggling with: "{problem}"

{f"Context: {context}" if context else ""}

Provide 2-3 practical solutions or approaches to help them. Be specific and actionable.

Focus on:
- What they can try right now
- Different approaches they might not have considered
- Common solutions that work for this type of problem

Keep it concise - 3-4 sentences max."""

        return await self._call_llm(prompt)
    
    async def _research_task_approach(self, task: str, context: str) -> str:
        """Research best approach for a task"""
        
        prompt = f"""The user needs to: "{task}"

{f"Context: {context}" if context else ""}

Suggest the most efficient approach or steps to accomplish this.

Focus on:
- Smart way to tackle this
- What to prioritize
- Common pitfalls to avoid

Keep it concise - 3-4 sentences max."""

        return await self._call_llm(prompt)
    
    async def _research_goal_strategy(self, goal: str, context: str) -> str:
        """Research strategy for achieving a goal"""
        
        prompt = f"""The user wants to: "{goal}"

{f"Context: {context}" if context else ""}

Provide a strategic approach to work toward this goal.

Focus on:
- First actionable steps
- Key principles that help
- Realistic timeframe or milestones

Keep it concise - 3-4 sentences max."""

        return await self._call_llm(prompt)
    
    async def _research_general_help(self, description: str, context: str) -> str:
        """General research for user needs"""
        
        prompt = f"""The user mentioned: "{description}"

{f"Context: {context}" if context else ""}

Provide helpful information, tips, or approaches that could assist them.

Keep it concise and actionable - 3-4 sentences max."""

        return await self._call_llm(prompt)
    
    async def _call_llm(self, prompt: str) -> str:
        """Call Mistral API for research"""
        
        try:
            response = requests.post(
                self.mistral_api_url,
                headers={
                    "Authorization": f"Bearer {self.mistral_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,  # Balanced for quality
                    "max_tokens": 400
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"[RESEARCH] API error: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"[RESEARCH] LLM call error: {e}")
            return ""
    
    def _calculate_confidence(self, findings: str) -> float:
        """Calculate confidence score for research findings"""
        if not findings:
            return 0.0
        
        # Basic heuristics
        score = 0.5  # Base confidence
        
        # Length indicates thoroughness
        if len(findings) > 100:
            score += 0.1
        if len(findings) > 200:
            score += 0.1
        
        # Specific indicators of quality
        quality_indicators = [
            'try', 'approach', 'solution', 'step', 'recommend',
            'consider', 'focus', 'key', 'important', 'effective'
        ]
        
        findings_lower = findings.lower()
        matches = sum(1 for indicator in quality_indicators if indicator in findings_lower)
        score += min(matches * 0.05, 0.3)
        
        return min(score, 1.0)
    
    def get_pending_solutions(self, user_id: str, max_age_hours: int = 48) -> List[ResearchResult]:
        """Get research results that haven't been offered yet"""
        if user_id not in self.research_results:
            return []
        
        cutoff = time.time() - (max_age_hours * 3600)
        
        pending = [
            result for result in self.research_results[user_id]
            if not result.offered and result.timestamp > cutoff
        ]
        
        # Sort by confidence (offer best solutions first)
        return sorted(pending, key=lambda r: r.confidence, reverse=True)
    
    def mark_offered(self, user_id: str, intent_description: str, channel: str = 'conversation'):
        """Mark research result as offered to user via specific channel"""
        if user_id not in self.research_results:
            return
        
        for result in self.research_results[user_id]:
            if intent_description.lower() in result.intent_description.lower():
                result.offered = True  # Legacy
                if channel not in result.offered_channels:
                    result.offered_channels.append(channel)
                print(f"🔬 Marked research as offered via {channel}: {result.intent_description[:50]}")
                break
    
    def get_fresh_solutions(self, user_id: str, last_check_time: float = 0, 
                           max_age_hours: int = 48) -> List[ResearchResult]:
        """
        Get solutions completed since last check that haven't been offered yet
        Perfect for in-conversation offers
        """
        if user_id not in self.research_results:
            return []
        
        cutoff = time.time() - (max_age_hours * 3600)
        
        fresh = [
            result for result in self.research_results[user_id]
            if (result.completion_time > last_check_time and  # Completed since last check
                len(result.offered_channels) == 0 and  # Not offered anywhere yet
                result.timestamp > cutoff)  # Not too old
        ]
        
        # Sort by confidence (offer best solutions first)
        return sorted(fresh, key=lambda r: r.confidence, reverse=True)
    
    def record_user_response(self, user_id: str, intent_description: str, response: str):
        """Record how user responded to offered help"""
        if user_id not in self.research_results:
            return
        
        for result in self.research_results[user_id]:
            if intent_description.lower() in result.intent_description.lower():
                result.user_response = response  # 'accepted', 'rejected', 'ignored'
                print(f"📊 User response to help: {response}")
                break
    
    def get_help_acceptance_rate(self, user_id: str) -> float:
        """Get rate at which user accepts offered help (for preference learning)"""
        if user_id not in self.research_results:
            return 0.5  # Default neutral
        
        offered = [r for r in self.research_results[user_id] if r.offered]
        
        if not offered:
            return 0.5
        
        accepted = len([r for r in offered if r.user_response == 'accepted'])
        
        return accepted / len(offered) if len(offered) > 0 else 0.5
    
    async def bulk_research(self, user_id: str, intents: List[UserIntent],
                           max_concurrent: int = 3) -> List[ResearchResult]:
        """Research multiple problems concurrently"""
        
        # Limit concurrent research
        research_tasks = []
        for intent in intents[:max_concurrent]:
            if intent.intent_type in ['problem', 'task']:  # Focus on actionable intents
                task = self.research_problem(user_id, intent)
                research_tasks.append(task)
        
        if research_tasks:
            results = await asyncio.gather(*research_tasks, return_exceptions=True)
            return [r for r in results if isinstance(r, ResearchResult)]
        
        return []
