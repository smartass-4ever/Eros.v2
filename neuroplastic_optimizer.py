"""
Neuroplastic Optimization System - Peak Efficiency Integration
Ensures all neuroplastic modules feed into orchestration decisions and work at maximum efficiency
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
from cognitive_orchestrator import CognitiveState, AttentionPriority

@dataclass
class NeuroplasticInsight:
    source: str  # imagination_engine, rem_processing, etc.
    content: Any
    confidence: float
    relevance_score: float
    timestamp: float
    cognitive_enhancement: Dict[str, float]  # How this insight enhances processing

@dataclass
class SystemEfficiencyMetrics:
    utilization_rate: float  # 0.0-1.0
    processing_efficiency: float
    integration_score: float
    neuroplastic_activity: float
    
class NeuroplasticPatternCache:
    """Intelligent caching for frequently accessed cognitive patterns"""
    
    def __init__(self, max_size: int = 1000):
        self.patterns = {}
        self.access_frequency = defaultdict(int)
        self.success_rate = defaultdict(float)
        self.max_size = max_size
        self.access_history = deque(maxlen=100)
        
    def store_pattern(self, pattern_key: str, result: Any, success_score: float):
        """Store cognitive pattern with success tracking"""
        if len(self.patterns) >= self.max_size:
            self._evict_least_useful()
            
        self.patterns[pattern_key] = {
            'result': result,
            'timestamp': time.time(),
            'access_count': 0,
            'success_score': success_score
        }
        
    def retrieve_pattern(self, pattern_key: str) -> Optional[Any]:
        """Retrieve cached pattern and update usage statistics"""
        if pattern_key in self.patterns:
            pattern = self.patterns[pattern_key]
            pattern['access_count'] += 1
            self.access_frequency[pattern_key] += 1
            self.access_history.append((pattern_key, time.time()))
            return pattern['result']
        return None
        
    def _evict_least_useful(self):
        """Remove least useful patterns based on frequency and success"""
        if not self.patterns:
            return
            
        # Calculate utility scores
        utility_scores = {}
        for key, pattern in self.patterns.items():
            frequency_score = self.access_frequency[key] / max(1, pattern['access_count'])
            recency_score = 1.0 / (time.time() - pattern['timestamp'] + 1)
            success_score = pattern['success_score']
            utility_scores[key] = frequency_score * 0.4 + recency_score * 0.3 + success_score * 0.3
            
        # Remove least useful
        least_useful = min(utility_scores.keys(), key=lambda k: utility_scores[k])
        del self.patterns[least_useful]
        
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights for pattern optimization"""
        if not self.access_history:
            return {}
            
        recent_accesses = list(self.access_history)[-20:]
        frequent_patterns = [access[0] for access in recent_accesses]
        
        return {
            'cache_hit_rate': len([p for p in frequent_patterns if p in self.patterns]) / len(frequent_patterns),
            'most_frequent': max(set(frequent_patterns), key=frequent_patterns.count) if frequent_patterns else None,
            'cache_size': len(self.patterns),
            'optimization_needed': len(self.patterns) > self.max_size * 0.8
        }

class PredictiveResourceAllocator:
    """Predictive resource allocation based on user patterns and system efficiency"""
    
    def __init__(self):
        self.user_patterns = {}
        self.system_performance_history = deque(maxlen=50)
        self.resource_allocation_history = {}
        
    def learn_user_pattern(self, user_id: str, interaction_data: Dict[str, Any], outcome_quality: float):
        """Learn from user interaction patterns"""
        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = {
                'interaction_types': defaultdict(int),
                'preferred_processing': {},
                'optimal_resources': {},
                'relationship_strength': 0.0,
                'conversation_complexity': 0.5
            }
            
        pattern = self.user_patterns[user_id]
        pattern['interaction_types'][interaction_data.get('type', 'general')] += 1
        
        # Learn optimal resource allocation for this user
        resource_key = interaction_data.get('cognitive_load_level', 'medium')
        if resource_key not in pattern['optimal_resources']:
            pattern['optimal_resources'][resource_key] = []
        pattern['optimal_resources'][resource_key].append(outcome_quality)
        
        # Update relationship strength
        pattern['relationship_strength'] = min(1.0, pattern['relationship_strength'] + 0.01)
        
    def predict_optimal_allocation(self, user_id: str, current_context: Dict[str, Any]) -> Dict[str, float]:
        """Predict optimal resource allocation for this user/context"""
        if user_id not in self.user_patterns:
            return self._default_allocation()
            
        pattern = self.user_patterns[user_id]
        
        # Base allocation on relationship strength
        relationship_bonus = pattern['relationship_strength'] * 0.2
        
        # Adjust based on user's optimal patterns
        allocation = {
            'memory_search': 0.3 + relationship_bonus,
            'reasoning': 0.25,
            'attention': 0.25,
            'expression': 0.2,
            'neuroplastic_enhancement': relationship_bonus
        }
        
        # Adjust based on context complexity
        complexity = current_context.get('complexity', 0.5)
        allocation['reasoning'] += complexity * 0.1
        allocation['neuroplastic_enhancement'] += complexity * 0.05
        
        return allocation
        
    def _default_allocation(self) -> Dict[str, float]:
        """Default resource allocation for new users"""
        return {
            'memory_search': 0.3,
            'reasoning': 0.25,
            'attention': 0.25,
            'expression': 0.2,
            'neuroplastic_enhancement': 0.1
        }

class CrossSystemLearningEngine:
    """Cross-system learning where each module informs others"""
    
    def __init__(self):
        self.system_interactions = defaultdict(list)
        self.learning_matrix = {}
        self.optimization_insights = {}
        
    def record_system_interaction(self, source_system: str, target_system: str, 
                                interaction_type: str, outcome_quality: float):
        """Record how one system affects another"""
        interaction = {
            'source': source_system,
            'target': target_system,
            'type': interaction_type,
            'quality': outcome_quality,
            'timestamp': time.time()
        }
        
        self.system_interactions[f"{source_system}->{target_system}"].append(interaction)
        
        # Learn optimal interaction patterns
        key = f"{source_system}->{target_system}->{interaction_type}"
        if key not in self.learning_matrix:
            self.learning_matrix[key] = []
        self.learning_matrix[key].append(outcome_quality)
        
    def get_optimal_system_sequence(self, starting_system: str, goal: str) -> List[str]:
        """Get optimal sequence of systems to achieve goal"""
        # Analyze learning matrix to find best path
        system_scores = {}
        
        for interaction_key, qualities in self.learning_matrix.items():
            if starting_system in interaction_key:
                avg_quality = sum(qualities) / len(qualities)
                target_system = interaction_key.split('->')[1].split('->')[0]
                system_scores[target_system] = avg_quality
                
        # Return systems ordered by effectiveness
        return sorted(system_scores.keys(), key=lambda s: system_scores[s], reverse=True)
        
    def get_system_enhancement_recommendations(self) -> Dict[str, List[str]]:
        """Get recommendations for enhancing each system"""
        recommendations = defaultdict(list)
        
        for interaction_key, qualities in self.learning_matrix.items():
            if len(qualities) >= 5:  # Enough data
                avg_quality = sum(qualities) / len(qualities)
                if avg_quality < 0.7:  # Room for improvement
                    source, target, interaction_type = interaction_key.split('->')
                    recommendations[source].append(f"Improve {interaction_type} with {target}")
                    
        return dict(recommendations)

class NeuroplasticOptimizer:
    """Main neuroplastic optimization system - coordinates all efficiency enhancements"""
    
    def __init__(self):
        self.pattern_cache = NeuroplasticPatternCache()
        self.resource_allocator = PredictiveResourceAllocator()
        self.cross_system_learning = CrossSystemLearningEngine()
        
        self.active_insights = deque(maxlen=10)
        self.system_metrics = {}
        self.optimization_cycle_count = 0
        
        # Neuroplastic enhancement factors
        self.enhancement_multipliers = {
            'imagination_active': 1.2,
            'rem_processing': 1.15,
            'creative_synthesis': 1.3,
            'memory_consolidation': 1.1,
            'pattern_recognition': 1.25
        }
        
    def integrate_neuroplastic_insight(self, insight: NeuroplasticInsight):
        """Integrate insight from neuroplastic modules"""
        self.active_insights.append(insight)
        
        # Cache valuable patterns
        pattern_key = f"{insight.source}_{hash(str(insight.content))}"
        self.pattern_cache.store_pattern(pattern_key, insight, insight.confidence)
        
        # Update system metrics
        self.system_metrics[insight.source] = {
            'last_activity': time.time(),
            'confidence': insight.confidence,
            'enhancement_level': sum(insight.cognitive_enhancement.values())
        }
    
    def record_learning_event(self, learning_type: str, topic: str, 
                              outcome_quality: float, user_id: str = None):
        """Record a learning event from knowledge/opinion systems for cross-system feedback"""
        self.cross_system_learning.record_system_interaction(
            source_system='world_model',
            target_system='response_generation',
            interaction_type=f'{learning_type}_{topic[:20]}',
            outcome_quality=outcome_quality
        )
        
        if user_id:
            self.resource_allocator.learn_user_pattern(
                user_id=user_id,
                interaction_data={'type': learning_type, 'topic': topic},
                outcome_quality=outcome_quality
            )
        
        pattern_key = f"learned_{learning_type}_{topic}"
        self.pattern_cache.store_pattern(pattern_key, {
            'type': learning_type,
            'topic': topic,
            'user_id': user_id,
            'timestamp': time.time()
        }, success_score=outcome_quality)
        
        try:
            from growth_tracker import growth_tracker
            growth_tracker.record_learning_event(learning_type, {
                'topic': topic,
                'quality': outcome_quality,
                'user_id': user_id
            })
        except Exception:
            pass
        
    def optimize_cognitive_processing(self, current_state: CognitiveState, 
                                   priority: AttentionPriority, user_id: str,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Main optimization function - coordinates all efficiency enhancements"""
        
        # Get predictive resource allocation
        optimal_allocation = self.resource_allocator.predict_optimal_allocation(user_id, context)
        
        # Apply neuroplastic enhancements
        enhanced_allocation = self._apply_neuroplastic_enhancements(optimal_allocation)
        
        # Get system sequence recommendations
        processing_sequence = self.cross_system_learning.get_optimal_system_sequence(
            'perception', context.get('goal', 'response_generation')
        )
        
        # Check for cached patterns
        context_key = f"{current_state.value}_{priority.value}_{user_id}"
        cached_strategy = self.pattern_cache.retrieve_pattern(context_key)
        
        self.optimization_cycle_count += 1
        
        return {
            'resource_allocation': enhanced_allocation,
            'processing_sequence': processing_sequence,
            'cached_strategy': cached_strategy,
            'neuroplastic_multiplier': self._calculate_neuroplastic_multiplier(),
            'optimization_insights': self._get_current_insights(),
            'efficiency_score': self._calculate_efficiency_score(),
            'system_recommendations': self.cross_system_learning.get_system_enhancement_recommendations()
        }
        
    def _apply_neuroplastic_enhancements(self, base_allocation: Dict[str, float]) -> Dict[str, float]:
        """Apply neuroplastic enhancements to resource allocation"""
        enhanced = base_allocation.copy()
        
        # Apply active neuroplastic insights
        for insight in self.active_insights:
            for enhancement_type, multiplier in insight.cognitive_enhancement.items():
                if enhancement_type in enhanced:
                    enhanced[enhancement_type] *= (1.0 + multiplier * 0.1)
                    
        # Apply system-wide enhancements
        neuroplastic_multiplier = self._calculate_neuroplastic_multiplier()
        for key in enhanced:
            enhanced[key] *= neuroplastic_multiplier
            
        # Normalize to ensure sum <= 1.0
        total = sum(enhanced.values())
        if total > 1.0:
            for key in enhanced:
                enhanced[key] /= total
                
        return enhanced
        
    def _calculate_neuroplastic_multiplier(self) -> float:
        """Calculate overall neuroplastic enhancement multiplier"""
        active_systems = len([m for m in self.system_metrics.values() 
                            if 'last_activity' in m and time.time() - m['last_activity'] < 10])
        
        base_multiplier = 1.0
        system_bonus = min(0.3, active_systems * 0.05)  # Up to 30% bonus
        
        return base_multiplier + system_bonus
        
    def _get_current_insights(self) -> List[Dict[str, Any]]:
        """Get current active insights for decision making"""
        return [
            {
                'source': insight.source,
                'relevance': insight.relevance_score,
                'confidence': insight.confidence,
                'enhancement': insight.cognitive_enhancement
            }
            for insight in self.active_insights
        ]
        
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall system efficiency score"""
        if not self.system_metrics:
            return 0.5
            
        # Factor in system utilization
        active_systems = len([m for m in self.system_metrics.values() 
                            if 'last_activity' in m and time.time() - m['last_activity'] < 30])
        utilization_score = active_systems / max(1, len(self.system_metrics))
        
        # Factor in cache efficiency
        cache_insights = self.pattern_cache.get_optimization_insights()
        cache_score = cache_insights.get('cache_hit_rate', 0.5)
        
        # Factor in neuroplastic activity - ✅ FIXED: Added safety check for enhancement_level
        avg_enhancement = sum(m.get('enhancement_level', 1.0) for m in self.system_metrics.values()) / len(self.system_metrics)
        neuroplastic_score = min(1.0, avg_enhancement / 2.0)
        
        return (utilization_score * 0.4 + cache_score * 0.3 + neuroplastic_score * 0.3)
        
    def update_system_performance(self, system_name: str, performance_data: Dict[str, Any]):
        """Update performance data for continuous optimization"""
        if system_name not in self.system_metrics:
            self.system_metrics[system_name] = {}
            
        self.system_metrics[system_name].update(performance_data)
        
        # Record cross-system interactions
        if 'influenced_systems' in performance_data:
            for influenced_system in performance_data['influenced_systems']:
                self.cross_system_learning.record_system_interaction(
                    system_name, influenced_system, 
                    performance_data.get('interaction_type', 'enhancement'),
                    performance_data.get('outcome_quality', 0.7)
                )
                
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health and optimization report"""
        return {
            'overall_efficiency': self._calculate_efficiency_score(),
            'active_systems': len([m for m in self.system_metrics.values() 
                                 if time.time() - m['last_activity'] < 30]),
            'neuroplastic_multiplier': self._calculate_neuroplastic_multiplier(),
            'cache_performance': self.pattern_cache.get_optimization_insights(),
            'optimization_cycles': self.optimization_cycle_count,
            'system_recommendations': self.cross_system_learning.get_system_enhancement_recommendations(),
            'active_insights_count': len(self.active_insights)
        }
    
    def get_neuroplastic_metrics(self) -> Dict[str, Any]:
        """Get live neuroplastic metrics for charting and visualization"""
        cache_insights = self.pattern_cache.get_optimization_insights()
        
        user_patterns_count = len(self.resource_allocator.user_patterns)
        total_interactions = sum(
            sum(pattern.get('interaction_types', {}).values())
            for pattern in self.resource_allocator.user_patterns.values()
        )
        avg_relationship_strength = (
            sum(p.get('relationship_strength', 0) for p in self.resource_allocator.user_patterns.values()) 
            / max(1, user_patterns_count)
        )
        
        learning_matrix_size = len(self.cross_system_learning.learning_matrix)
        total_system_interactions = sum(
            len(interactions) for interactions in self.cross_system_learning.system_interactions.values()
        )
        
        return {
            'pattern_cache': {
                'patterns_stored': len(self.pattern_cache.patterns),
                'cache_hit_rate': cache_insights.get('cache_hit_rate', 0.0),
                'most_frequent_pattern': cache_insights.get('most_frequent', None),
                'cache_size': cache_insights.get('cache_size', 0)
            },
            'user_adaptation': {
                'users_tracked': user_patterns_count,
                'total_interactions': total_interactions,
                'avg_relationship_strength': avg_relationship_strength
            },
            'cross_system_learning': {
                'learning_matrix_entries': learning_matrix_size,
                'total_system_interactions': total_system_interactions
            },
            'system_metrics': {
                'active_systems': len([m for m in self.system_metrics.values() 
                                      if 'last_activity' in m and time.time() - m.get('last_activity', 0) < 60]),
                'total_systems': len(self.system_metrics),
                'optimization_cycles': self.optimization_cycle_count
            },
            'active_insights': len(self.active_insights),
            'efficiency_score': self._calculate_efficiency_score(),
            'neuroplastic_multiplier': self._calculate_neuroplastic_multiplier()
        }
    
    def on_experience(self, experience):
        """ExperienceBus subscriber - update neural weights from every experience"""
        try:
            from experience_bus import ExperienceType
            
            user_id = experience.user_id
            emotional_intensity = experience.get_emotional_intensity()
            has_learning = experience.has_learning_opportunity()
            
            interaction_data = {
                'type': experience.experience_type.value,
                'emotional_intensity': emotional_intensity,
                'cognitive_load_level': 'high' if has_learning else 'medium'
            }
            outcome_quality = 0.7 + (emotional_intensity * 0.3) if has_learning else 0.5
            
            self.resource_allocator.learn_user_pattern(user_id, interaction_data, outcome_quality)
            
            if experience.belief_conflict or experience.curiosity_gap:
                insight = NeuroplasticInsight(
                    source='experience_bus',
                    content=experience.message_content or '',
                    confidence=0.8 if experience.belief_conflict else 0.7,
                    relevance_score=emotional_intensity,
                    timestamp=experience.timestamp,
                    cognitive_enhancement={
                        'memory_search': 0.1 if experience.belief_conflict else 0.05,
                        'reasoning': 0.15 if experience.curiosity_gap else 0.1,
                        'expression': 0.1
                    }
                )
                self.integrate_neuroplastic_insight(insight)
            
            if experience.experience_type == ExperienceType.INSIGHT_GENERATED:
                self.cross_system_learning.record_system_interaction(
                    source_system='inner_life',
                    target_system='response_generation',
                    interaction_type='insight_to_response',
                    outcome_quality=emotional_intensity
                )
            
            try:
                from experience_bus import get_experience_bus
                bus = get_experience_bus()
                bus.contribute_learning("NeuroplasticOptimizer", {
                    'efficiency_score': self._calculate_efficiency_score(),
                    'neuroplastic_multiplier': self._calculate_neuroplastic_multiplier(),
                    'active_insights': len(self.active_insights)
                })
            except Exception:
                pass
                
        except Exception as e:
            print(f"⚠️ NeuroplasticOptimizer experience error: {e}")
    
    def subscribe_to_bus(self):
        """Subscribe to the global ExperienceBus"""
        try:
            from experience_bus import get_experience_bus
            bus = get_experience_bus()
            bus.subscribe("NeuroplasticOptimizer", self.on_experience)
            print("🧠 NeuroplasticOptimizer subscribed to ExperienceBus - neural adaptation active")
        except Exception as e:
            print(f"⚠️ NeuroplasticOptimizer bus subscription failed: {e}")