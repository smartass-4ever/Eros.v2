"""
Growth Tracker - Visualize neuroplastic system development over time
Tracks learning events, pattern formation, and relationship growth

Now integrated with ExperienceBus for unified learning across all systems.
"""

import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
import io
import os

class GrowthTracker:
    """Track and visualize neuroplastic growth over time"""
    
    def __init__(self):
        self.learning_events = []
        self.pattern_snapshots = []
        self.relationship_history = defaultdict(list)
        self.system_metrics_history = []
        self._load_from_database()
        print("📈 Growth tracker initialized")
    
    def _load_from_database(self):
        """Load historical growth data from database"""
        try:
            from cns_database import CNSDatabase
            from sqlalchemy import text
            
            db = CNSDatabase()
            session = db.get_session()
            try:
                result = session.execute(text("""
                    SELECT event_type, event_data, timestamp 
                    FROM growth_events 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                """))
                for row in result:
                    self.learning_events.append({
                        'type': row[0],
                        'data': row[1],
                        'timestamp': row[2].timestamp() if hasattr(row[2], 'timestamp') else row[2]
                    })
                if self.learning_events:
                    print(f"📈 Loaded {len(self.learning_events)} historical growth events")
            finally:
                session.close()
        except Exception as e:
            pass
    
    def record_learning_event(self, event_type: str, data: Dict[str, Any]):
        """Record a learning event with timestamp"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        }
        self.learning_events.append(event)
        
        try:
            from cns_database import CNSDatabase
            from sqlalchemy import text
            import json
            
            db = CNSDatabase()
            session = db.get_session()
            try:
                session.execute(text("""
                    INSERT INTO growth_events (event_type, event_data, timestamp)
                    VALUES (:event_type, :event_data, NOW())
                """), {
                    'event_type': event_type,
                    'event_data': json.dumps(data)
                })
                session.commit()
            finally:
                session.close()
        except Exception as e:
            pass
    
    def record_relationship_snapshot(self, user_id: str, metrics: Dict[str, float]):
        """Record relationship metrics for a user"""
        snapshot = {
            'timestamp': time.time(),
            **metrics
        }
        self.relationship_history[user_id].append(snapshot)
    
    def record_system_metrics(self, metrics: Dict[str, float]):
        """Record overall system performance metrics"""
        self.system_metrics_history.append({
            'timestamp': time.time(),
            **metrics
        })
    
    def on_experience(self, experience):
        """ExperienceBus subscriber - learn from every experience"""
        try:
            from experience_bus import ExperienceType
            
            event_type_map = {
                ExperienceType.CONVERSATION: 'conversation',
                ExperienceType.EMOTIONAL_MOMENT: 'emotional_moment',
                ExperienceType.BELIEF_CHALLENGE: 'belief_challenge',
                ExperienceType.CURIOSITY_GAP: 'curiosity_gap',
                ExperienceType.LEARNING_EVENT: 'learning_event',
                ExperienceType.RELATIONSHIP_SHIFT: 'relationship_shift',
                ExperienceType.INSIGHT_GENERATED: 'insight_generated',
                ExperienceType.PROACTIVE_OUTREACH: 'proactive_outreach'
            }
            
            event_type = event_type_map.get(experience.experience_type, 'unknown')
            
            event_data = {
                'user_id': experience.user_id,
                'emotional_intensity': experience.get_emotional_intensity(),
                'emotional_valence': experience.get_emotional_valence(),
                'has_learning_opportunity': experience.has_learning_opportunity(),
                'relationship_stage': experience.relationship_stage
            }
            
            if experience.belief_conflict:
                event_data['belief_conflict'] = True
            if experience.curiosity_gap:
                event_data['curiosity_gap'] = True
            
            self.record_learning_event(f"bus_{event_type}", event_data)
            
            if experience.relationship_stage:
                self.record_relationship_snapshot(experience.user_id, {
                    'stage': experience.relationship_stage,
                    'emotional_intensity': experience.get_emotional_intensity(),
                    'emotional_valence': experience.get_emotional_valence()
                })
            
            try:
                from experience_bus import get_experience_bus
                bus = get_experience_bus()
                growth_summary = self.get_growth_summary()
                bus.contribute_learning("GrowthTracker", {
                    'events_last_hour': growth_summary.get('events_last_hour', 0),
                    'total_events': growth_summary.get('total_events', 0),
                    'users_tracked': growth_summary.get('users_tracked', 0)
                })
            except Exception:
                pass
                
        except Exception as e:
            print(f"⚠️ GrowthTracker experience error: {e}")
    
    def subscribe_to_bus(self):
        """Subscribe to the global ExperienceBus"""
        try:
            from experience_bus import get_experience_bus
            bus = get_experience_bus()
            bus.subscribe("GrowthTracker", self.on_experience)
            print("📈 GrowthTracker subscribed to ExperienceBus - unified learning active")
        except Exception as e:
            print(f"⚠️ GrowthTracker bus subscription failed: {e}")
    
    def get_growth_summary(self) -> Dict[str, Any]:
        """Get a summary of growth over time"""
        now = time.time()
        hour_ago = now - 3600
        day_ago = now - 86400
        week_ago = now - 604800
        
        events_last_hour = len([e for e in self.learning_events if e['timestamp'] > hour_ago])
        events_last_day = len([e for e in self.learning_events if e['timestamp'] > day_ago])
        events_last_week = len([e for e in self.learning_events if e['timestamp'] > week_ago])
        
        event_types = defaultdict(int)
        for event in self.learning_events:
            event_types[event['type']] += 1
        
        return {
            'total_events': len(self.learning_events),
            'events_last_hour': events_last_hour,
            'events_last_day': events_last_day,
            'events_last_week': events_last_week,
            'event_breakdown': dict(event_types),
            'users_tracked': len(self.relationship_history),
            'system_snapshots': len(self.system_metrics_history)
        }
    
    def generate_growth_chart(self, chart_type: str = 'learning', user_id: str = None) -> Optional[str]:
        """Generate a growth visualization chart and save to file"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == 'learning':
                self._plot_learning_growth(ax)
            elif chart_type == 'relationship' and user_id:
                self._plot_relationship_growth(ax, user_id)
            elif chart_type == 'system':
                self._plot_system_metrics(ax)
            elif chart_type == 'overview':
                self._plot_overview(fig)
            else:
                self._plot_learning_growth(ax)
            
            plt.tight_layout()
            
            os.makedirs('charts', exist_ok=True)
            filename = f"charts/growth_{chart_type}_{int(time.time())}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"📊 Chart saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"⚠️ Chart generation failed: {e}")
            return None
    
    def _plot_learning_growth(self, ax):
        """Plot learning events over time"""
        if not self.learning_events:
            ax.text(0.5, 0.5, 'No learning data yet', ha='center', va='center', fontsize=14)
            ax.set_title('Learning Growth (No Data)')
            return
        
        events_by_hour = defaultdict(int)
        for event in self.learning_events:
            hour_key = datetime.fromtimestamp(event['timestamp']).replace(minute=0, second=0, microsecond=0)
            events_by_hour[hour_key] += 1
        
        if events_by_hour:
            times = sorted(events_by_hour.keys())
            counts = [events_by_hour[t] for t in times]
            
            cumulative = []
            total = 0
            for c in counts:
                total += c
                cumulative.append(total)
            
            ax.fill_between(times, cumulative, alpha=0.3, color='#667eea')
            ax.plot(times, cumulative, color='#667eea', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Time')
            ax.set_ylabel('Total Learning Events')
            ax.set_title('🧠 Neuroplastic Growth Over Time')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            plt.xticks(rotation=45)
            ax.grid(True, alpha=0.3)
    
    def _plot_relationship_growth(self, ax, user_id: str):
        """Plot relationship development for a specific user"""
        history = self.relationship_history.get(user_id, [])
        
        if not history:
            ax.text(0.5, 0.5, f'No relationship data for user', ha='center', va='center', fontsize=14)
            ax.set_title('Relationship Growth (No Data)')
            return
        
        times = [datetime.fromtimestamp(h['timestamp']) for h in history]
        trust = [h.get('trust', 0) for h in history]
        dependency = [h.get('dependency', 0) for h in history]
        intimacy = [h.get('intimacy', 0) for h in history]
        
        ax.plot(times, trust, label='Trust', color='#48bb78', linewidth=2)
        ax.plot(times, dependency, label='Dependency', color='#ed8936', linewidth=2)
        ax.plot(times, intimacy, label='Intimacy', color='#ed64a6', linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Score (0-1)')
        ax.set_title('💕 Relationship Development')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_system_metrics(self, ax):
        """Plot system performance metrics"""
        if not self.system_metrics_history:
            ax.text(0.5, 0.5, 'No system metrics yet', ha='center', va='center', fontsize=14)
            ax.set_title('System Metrics (No Data)')
            return
        
        times = [datetime.fromtimestamp(m['timestamp']) for m in self.system_metrics_history]
        efficiency = [m.get('efficiency', 0) for m in self.system_metrics_history]
        
        ax.plot(times, efficiency, color='#38b2ac', linewidth=2)
        ax.fill_between(times, efficiency, alpha=0.3, color='#38b2ac')
        ax.set_xlabel('Time')
        ax.set_ylabel('Efficiency')
        ax.set_title('⚡ System Efficiency Over Time')
        ax.grid(True, alpha=0.3)
    
    def _plot_overview(self, fig):
        """Create a multi-panel overview"""
        fig.clear()
        
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_learning_growth(ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        event_types = defaultdict(int)
        for event in self.learning_events[-100:]:
            event_types[event['type']] += 1
        if event_types:
            labels = list(event_types.keys())[:5]
            values = [event_types[l] for l in labels]
            colors = ['#667eea', '#48bb78', '#ed8936', '#ed64a6', '#38b2ac']
            ax2.pie(values, labels=labels, colors=colors[:len(labels)], autopct='%1.0f%%')
            ax2.set_title('📊 Learning Event Types')
        else:
            ax2.text(0.5, 0.5, 'No data', ha='center', va='center')
        
        ax3 = fig.add_subplot(gs[1, :])
        summary = self.get_growth_summary()
        summary_text = f"""
        Total Learning Events: {summary['total_events']}
        Last Hour: {summary['events_last_hour']} events
        Last Day: {summary['events_last_day']} events
        Last Week: {summary['events_last_week']} events
        Users Tracked: {summary['users_tracked']}
        """
        ax3.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
                fontfamily='monospace', transform=ax3.transAxes)
        ax3.axis('off')
        ax3.set_title('📈 Growth Summary')
        
        fig.suptitle('🧠 Eros Neuroplastic Growth Dashboard', fontsize=14, fontweight='bold')


growth_tracker = GrowthTracker()
