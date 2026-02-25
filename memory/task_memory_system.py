"""
Task Memory System - Track temporal tasks, events, deadlines
Enables proactive reminders and follow-ups
"""

import time
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dateutil import parser as date_parser
import calendar


class TemporalTask:
    """Represents a time-bound task or event"""
    def __init__(self, description: str, date_info: Dict[str, Any], task_type: str):
        self.description = description
        self.date_info = date_info  # {'type': 'specific/recurring', 'timestamp': ..., etc}
        self.task_type = task_type  # 'deadline', 'event', 'birthday', 'anniversary', 'reminder'
        self.created_at = time.time()
        self.completed = False
        self.reminded = False
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'description': self.description,
            'date_info': self.date_info,
            'task_type': self.task_type,
            'created_at': self.created_at,
            'completed': self.completed,
            'reminded': self.reminded
        }


class TaskMemorySystem:
    """
    Tracks temporal tasks, deadlines, events, birthdays
    Enables proactive reminders and timely check-ins
    """
    
    def __init__(self):
        # Storage: {user_id: List[TemporalTask]}
        self.user_tasks: Dict[str, List[TemporalTask]] = {}
        
        print("📅 Task memory system initialized - temporal awareness active")
    
    def extract_temporal_info(self, message: str) -> List[Tuple[str, Dict[str, Any], str]]:
        """
        Extract dates, deadlines, events from message
        Returns list of (description, date_info, task_type)
        """
        results = []
        msg_lower = message.lower()
        
        # Birthday detection
        birthday_match = self._extract_birthday(message)
        if birthday_match:
            results.append(birthday_match)
        
        # Deadline/event detection
        temporal_matches = self._extract_temporal_events(message)
        results.extend(temporal_matches)
        
        return results
    
    def _extract_birthday(self, message: str) -> Optional[Tuple[str, Dict[str, Any], str]]:
        """Extract birthday information"""
        msg_lower = message.lower()
        
        # Pattern: "my birthday is [date]"
        birthday_patterns = [
            r"(?:my|the) birthday (?:is|was|will be) (?:on )?(.+?)(?:\.|,|$)",
            r"born (?:on|in) (.+?)(?:\.|,|$)",
            r"birthday.*?(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)",
            r"birthday.*?(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}"
        ]
        
        for pattern in birthday_patterns:
            match = re.search(pattern, msg_lower)
            if match:
                date_str = match.group(1).strip()
                
                try:
                    # Try to parse the date
                    parsed_date = self._parse_date(date_str)
                    
                    if parsed_date:
                        # Extract month and day (year doesn't matter for recurring birthdays)
                        date_info = {
                            'type': 'recurring_annual',
                            'month': parsed_date.month,
                            'day': parsed_date.day,
                            'original_text': date_str
                        }
                        
                        return (
                            f"Birthday on {calendar.month_name[parsed_date.month]} {parsed_date.day}",
                            date_info,
                            'birthday'
                        )
                except:
                    pass
        
        return None
    
    def _extract_temporal_events(self, message: str) -> List[Tuple[str, Dict[str, Any], str]]:
        """Extract time-bound events and deadlines"""
        results = []
        msg_lower = message.lower()
        
        # Patterns for temporal expressions
        temporal_patterns = [
            (r"(.+?)\s+(?:tomorrow|tmrw)", 'tomorrow', 'deadline'),
            (r"(.+?)\s+(?:today|tonight)", 'today', 'deadline'),
            (r"(.+?)\s+next\s+(\w+)", 'next_weekday', 'event'),
            (r"(.+?)\s+(?:on|this)\s+(\w+day)", 'this_weekday', 'event'),
            (r"(.+?)\s+in\s+(\d+)\s+(day|week|month)s?", 'relative', 'deadline'),
            (r"(?:exam|test|interview|meeting|appointment).*?(?:on|at)?\s*(.+?)(?:\.|,|$)", 'event_date', 'event')
        ]
        
        for pattern, date_type, task_type in temporal_patterns:
            matches = re.finditer(pattern, msg_lower)
            for match in matches:
                try:
                    if date_type == 'tomorrow':
                        description = match.group(1).strip()
                        target_date = datetime.now() + timedelta(days=1)
                        date_info = {
                            'type': 'specific',
                            'timestamp': target_date.timestamp(),
                            'date_str': 'tomorrow'
                        }
                        results.append((description, date_info, task_type))
                    
                    elif date_type == 'today':
                        description = match.group(1).strip()
                        target_date = datetime.now()
                        date_info = {
                            'type': 'specific',
                            'timestamp': target_date.timestamp(),
                            'date_str': 'today'
                        }
                        results.append((description, date_info, task_type))
                    
                    elif date_type == 'next_weekday':
                        description = match.group(1).strip()
                        weekday = match.group(2).strip()
                        target_date = self._get_next_weekday(weekday)
                        if target_date:
                            date_info = {
                                'type': 'specific',
                                'timestamp': target_date.timestamp(),
                                'date_str': f'next {weekday}'
                            }
                            results.append((description, date_info, task_type))
                    
                    elif date_type == 'relative':
                        description = match.group(1).strip()
                        amount = int(match.group(2))
                        unit = match.group(3)
                        
                        if unit == 'day':
                            target_date = datetime.now() + timedelta(days=amount)
                        elif unit == 'week':
                            target_date = datetime.now() + timedelta(weeks=amount)
                        elif unit == 'month':
                            target_date = datetime.now() + timedelta(days=amount * 30)
                        else:
                            continue
                        
                        date_info = {
                            'type': 'specific',
                            'timestamp': target_date.timestamp(),
                            'date_str': f'in {amount} {unit}{"s" if amount > 1 else ""}'
                        }
                        results.append((description, date_info, task_type))
                
                except Exception as e:
                    continue
        
        return results
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats"""
        try:
            # Use dateutil parser for flexible date parsing
            return date_parser.parse(date_str, fuzzy=True)
        except:
            pass
        
        # Manual patterns for common formats
        patterns = [
            r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?",  # MM/DD or MM/DD/YYYY
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})",  # Month Day
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str.lower())
            if match:
                try:
                    if '/' in date_str or '-' in date_str:
                        month = int(match.group(1))
                        day = int(match.group(2))
                        year = int(match.group(3)) if match.group(3) else datetime.now().year
                        if year < 100:
                            year += 2000
                        return datetime(year, month, day)
                    else:
                        # Month name
                        months = {
                            'january': 1, 'february': 2, 'march': 3, 'april': 4,
                            'may': 5, 'june': 6, 'july': 7, 'august': 8,
                            'september': 9, 'october': 10, 'november': 11, 'december': 12
                        }
                        month = months[match.group(1)]
                        day = int(match.group(2))
                        return datetime(datetime.now().year, month, day)
                except:
                    pass
        
        return None
    
    def _get_next_weekday(self, weekday_str: str) -> Optional[datetime]:
        """Get next occurrence of a weekday"""
        weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        target_weekday = weekdays.get(weekday_str.lower())
        if target_weekday is None:
            return None
        
        today = datetime.now()
        days_ahead = target_weekday - today.weekday()
        
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        
        return today + timedelta(days=days_ahead)
    
    def add_task(self, user_id: str, description: str, date_info: Dict[str, Any], task_type: str):
        """Add a temporal task for user"""
        if user_id not in self.user_tasks:
            self.user_tasks[user_id] = []
        
        task = TemporalTask(description, date_info, task_type)
        self.user_tasks[user_id].append(task)
        
        # Keep only last 100 tasks
        self.user_tasks[user_id] = self.user_tasks[user_id][-100:]
        
        print(f"📅 Added temporal task: {description} ({task_type})")
    
    def get_upcoming_tasks(self, user_id: str, days_ahead: int = 7) -> List[TemporalTask]:
        """Get tasks coming up in the next N days"""
        if user_id not in self.user_tasks:
            return []
        
        now = time.time()
        future_cutoff = now + (days_ahead * 86400)
        
        upcoming = []
        for task in self.user_tasks[user_id]:
            if task.completed:
                continue
            
            if task.date_info['type'] == 'specific':
                task_time = task.date_info['timestamp']
                if now <= task_time <= future_cutoff:
                    upcoming.append(task)
            
            elif task.date_info['type'] == 'recurring_annual':
                # Check if birthday is coming up this year
                today = datetime.now()
                this_year_birthday = datetime(
                    today.year,
                    task.date_info['month'],
                    task.date_info['day']
                )
                
                if this_year_birthday.timestamp() >= now and this_year_birthday.timestamp() <= future_cutoff:
                    upcoming.append(task)
        
        return upcoming
    
    def get_today_tasks(self, user_id: str) -> List[TemporalTask]:
        """Get tasks for today"""
        if user_id not in self.user_tasks:
            return []
        
        today = datetime.now().date()
        today_tasks = []
        
        for task in self.user_tasks[user_id]:
            if task.completed:
                continue
            
            if task.date_info['type'] == 'specific':
                task_date = datetime.fromtimestamp(task.date_info['timestamp']).date()
                if task_date == today:
                    today_tasks.append(task)
            
            elif task.date_info['type'] == 'recurring_annual':
                if task.date_info['month'] == today.month and task.date_info['day'] == today.day:
                    today_tasks.append(task)
        
        return today_tasks
    
    def mark_completed(self, user_id: str, description: str):
        """Mark a task as completed"""
        if user_id not in self.user_tasks:
            return
        
        for task in self.user_tasks[user_id]:
            if description.lower() in task.description.lower():
                task.completed = True
                print(f"✅ Task completed: {task.description}")
                break
