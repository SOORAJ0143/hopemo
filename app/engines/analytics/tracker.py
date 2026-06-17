# app/engines/analytics/tracker.py
from app.models.database import SessionLocal, EmotionalAnalytics
from datetime import datetime
import numpy as np

class AnalyticsTracker:
    async def update(self, user_id: str, emotion_scores: dict):
        db = SessionLocal()
        # Get today's entry or create
        today = datetime.utcnow().date()
        entry = db.query(EmotionalAnalytics).filter(
            EmotionalAnalytics.user_id == user_id,
            EmotionalAnalytics.date >= datetime(today.year, today.month, today.day)
        ).first()
        if not entry:
            entry = EmotionalAnalytics(user_id=user_id, date=datetime.utcnow(), entries_count=0)
            db.add(entry)
        
        # Update running averages
        n = entry.entries_count + 1
        entry.avg_anxiety = (entry.avg_anxiety * entry.entries_count + emotion_scores.get('anxiety', 0)) / n
        entry.avg_stress = (entry.avg_stress * entry.entries_count + emotion_scores.get('stress', 0)) / n
        entry.avg_hope = (entry.avg_hope * entry.entries_count + emotion_scores.get('hope', 0)) / n
        entry.entries_count = n
        db.commit()
        db.close()