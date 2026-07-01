# app/engines/analytics/tracker.py
from app.models.database import SessionLocal, EmotionalAnalytics
from datetime import datetime
import numpy as np

class AnalyticsTracker:
    async def update(self, user_id: str, emotion_scores: dict):
        db = SessionLocal()
        try:
            # Get today's entry or create
            today = datetime.utcnow().date()
            entry = db.query(EmotionalAnalytics).filter(
                EmotionalAnalytics.user_id == user_id,
                EmotionalAnalytics.date >= datetime(today.year, today.month, today.day)
            ).first()
            if not entry:
                entry = EmotionalAnalytics(
                    user_id=user_id,
                    date=datetime.utcnow(),
                    avg_anxiety=0.0,
                    avg_stress=0.0,
                    avg_hope=0.0,
                    entries_count=0,
                )
                db.add(entry)

            # Update running averages
            count = entry.entries_count or 0
            n = count + 1
            avg_anxiety = entry.avg_anxiety or 0.0
            avg_stress = entry.avg_stress or 0.0
            avg_hope = entry.avg_hope or 0.0

            entry.avg_anxiety = (avg_anxiety * count + emotion_scores.get('anxiety', 0)) / n
            entry.avg_stress = (avg_stress * count + emotion_scores.get('stress', 0)) / n
            entry.avg_hope = (avg_hope * count + emotion_scores.get('hope', 0)) / n
            entry.dominant_emotion = emotion_scores.get("primary_emotion")
            entry.entries_count = n
            db.commit()
        finally:
            db.close()
