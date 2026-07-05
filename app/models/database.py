# app/models/database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import uuid
import os
from app.config import settings
from datetime import datetime
# Read DATABASE_URL from environment (Render sets this automatically)
DATABASE_URL = settings.DATABASE_URL  

# Configure engine based on database type
if DATABASE_URL.startswith("postgresql"):
    # PostgreSQL (production on Render)
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=False
    )
else:
    # SQLite (local development)
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)

class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=generate_uuid)
    conversation_id = Column(String)
    role = Column(String)
    content = Column(Text)
    emotion_scores = Column(JSON)
    safety_flags = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class EmotionalAnalytics(Base):
    __tablename__ = "emotional_analytics"
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String)
    date = Column(DateTime, default=datetime.utcnow)
    avg_anxiety = Column(Float)
    avg_stress = Column(Float)
    avg_hope = Column(Float)
    dominant_emotion = Column(String)
    entries_count = Column(Integer)

Base.metadata.create_all(bind=engine)