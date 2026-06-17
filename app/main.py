# app/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
import asyncio
import uuid

from app.config import settings
from app.engines.emotion.detector import EmotionalAnalyzer
from app.engines.safety.detector import SafetyDetector
from app.engines.rag.retriever import VectorStore, RAGRetriever
from app.engines.rag.reranker import Reranker, ContextMerger
from app.engines.response.generator import ResponseGenerator
from app.engines.response.validator import ResponseValidator
from app.engines.analytics.tracker import AnalyticsTracker
from app.engines.memory.hopfield import HopfieldAssociativeMemory
from app.models.database import SessionLocal, Conversation, Message

# Initialize components
emotion_analyzer = EmotionalAnalyzer()
safety_detector = SafetyDetector()
vector_store = VectorStore()
rag_retriever = RAGRetriever(vector_store)
reranker = Reranker()
context_merger = ContextMerger()
response_generator = ResponseGenerator()
response_validator = ResponseValidator()
analytics_tracker = AnalyticsTracker()
hopfield_memory = HopfieldAssociativeMemory(vector_store)

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

class ChatRequest(BaseModel):
    message: str
    user_id: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    emotion: dict
    safety: dict

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 1. Safety analysis
    safety = await safety_detector.check(req.message)
    if not safety["safe"] and safety["risk_level"] == "critical":
        return ChatResponse(
            response="I'm concerned about you. Please reach out to the Suicide & Crisis Lifeline at 988. Would you like me to help you find local resources?",
            conversation_id=req.conversation_id or str(uuid.uuid4()),
            emotion={},
            safety=safety
        )
    
    # 2. Emotion detection
    emotion = await emotion_analyzer.analyze(req.message)
    
    # 3. Get embedding
    from app.services.embedding import get_embedding
    query_embedding = await get_embedding(req.message)
    
    # 4. Retrieve memories, knowledge, associative
    retrieval = await rag_retriever.retrieve(req.message, req.user_id)
    
    # 5. Rerank
    all_docs = retrieval['memories'] + retrieval['knowledge']
    reranked = reranker.rank(req.message, all_docs)
    
    # 6. Merge context
    context = context_merger.merge(req.message, reranked, retrieval['knowledge'], retrieval['associative'])
    
    # 7. Generate response
    response_text = await response_generator.generate(req.message, context, emotion, safety)
    
    # 8. Validate response
    validation = await response_validator.validate(response_text)
    if not validation['safe']:
        response_text = "I want to ensure our conversation remains helpful. Could you rephrase or let me know how I can better support you?"
    
    # 9. Update memory (store embedding + metadata)
    await vector_store.upsert(req.user_id, req.message, query_embedding, {"type": "user_message", "emotion": emotion})
    # Also store in Hopfield associative memory
    await hopfield_memory.add_memory(req.user_id, req.message, query_embedding, {"type": "user_message"})
    
    # 10. Update analytics
    await analytics_tracker.update(req.user_id, emotion)
    
    # 11. Store conversation in SQLite
    db = SessionLocal()
    if not req.conversation_id:
        conv = Conversation(user_id=req.user_id)
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conv_id = conv.id
    else:
        conv_id = req.conversation_id
    msg = Message(conversation_id=conv_id, role="user", content=req.message, emotion_scores=emotion, safety_flags=safety)
    db.add(msg)
    msg2 = Message(conversation_id=conv_id, role="assistant", content=response_text)
    db.add(msg2)
    db.commit()
    db.close()
    
    return ChatResponse(
        response=response_text,
        conversation_id=conv_id,
        emotion=emotion,
        safety=safety
    )

@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("HOPEMO started")
    yield

app.router.lifespan_context = lifespan