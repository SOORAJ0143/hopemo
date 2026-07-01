# app/main.py
from app.utils.helpers import convert_numpy
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
import asyncio
import uuid
import openai
import logging
import json

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

logger = logging.getLogger(__name__)

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

@app.post("/admin/ingest")
async def admin_ingest():
    from scripts.ingest_knowledge import main
    await main()
    return {"status": "Ingestion started"}

@app.exception_handler(openai.AuthenticationError)
async def openai_authentication_exception_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={
            "detail": "OpenAI authentication failed. Check OPENAI_API_KEY in your .env file, then restart the server."
        },
    )

@app.exception_handler(openai.RateLimitError)
async def openai_rate_limit_exception_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "OpenAI rate limit reached. Try again later or check your OpenAI account limits."},
    )

@app.exception_handler(openai.APIConnectionError)
async def openai_connection_exception_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"detail": "Could not connect to OpenAI. Check your internet connection and try again."},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.exception("Unhandled application error")
    if settings.ENVIRONMENT == "development":
        return JSONResponse(status_code=500, content={"detail": str(exc)})
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

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

    from app.utils.helpers import convert_numpy   # ensure import is at top of file
    emotion = convert_numpy(emotion)
    safety = convert_numpy(safety) 
    
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
    try:
        metadata = {
    "type": "user_message",
    "primary_emotion": emotion.get("primary_emotion", ""),
    "intensity": float(emotion.get("intensity", 0.0)),
    "emotion_json": json.dumps(emotion)
}
        await vector_store.upsert(req.user_id, req.message, query_embedding, metadata)
        # Also store in Hopfield associative memory
        await hopfield_memory.add_memory(req.user_id, req.message, query_embedding, {"type": "user_message", "text": req.message})
    except Exception:
        logger.exception("Failed to store vector memory")
    
    # 10. Update analytics
    try:
        await analytics_tracker.update(req.user_id, emotion)
    except Exception:
        logger.exception("Failed to update analytics")
    
    # 11. Store conversation in SQLite
    db = SessionLocal()
    conv_id = req.conversation_id or str(uuid.uuid4())
    try:
        if not req.conversation_id:
            conv = Conversation(id=conv_id, user_id=req.user_id)
            db.add(conv)
            db.flush()
        msg = Message(conversation_id=conv_id, role="user", content=req.message, emotion_scores=emotion, safety_flags=safety)
        db.add(msg)
        msg2 = Message(conversation_id=conv_id, role="assistant", content=response_text)
        db.add(msg2)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to store conversation")
    finally:
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
