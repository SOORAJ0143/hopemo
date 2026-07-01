# app/engines/rag/retriever.py
import chromadb
from chromadb.config import Settings
import asyncio
from typing import List
from concurrent.futures import ThreadPoolExecutor
import logging
from app.config import settings
from app.engines.memory.hopfield import HopfieldAssociativeMemory
import app.services.embedding

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        chroma_settings = Settings(anonymized_telemetry=False)
        try:
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=chroma_settings
            )
            self.persistent = True
        except Exception:
            logger.exception("Persistent ChromaDB failed to start; using in-memory ChromaDB")
            self.client = chromadb.EphemeralClient(settings=chroma_settings)
            self.persistent = False
        self.memory_collection = self.client.get_or_create_collection("user_memories")
        self.knowledge_collection = self.client.get_or_create_collection("knowledge_base")
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def upsert(self, user_id: str, text: str, embedding: List[float], metadata: dict):
        def _upsert():
            doc_id = f"{user_id}_{hash(text)}"
            metadata["user_id"] = user_id
            self.memory_collection.upsert(
                ids=[doc_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        await asyncio.get_event_loop().run_in_executor(self.executor, _upsert)
    
    async def search_memories(self, user_id: str, query_embedding: List[float], top_k=5):
        def _search():
            return self.memory_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"user_id": user_id}
            )
        result = await asyncio.get_event_loop().run_in_executor(self.executor, _search)
        matches = []
        if result['ids']:
            for i in range(len(result['ids'][0])):
                matches.append({
                    "text": result['documents'][0][i],
                    "metadata": result['metadatas'][0][i],
                    "distance": result['distances'][0][i]
                })
        return matches
    
    async def search_knowledge(self, query_embedding: List[float], top_k=5):
        def _search():
            return self.knowledge_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
        result = await asyncio.get_event_loop().run_in_executor(self.executor, _search)
        matches = []
        if result['ids']:
            for i in range(len(result['ids'][0])):
                matches.append({
                    "text": result['documents'][0][i],
                    "metadata": result['metadatas'][0][i],
                    "distance": result['distances'][0][i]
                })
        return matches

class RAGRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.associative_memory = HopfieldAssociativeMemory(vector_store)
    
    async def retrieve(self, query: str, user_id: str, top_k=5):
        # 1. Get query embedding
        query_embedding = await app.services.embedding.get_embedding(query)
        # 2. Retrieve from user memory (vector)
        memories = await self.vector_store.search_memories(user_id, query_embedding, top_k)
        # 3. Retrieve from knowledge base (RAG)
        knowledge = await self.vector_store.search_knowledge(query_embedding, top_k)
        # 4. Associative retrieval (Hopfield)
        associative = await self.associative_memory.retrieve_associative(query_embedding, user_id, top_k=2)
        return {
            "memories": memories,
            "knowledge": knowledge,
            "associative": associative
        }
