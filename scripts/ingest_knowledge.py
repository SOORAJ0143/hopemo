# scripts/ingest_knowledge.py
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from app.engines.rag.retriever import VectorStore
from app.services.embedding import get_embedding
from app.engines.knowledge.ingestor import KnowledgeIngestor  # create simple ingestor

async def main():
    vs = VectorStore()
    ingestor = KnowledgeIngestor(vs)
    # Ingest all PDFs from knowledge_base folder
    for file in os.listdir("./knowledge_base"):
        if file.endswith(".pdf"):
            await ingestor.ingest(f"./knowledge_base/{file}", category="therapy")
    print("Ingestion complete")

if __name__ == "__main__":
    asyncio.run(main())