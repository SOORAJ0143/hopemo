import os
from pypdf import PdfReader
from app.services.embedding import get_embedding

class KnowledgeIngestor:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    async def ingest(self, file_path: str, category: str = "general"):
        # 1. Read PDF
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        # 2. Simple chunking (split by paragraphs)
        chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 50]
        
        # 3. Generate embeddings and store each chunk
        for chunk in chunks[:20]:  # Limit to 20 chunks to save OpenAI costs on first test
            embedding = await get_embedding(chunk)
            doc_id = f"{os.path.basename(file_path)}_{hash(chunk)}"
            
            self.vector_store.knowledge_collection.upsert(
                ids=[doc_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"category": category, "source": file_path}]
            )
        print(f"Processed {file_path} ({len(chunks)} chunks)")
