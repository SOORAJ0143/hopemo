# app/services/embedding.py
import openai
from app.config import settings

client = openai.AsyncOpenAI()


async def get_embedding(text: str) -> list:
    response = await client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding