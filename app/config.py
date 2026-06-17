# app/config.py
# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict  # <-- IMPORT THIS
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "HOPEMO"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-3.5-turbo"  # Since you wanted GPT-3.5
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX: Optional[str] = None
    SECRET_KEY: str
    CHROMA_PERSIST_DIR: str = "./data/chromadb"
    USE_PINECONE: bool = False

    # V2 STANDARD WAY:
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()