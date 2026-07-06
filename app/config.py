# app/config.py
# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict  # <-- IMPORT THIS
from typing import Optional
import os

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
    HF_TOKEN: Optional[str] = None 
    DATABASE_URL: str = "sqlite:///./data/hopemo.db"
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./hopemo.db")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # V2 STANDARD WAY:
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()