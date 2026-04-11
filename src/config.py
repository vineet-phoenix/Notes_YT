import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    """Application configuration with environment variable support."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    EMBEDDINGS_DIR = DATA_DIR / "embeddings"
    CHATS_DIR = DATA_DIR / "chats"
    EXPORTS_DIR = DATA_DIR / "exports"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # YouTube & Media
    YOUTUBE_CHUNK_SIZE = 1200
    VIDEO_DOWNLOAD_DIR = DATA_DIR / "videos"
    
    # AI/LLM Models
    LOCAL_MODEL_NAME = "google/flan-t5-base"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL_NAME = "gemini-pro-vision"
    
    # Vector Database
    VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chromadb")
    
    # Processing
    MAX_CHUNK_TOKENS = 512
    SCENE_DETECTION_THRESHOLD = 27.0
    
    # Performance
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Create necessary directories
for directory in [settings.EMBEDDINGS_DIR, settings.CHATS_DIR, 
                  settings.EXPORTS_DIR, settings.LOGS_DIR, 
                  settings.VIDEO_DOWNLOAD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
