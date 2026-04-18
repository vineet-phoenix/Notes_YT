import os
import torch
from pathlib import Path
from typing import ClassVar
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    """Application configuration with environment variable support."""
    
    # Project paths (ClassVar - not Pydantic fields)
    PROJECT_ROOT: ClassVar[Path] = Path(__file__).parent.parent
    DATA_DIR: ClassVar[Path] = PROJECT_ROOT / "data"
    EMBEDDINGS_DIR: ClassVar[Path] = DATA_DIR / "embeddings"
    CHATS_DIR: ClassVar[Path] = DATA_DIR / "chats"
    EXPORTS_DIR: ClassVar[Path] = DATA_DIR / "exports"
    LOGS_DIR: ClassVar[Path] = PROJECT_ROOT / "logs"
    
    # YouTube & Media
    YOUTUBE_CHUNK_SIZE: int = 1200
    VIDEO_DOWNLOAD_DIR: ClassVar[Path] = DATA_DIR / "videos"
    
    # AI/LLM Models
    LOCAL_MODEL_NAME: str = "Qwen/Qwen3-0.6B"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Whisper Audio Transcription
    WHISPER_MODEL: str = "base"  # Options: tiny, base, small, medium, large
    AUDIO_CHUNK_SIZE: int = 30  # seconds
    
    # Gemini API
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    
    # Vector Database
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "chromadb")
    
    # Processing
    MAX_CHUNK_TOKENS: int = 512
    SCENE_DETECTION_THRESHOLD: float = 27.0
    
    # Performance
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE: int = 32
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Create necessary directories
for directory in [settings.EMBEDDINGS_DIR, settings.CHATS_DIR, 
                  settings.EXPORTS_DIR, settings.LOGS_DIR, 
                  settings.VIDEO_DOWNLOAD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
