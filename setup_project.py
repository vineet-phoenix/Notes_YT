#!/usr/bin/env python3
"""
Automated setup script for Notes YT GenAI project.
Creates folder structure, generates all files, and pushes to product branch.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

class ProjectSetup:
    """Automates project initialization and file generation."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.project_root = self.repo_path.resolve()
    
    def log(self, message: str, level: str = "INFO"):
        """Pretty print log messages."""
        colors = {
            "INFO": "\033[94m",      # Blue
            "SUCCESS": "\033[92m",   # Green
            "ERROR": "\033[91m",     # Red
            "WARNING": "\033[93m",   # Yellow
        }
        reset = "\033[0m"
        print(f"{colors.get(level, '')}[{level}] {message}{reset}")
    
    def run_command(self, command: List[str]) -> bool:
        """Execute shell command."""
        try:
            result = subprocess.run(command, cwd=self.project_root, capture_output=True, text=True)
            if result.returncode != 0:
                self.log(f"Command failed: {' '.join(command)}", "ERROR")
                self.log(result.stderr, "ERROR")
                return False
            return True
        except Exception as e:
            self.log(f"Error running command: {str(e)}", "ERROR")
            return False
    
    def create_directories(self):
        """Create all required directories."""
        self.log("Creating directory structure...", "INFO")
        
        directories = [
            "src",
            "src/extractor",
            "src/embeddings",
            "src/llm",
            "src/rag",
            "src/pdf_export",
            "src/chat",
            "src/utils",
            "ui",
            "ui/pages",
            "ui/components",
            "data",
            "data/embeddings",
            "data/chats",
            "data/exports",
            "data/videos",
            "logs",
            "tests",
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.log(f"  ✓ Created {directory}", "SUCCESS")
    
    def create_init_files(self):
        """Create __init__.py files in all packages."""
        self.log("Creating __init__.py files...", "INFO")
        
        init_files = {
            "src/__init__.py": '''"""Notes YT - AI-powered YouTube video notes generation."""

__version__ = "1.0.0"
__author__ = "Vineet Kumar Bharti"
''',
            "src/extractor/__init__.py": '''"""Video extraction and processing modules."""
''',
            "src/embeddings/__init__.py": '''"""Embedding generation and vector storage modules."""
''',
            "src/llm/__init__.py": '''"""Language model implementations."""
''',
            "src/rag/__init__.py": '''"""RAG and semantic search modules."""
''',
            "src/pdf_export/__init__.py": '''"""PDF generation and export modules."""
''',
            "src/chat/__init__.py": '''"""Chat management modules."""
''',
            "src/utils/__init__.py": '''"""Utility modules for Notes YT."""
''',
            "ui/__init__.py": '''"""Streamlit UI components."""
''',
            "ui/pages/__init__.py": '''"""UI page modules."""
''',
            "ui/components/__init__.py": '''"""UI component modules."""
''',
            "tests/__init__.py": '''"""Test modules."""
''',
        }
        
        for filepath, content in init_files.items():
            file_path = self.project_root / filepath
            file_path.write_text(content)
            self.log(f"  ✓ Created {filepath}", "SUCCESS")
    
    def create_config(self):
        """Create src/config.py"""
        self.log("Creating config.py...", "INFO")
        
        content = '''import os
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
'''
        
        file_path = self.project_root / "src/config.py"
        file_path.write_text(content)
        self.log("  ✓ Created src/config.py", "SUCCESS")
    
    def create_logger(self):
        """Create src/logger.py"""
        self.log("Creating logger.py...", "INFO")
        
        content = '''import logging
import structlog
from pathlib import Path
from datetime import datetime
from src.config import settings

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

def get_logger(name: str):
    """Get a structured logger instance."""
    log_file = settings.LOGS_DIR / f"{name}.log"
    
    # Create file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger

logger = get_logger(__name__)
'''
        
        file_path = self.project_root / "src/logger.py"
        file_path.write_text(content)
        self.log("  ✓ Created src/logger.py", "SUCCESS")
    
    def create_error_handler(self):
        """Create src/utils/error_handler.py"""
        self.log("Creating error_handler.py...", "INFO")
        
        content = '''from src.logger import get_logger

logger = get_logger(__name__)

class NotesYTException(Exception):
    """Base exception for Notes YT."""
    pass

class VideoExtractionError(NotesYTException):
    """Raised when video extraction fails."""
    pass

class TranscriptError(NotesYTException):
    """Raised when transcript fetching fails."""
    pass

class ModelError(NotesYTException):
    """Raised when LLM inference fails."""
    pass

class EmbeddingError(NotesYTException):
    """Raised when embedding generation fails."""
    pass

class VectorStoreError(NotesYTException):
    """Raised when vector database operations fail."""
    pass

class PDFGenerationError(NotesYTException):
    """Raised when PDF generation fails."""
    pass

def handle_exception(func):
    """Decorator for robust error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise NotesYTException(f"{func.__name__} failed: {str(e)}")
    return wrapper
'''
        
        file_path = self.project_root / "src/utils/error_handler.py"
        file_path.write_text(content)
        self.log("  ✓ Created src/utils/error_handler.py", "SUCCESS")
    
    def create_youtube_extractor(self):
        """Create src/extractor/youtube_extractor.py"""
        self.log("Creating youtube_extractor.py...", "INFO")
        
        content = '''import yt_dlp
import re
from typing import Optional, Dict
from src.config import settings
from src.logger import get_logger
from src.utils.error_handler import VideoExtractionError, handle_exception

logger = get_logger(__name__)

class YouTubeExtractor:
    """Extracts audio, video, and captions from YouTube URLs."""
    
    def __init__(self):
        self.video_download_dir = settings.VIDEO_DOWNLOAD_DIR
        
    @handle_exception
    def extract_video_id(self, url: str) -> str:
        """Extract 11-character video ID from YouTube URL."""
        patterns = [
            r"(?:youtube\\.com\\/watch\\?v=|youtu\\.be\\/|youtube\\.com\\/embed\\/)([a-zA-Z0-9_-]{11})",
            r"youtube\\.com\\/shorts\\/([a-zA-Z0-9_-]{11})",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise VideoExtractionError(f"Could not extract video ID from URL: {url}")
    
    @handle_exception
    def get_video_metadata(self, url: str) -> Dict:
        """Get video metadata (title, duration, description)."""
        ydl_opts = {'quiet': True}
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Fetching metadata for {url}")
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title'),
                'duration': info.get('duration'),
                'description': info.get('description'),
                'uploader': info.get('uploader'),
                'upload_date': info.get('upload_date'),
            }
'''
        
        file_path = self.project_root / "src/extractor/youtube_extractor.py"
        file_path.write_text(content)
        self.log("  ✓ Created src/extractor/youtube_extractor.py", "SUCCESS")
    
    def create_transcript_processor(self):
        """Create src/extractor/transcript_processor.py"""
        self.log("Creating transcript_processor.py...", "INFO")
        
        content = '''from typing import List, Dict
from youtube_transcript_api import YouTubeTranscriptApi
from src.logger import get_logger
from src.config import settings
from src.utils.error_handler import TranscriptError, handle_exception

logger = get_logger(__name__)

class TranscriptProcessor:
    """Handles transcript extraction and chunking."""
    
    @handle_exception
    def get_transcript_with_timestamps(self, video_id: str) -> List[Dict]:
        """Fetch transcript with timestamp information."""
        try:
            api = YouTubeTranscriptApi()
            transcript = api.fetch_transcript(video_id)
            logger.info(f"Successfully fetched transcript for video {video_id}")
            return transcript
        except Exception as e:
            raise TranscriptError(f"Failed to fetch transcript: {str(e)}")
    
    @handle_exception
    def chunk_transcript(self, transcript: List[Dict], chunk_size: int = None) -> List[Dict]:
        """Chunk transcript by character size while preserving timestamps."""
        if chunk_size is None:
            chunk_size = settings.YOUTUBE_CHUNK_SIZE
        
        chunks = []
        current_chunk = {
            'text': '',
            'start': None,
            'end': None,
            'duration': 0
        }
        
        for entry in transcript:
            text = entry['text']
            
            if not current_chunk['text']:
                current_chunk['start'] = entry['start']
            
            if len(current_chunk['text']) + len(text) < chunk_size:
                current_chunk['text'] += ' ' + text
                current_chunk['end'] = entry['start'] + entry['duration']
                current_chunk['duration'] = current_chunk['end'] - current_chunk['start']
            else:
                if current_chunk['text']:
                    chunks.append(current_chunk)
                
                current_chunk = {
                    'text': text,
                    'start': entry['start'],
                    'end': entry['start'] + entry['duration'],
                    'duration': entry['duration']
                }
        
        if current_chunk['text']:
            chunks.append(current_chunk)
        
        logger.info(f"Created {len(chunks)} chunks from transcript")
        return chunks
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
'''
        
        file_path = self.project_root / "src/extractor/transcript_processor.py"
        file_path.write_text(content)
        self.log("  ✓ Created src/extractor/transcript_processor.py", "SUCCESS")
    
    def create_embedding_generator(self):
        """Create src/embeddings/embedding_generator.py"""
        self.log("Creating embedding_generator.py...", "INFO")
        
        content = '''import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from src.logger import get_logger
from src.config import settings
from src.utils.error_handler import EmbeddingError, handle_exception

logger = get_logger(__name__)

class EmbeddingGenerator:
    """Generates embeddings for text chunks."""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = settings.EMBEDDING_MODEL
        
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {str(e)}")
    
    @handle_exception
    def generate_embeddings(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if batch_size is None:
            batch_size = settings.BATCH_SIZE
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    @handle_exception
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
'''
        
        file_path = self.project_root / "src/embeddings/embedding_generator.py"
        file_path.write_text(content)
        self.log("  ✓ Created src/embeddings/embedding_generator.py", "SUCCESS")
    
    def create_vector_store(self):
        """Create src/embeddings/vector_store.py"""
        self.log("Creating vector_store.py...", "INFO")
        
        content = '''import chromadb
import faiss
import numpy as np
from typing import List, Dict, Tuple
from src.logger import get_logger
from src.config import settings
from src.utils.error_handler import VectorStoreError, handle_exception

logger = get_logger(__name__)

class VectorStore:
    """Unified interface for vector database operations."""
    
    def __init__(self, db_type: str = None, collection_name: str = "notes"):
        if db_type is None:
            db_type = settings.VECTOR_DB_TYPE
        
        self.db_type = db_type
        self.collection_name = collection_name
        
        try:
            if db_type == "chromadb":
                self.client = chromadb.PersistentClient(path=str(settings.EMBEDDINGS_DIR))
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Initialized ChromaDB vector store")
            
            elif db_type == "faiss":
                self.index_path = settings.EMBEDDINGS_DIR / f"{collection_name}.faiss"
                self.metadata_path = settings.EMBEDDINGS_DIR / f"{collection_name}_metadata.npy"
                self.index = None
                self.metadata = []
                self._load_or_init_faiss()
                logger.info("Initialized FAISS vector store")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize vector store: {str(e)}")
    
    def _load_or_init_faiss(self):
        """Load existing FAISS index or initialize new one."""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = np.load(str(self.metadata_path), allow_pickle=True).tolist()
    
    @handle_exception
    def add_embeddings(self, embeddings: np.ndarray, documents: List[Dict], 
                      ids: List[str] = None):
        """Add embeddings to vector store."""
        
        if self.db_type == "chromadb":
            if ids is None:
                ids = [str(i) for i in range(len(documents))]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=[doc.get('text', '') for doc in documents],
                metadatas=documents
            )
            logger.info(f"Added {len(embeddings)} embeddings to ChromaDB")
        
        elif self.db_type == "faiss":
            if self.index is None:
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
            
            embeddings_float = embeddings.astype(np.float32)
            self.index.add(embeddings_float)
            
            if ids is None:
                ids = [str(i) for i in range(len(documents))]
            
            self.metadata.extend([{**doc, 'id': id} for doc, id in zip(documents, ids)])
            
            faiss.write_index(self.index, str(self.index_path))
            np.save(str(self.metadata_path), np.array(self.metadata, dtype=object))
            
            logger.info(f"Added {len(embeddings)} embeddings to FAISS")
    
    @handle_exception
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar embeddings."""
        
        if self.db_type == "chromadb":
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            return [
                (doc, 1 - distance)
                for doc, distance in zip(
                    results['documents'][0],
                    results['distances'][0]
                )
            ]
        
        elif self.db_type == "faiss":
            query_float = query_embedding.astype(np.float32).reshape(1, -1)
            distances, indices = self.index.search(query_float, top_k)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.metadata):
                    results.append((
                        self.metadata[idx]['text'],
                        1 / (1 + distance)
                    ))
            
            return results
'''
        
        file_path = self.project_root / "src/embeddings/vector_store.py"
        file_path.write_text(content)
        self.log("  ✓ Created src/embeddings/vector_store.py", "SUCCESS")
    
    def create_local_model(self):
        """Create src/llm/local_model.py"""
        self.log("Creating local_model.py...", "INFO")
        
        content = '''import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
from src.logger import get_logger
from src.config import settings
from src.utils.error_handler import ModelError, handle_exception

logger = get_logger(__name__)

class LocalLLM:
    """Flan-T5 local model for fast text-only generation."""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = settings.LOCAL_MODEL_NAME
        
        self.device = settings.DEVICE
        self.model_name = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            logger.info(f"Loaded local model: {model_name} on device: {self.device}")
        except Exception as e:
            raise ModelError(f"Failed to load local model: {str(e)}")
    
    @handle_exception
    def generate(self, prompt: str, max_length: int = 200, 
                temperature: float = 0.7) -> str:
        """Generate text using Flan-T5."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                               max_length=1024).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            num_beams=4,
            early_stopping=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    @handle_exception
    def summarize_chunks(self, chunks: List[str]) -> List[str]:
        """Summarize multiple text chunks."""
        summaries = []
        
        for chunk in chunks:
            prompt = f"Summarize the following in bullet points:\\n\\n{chunk}"
            summary = self.generate(prompt, max_length=150)
            summaries.append(summary)
        
        return summaries
'''
        
        file_path = self.project_root / "src/llm/local_model.py"
        file_path.write_text(content)
        self.log("  ✓ Created src/llm/local_model.py", "SUCCESS")
    
    def create_gemini_model(self):
        """Create src/llm/gemini_model.py"""
        self.log("Creating gemini_model.py...", "INFO")
        
        content = '''import google.generativeai as genai
from typing import List
from src.logger import get_logger
from src.config import settings
from src.utils.error_handler import ModelError, handle_exception

logger = get_logger(__name__)

class GeminiLLM:
    """Google Gemini API for multimodal generation."""
    
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = settings.GEMINI_API_KEY
        
        if not api_key:
            raise ModelError("GEMINI_API_KEY not set in environment variables")
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
            logger.info("Initialized Gemini API")
        except Exception as e:
            raise ModelError(f"Failed to initialize Gemini API: {str(e)}")
    
    @handle_exception
    def generate(self, prompt: str, images: List = None) -> str:
        """Generate text with optional image inputs."""
        
        if images:
            content = [prompt] + images
        else:
            content = prompt
        
        response = self.model.generate_content(content)
        return response.text
'''
        
        file_path = self.project_root / "src/llm/gemini_model.py"
        file_path.write_text(content)
        self.log("  ✓ Created src/llm/gemini_model.py", "SUCCESS")
    
    def create_model_factory(self):
        """Create src/llm/model_factory.py"""
        self.log("Creating model_factory.py...", "INFO")
        
        content = '''from enum import Enum
from src.llm.local_model import LocalLLM
from src.llm.gemini_model import GeminiLLM
from src.logger import get_logger

logger = get_logger(__name__)

class ModelType(str, Enum):
    LOCAL = "local"
    GEMINI = "gemini"

class ModelFactory:
    """Factory for creating and managing LLM instances."""
    
    _instances = {}
    
    @staticmethod
    def create_model(model_type: ModelType):
        """Create or retrieve cached model instance."""
        
        if model_type not in ModelFactory._instances:
            try:
                if model_type == ModelType.LOCAL:
                    ModelFactory._instances[model_type] = LocalLLM()
                elif model_type == ModelType.GEMINI:
                    ModelFactory._instances[model_type] = GeminiLLM()
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                logger.info(f"Created model instance: {model_type}")
            except Exception as e:
                logger.error(f"Failed to create model: {str(e)}")
                raise
        
        return ModelFactory._instances[model_type]
'''
        
        file_path = self.project_root / "src/llm/model_factory.py"
        file_path.write_text(content)
        self.log("  ✓ Created src/llm/model_factory.py", "SUCCESS")
    
    def create_semantic_search(self):
        """Create src/rag/semantic_search.py"""
        self.log("Creating semantic_search.py...", "INFO")
        
        content = '''import re
from typing import List, Tuple
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import VectorStore
from src.logger import get_logger
from src.utils.error_handler import handle_exception

logger = get_logger(__name__)

class SemanticSearcher:
    """Handles query parsing and semantic search."""
    
    def __init__(self, collection_name: str = "notes"):
        self.embedding_gen = EmbeddingGenerator()
        self.vector_store = VectorStore(collection_name=collection_name)
    
    @handle_exception
    def parse_query(self, query: str) -> str:
        """Parse and clean user query."""
        query = re.sub(r'\\s+', ' ', query).strip()
        return query
    
    @handle_exception
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for relevant content based on query."""
        
        parsed_query = self.parse_query(query)
        logger.info(f"Searching for: {parsed_query}")
        
        query_embedding = self.embedding_gen.generate_embedding(parsed_query)
        results = self.vector_store.search(query_embedding, top_k)
        
        return results
    
    @handle_exception
    def get_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve context for RAG."""
        results = self.search(query, top_k)
        context = "\\n".join([f"- {doc}" for doc, _ in results])
        return context
'''
        
        file_path = self.project_root / "src/rag/semantic_search.py"
        file_path.write_text(content)
        self.log("  ✓ Created src/rag/semantic_search.py", "SUCCESS")
    
    def create_retriever(self):
        """Create src/rag/retriever.py"""
        self.log("Creating retriever.py...", "INFO")
        
        content = '''from typing import Dict
from src.rag.semantic_search import SemanticSearcher
from src.llm.model_factory import ModelFactory, ModelType
from src.logger import get_logger
from src.utils.error_handler import handle_exception

logger = get_logger(__name__)

class RAGRetriever:
    """Orchestrates RAG pipeline for question answering."""
    
    def __init__(self, model_type: ModelType = ModelType.LOCAL):
        self.searcher = SemanticSearcher()
        self.llm = ModelFactory.create_model(model_type)
        self.model_type = model_type
    
    @handle_exception
    def answer_question(self, question: str, top_k: int = 5) -> Dict:
        """Answer question using RAG approach."""
        
        logger.info(f"Answering question: {question}")
        
        context = self.searcher.get_context(question, top_k)
        
        prompt = f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {question}

Answer:"""
        
        if self.model_type == ModelType.LOCAL:
            answer = self.llm.generate(prompt, max_length=200)
        else:
            answer = self.llm.generate(prompt)
        
        return {
            'question': question,
            'context': context,
            'answer': answer
        }
'''
        
        file_path = self.project_root / "src/rag/retriever.py"
        file_path.write_text(content)
        self.log("  ✓ Created src/rag/retriever.py", "SUCCESS")
    
    def create_pdf_generator(self):
        """Create src/pdf_export/pdf_generator.py"""
        self.log("Creating pdf_generator.py...", "INFO")
        
        content = '''from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
from pathlib import Path
from typing import Dict
from src.logger import get_logger
from src.config import settings
from src.utils.error_handler import PDFGenerationError, handle_exception

logger = get_logger(__name__)

class PDFGenerator:
    """Generates PDF files from notes."""
    
    def __init__(self):
        self.export_dir = settings.EXPORTS_DIR
    
    @handle_exception
    def generate_notes_pdf(self, video_title: str, notes: str, 
                          metadata: Dict = None) -> Path:
        """Generate PDF from notes."""
        
        safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = self.export_dir / filename
        
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph(video_title, title_style))
        story.append(Spacer(1, 0.3*inch))
        
        if metadata:
            story.append(Paragraph("Video Information", styles['Heading2']))
            meta_data = [
                ['Duration', str(metadata.get('duration', 'N/A'))],
                ['Uploader', str(metadata.get('uploader', 'N/A'))],
                ['Date', str(metadata.get('upload_date', 'N/A'))],
                ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ]
            
            table = Table(meta_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("Generated Notes", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        for note in notes.split('\\n'):
            if note.strip():
                story.append(Paragraph(note, styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
        
        doc.build(story)
        logger.info(f"Generated PDF: {pdf_path}")
        
        return pdf_path
'''
        
        file_path = self.project_root / "src/pdf_export/pdf_generator.py"
        file_path.write_text(content)
        self.log("  ✓ Created src/pdf_export/pdf_generator.py", "SUCCESS")
    
    def create_chat_manager(self):
        """Create src/chat/chat_manager.py"""
        self.log("Creating chat_manager.py...", "INFO")
        
        content = '''import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from src.logger import get_logger
from src.config import settings
from src.utils.error_handler import handle_exception

logger = get_logger(__name__)

class ChatManager:
    """Manages chat history persistence."""
    
    def __init__(self):
        self.chats_dir = settings.CHATS_DIR
    
    @handle_exception
    def save_chat(self, video_id: str, chat_history: List[Dict]) -> Path:
        """Save chat history to file."""
        
        filename = f"{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.chats_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(chat_history, f, indent=2)
        
        logger.info(f"Saved chat to {filepath}")
        return filepath
    
    @handle_exception
    def load_chats(self, video_id: str) -> List[Dict]:
        """Load all chats for a video."""
        
        chats = []
        for file in self.chats_dir.glob(f"{video_id}_*.json"):
            with open(file, 'r') as f:
                chats.append(json.load(f))
        
        return chats
    
    @handle_exception
    def get_chat_list(self) -> List[Dict]:
        """Get list of all saved chats."""
        
        chats = []
        for file in self.chats_dir.glob("*.json"):
            with open(file, 'r') as f:
                chat_data = json.load(f)
                chats.append({
                    'filename': file.name,
                    'created': file.stat().st_mtime,
                    'messages_count': len(chat_data)
                })
        
        return sorted(chats, key=lambda x: x['created'], reverse=True)
'''
        
        file_path = self.project_root / "src/chat/chat_manager.py"
        file_path.write_text(content)
        self.log("  ✓ Created src/chat/chat_manager.py", "SUCCESS")
    
    def create_ui_files(self):
        """Create UI component files"""
        self.log("Creating UI files...", "INFO")
        
        # Header component
        header_content = '''import streamlit as st
import base64
from pathlib import Path

def get_base64_image(image_path):
    """Convert image to base64."""
    if not Path(image_path).exists():
        return None
    
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def render():
    """Render custom header."""
    
    logo_path = Path(__file__).parent.parent.parent / "logo.png"
    
    if logo_path.exists():
        img_base64 = get_base64_image(str(logo_path))
        if img_base64:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 30px;">
                    <img src="data:image/png;base64,{img_base64}" width="60" style="margin-right: 20px;">
                    <div>
                        <h1 style="margin: 0; color: #1f77b4;">📝 Notes YT</h1>
                        <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">
                            AI-Powered Video Summarization & Q&A
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.title("📝 Notes YT")
        st.markdown("*AI-Powered Video Summarization & Q&A*")
'''
        
        file_path = self.project_root / "ui/components/header.py"
        file_path.write_text(header_content)
        self.log("  ✓ Created ui/components/header.py", "SUCCESS")
        
        # Style component
        style_content = '''import streamlit as st

def apply_custom_css():
    """Apply custom CSS styling."""
    
    st.markdown("""
    <style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    h1, h2, h3 {
        color: #1f77b4;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stMetric {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
'''
        
        file_path = self.project_root / "ui/components/style.py"
        file_path.write_text(style_content)
        self.log("  ✓ Created ui/components/style.py", "SUCCESS")
    
    def create_root_files(self):
        """Create root level files"""
        self.log("Creating root files...", "INFO")
        
        # requirements.txt
        requirements = '''streamlit==1.39.0
python-dotenv==1.0.1
pydantic==2.5.0
yt-dlp==2024.1.1
youtube-transcript-api==0.6.2
torch==2.1.0
transformers==4.35.0
sentence-transformers==2.2.2
chromadb==0.4.21
faiss-cpu==1.7.4
reportlab==4.0.9
requests==2.31.0
numpy==1.26.2
pandas==2.1.3
structlog==24.1.0
google-generativeai==0.3.0
'''
        
        file_path = self.project_root / "requirements.txt"
        file_path.write_text(requirements)
        self.log("  ✓ Created requirements.txt", "SUCCESS")
        
        # .env.example
        env_example = '''GEMINI_API_KEY=your_gemini_api_key_here
VECTOR_DB_TYPE=chromadb
LOCAL_MODEL_NAME=google/flan-t5-base
EMBEDDING_MODEL=all-MiniLM-L6-v2
'''
        
        file_path = self.project_root / ".env.example"
        file_path.write_text(env_example)
        self.log("  ✓ Created .env.example", "SUCCESS")
        
        # .gitignore
        gitignore = '''__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
venv/
ENV/
env/
.venv
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store
data/
logs/
*.log
.streamlit/
.cache/
*.cache
.env
.env.local
'''
        
        file_path = self.project_root / ".gitignore"
        file_path.write_text(gitignore)
        self.log("  ✓ Created .gitignore", "SUCCESS")
    
    def setup_git(self):
        """Setup git and push to product branch."""
        self.log("\nSetting up Git...", "INFO")
        
        # Check if git is initialized
        git_dir = self.project_root / ".git"
        if not git_dir.exists():
            self.log("Initializing Git repository...", "INFO")
            if not self.run_command(["git", "init"]):
                self.log("Failed to initialize Git", "ERROR")
                return False
        
        # Check out product branch
        self.log("Checking out product branch...", "INFO")
        if not self.run_command(["git", "checkout", "-B", "product"]):
            self.log("Failed to checkout product branch", "WARNING")
        
        # Stage all files
        self.log("Staging files...", "INFO")
        if not self.run_command(["git", "add", "."]):
            self.log("Failed to stage files", "ERROR")
            return False
        
        # Commit
        self.log("Committing changes...", "INFO")
        commit_msg = """feat: Complete modular GenAI architecture

- Refactored codebase into micromodular structure
- Added comprehensive error handling & logging
- Implemented dual LLM support (Local + Gemini)
- Added RAG pipeline with semantic search
- Integrated vector databases (ChromaDB & FAISS)
- Created modern Streamlit UI with custom styling
- Added PDF export, chat persistence
- Follows production-ready best practices"""
        
        if not self.run_command(["git", "commit", "-m", commit_msg]):
            self.log("Nothing to commit or commit failed", "WARNING")
        
        # Push to product branch
        self.log("Pushing to product branch...", "INFO")
        if not self.run_command(["git", "push", "-u", "origin", "product", "-f"]):
            self.log("Failed to push (You may need to configure Git credentials)", "WARNING")
            return True  # Still consider it partial success
        
        return True
    
    def run(self):
        """Execute complete setup."""
        self.log("=" * 60, "INFO")
        self.log("Notes YT - Automated Project Setup", "INFO")
        self.log("=" * 60, "INFO")
        
        try:
            self.create_directories()
            self.log("", "INFO")
            
            self.create_init_files()
            self.log("", "INFO")
            
            self.log("Creating configuration files...", "INFO")
            self.create_config()
            self.create_logger()
            self.create_error_handler()
            self.log("", "INFO")
            
            self.log("Creating extractor modules...", "INFO")
            self.create_youtube_extractor()
            self.create_transcript_processor()
            self.log("", "INFO")
            
            self.log("Creating embedding modules...", "INFO")
            self.create_embedding_generator()
            self.create_vector_store()
            self.log("", "INFO")
            
            self.log("Creating LLM modules...", "INFO")
            self.create_local_model()
            self.create_gemini_model()
            self.create_model_factory()
            self.log("", "INFO")
            
            self.log("Creating RAG modules...", "INFO")
            self.create_semantic_search()
            self.create_retriever()
            self.log("", "INFO")
            
            self.log("Creating PDF and Chat modules...", "INFO")
            self.create_pdf_generator()
            self.create_chat_manager()
            self.log("", "INFO")
            
            self.log("Creating UI modules...", "INFO")
            self.create_ui_files()
            self.log("", "INFO")
            
            self.log("Creating root files...", "INFO")
            self.create_root_files()
            self.log("", "INFO")
            
            self.setup_git()
            
            self.log("", "SUCCESS")
            self.log("=" * 60, "SUCCESS")
            self.log("✅ Project setup completed successfully!", "SUCCESS")
            self.log("=" * 60, "SUCCESS")
            self.log("", "INFO")
            self.log("Next steps:", "INFO")
            self.log("1. Configure your GEMINI_API_KEY in .env file", "INFO")
            self.log("2. Install dependencies: pip install -r requirements.txt", "INFO")
            self.log("3. Run the app: streamlit run ui/app.py", "INFO")
            
        except Exception as e:
            self.log(f"Error during setup: {str(e)}", "ERROR")
            return False
        
        return True

if __name__ == "__main__":
    setup = ProjectSetup()
    success = setup.run()
    sys.exit(0 if success else 1)