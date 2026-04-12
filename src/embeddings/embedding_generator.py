import numpy as np
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
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.model.get_embedding_dimension()