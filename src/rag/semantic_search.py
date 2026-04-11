import re
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
        query = re.sub(r'\s+', ' ', query).strip()
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
        context = "\n".join([f"- {doc}" for doc, _ in results])
        return context
