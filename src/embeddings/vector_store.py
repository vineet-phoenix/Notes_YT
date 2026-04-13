import chromadb
import faiss
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
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
            
            # Clean metadata: remove empty lists and convert complex objects to strings
            clean_metadatas = []
            for doc in documents:
                clean_meta = {}
                for key, value in doc.items():
                    if key == 'scenes':
                        # Convert scenes list to string or skip if empty
                        if value and isinstance(value, list):
                            clean_meta[key] = f"{len(value)} scenes detected"
                        continue
                    elif value and not isinstance(value, (list, dict)):
                        # Only include non-empty scalar values
                        clean_meta[key] = str(value)[:500]  # Limit string length
                
                clean_metadatas.append(clean_meta if clean_meta else {'source': 'transcript'})
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=[doc.get('combined_text', doc.get('text', '')) for doc in documents],
                metadatas=clean_metadatas
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
    
    @handle_exception
    def clear(self):
        """Clear all embeddings from vector store."""
        if self.db_type == "chromadb":
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            logger.info("Cleared ChromaDB collection")
        
        elif self.db_type == "faiss":
            self.index = None
            self.metadata = []
            if self.index_path.exists():
                self.index_path.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()
            logger.info("Cleared FAISS index")