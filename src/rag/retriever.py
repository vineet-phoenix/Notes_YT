from typing import Dict
from src.rag.semantic_search import SemanticSearcher
from src.llm.model_factory import ModelFactory, ModelType
from src.logger import get_logger
from src.utils.error_handler import handle_exception

logger = get_logger(__name__)

class RAGRetriever:
    """Orchestrates RAG pipeline for question answering using Qwen3 or Gemini."""
    
    def __init__(self, model_type: ModelType = ModelType.LOCAL):
        self.searcher = SemanticSearcher()
        self.llm = ModelFactory.create_model(model_type)
        self.model_type = model_type
        logger.info(f"Initialized RAG with model: {model_type.value}")
    
    @handle_exception
    def answer_question(self, question: str, top_k: int = 5) -> Dict:
        """Answer question using RAG approach with context retrieval."""
        
        logger.info(f"Answering question: {question}")
        
        context = self.searcher.get_context(question, top_k)
        
        prompt = f"""Based on the following context from video transcripts, answer the question concisely and accurately.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        if self.model_type == ModelType.LOCAL:
            # Use Qwen3 for efficient local inference
            answer = self.llm.generate(prompt, max_new_tokens=200, temperature=0.7)
        else:
            # Use Gemini for high-quality answers
            answer = self.llm.generate(prompt)
        
        return {
            'question': question,
            'context': context,
            'answer': answer.strip(),
            'model': self.model_type.value
        }
