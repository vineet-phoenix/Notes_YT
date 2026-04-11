from typing import Dict
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
