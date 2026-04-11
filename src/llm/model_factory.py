from enum import Enum
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
