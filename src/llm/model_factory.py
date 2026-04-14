from enum import Enum
from src.llm.local_model import LocalLLM
from src.llm.gemini_model import GeminiLLM
from src.logger import get_logger
from src.config import settings

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

        # Refresh local model if config changed during a live Streamlit session.
        if model_type == ModelType.LOCAL and model_type in ModelFactory._instances:
            cached = ModelFactory._instances[model_type]
            cached_name = getattr(cached, "model_name", None)
            if cached_name != settings.LOCAL_MODEL_NAME:
                logger.info(
                    f"Local model changed ({cached_name} -> {settings.LOCAL_MODEL_NAME}); reloading instance"
                )
                del ModelFactory._instances[model_type]

        # Refresh gemini model if config changed
        if model_type == ModelType.GEMINI and model_type in ModelFactory._instances:
            cached = ModelFactory._instances[model_type]
            cached_name = getattr(cached, "model_name", None)
            if cached_name != settings.GEMINI_MODEL_NAME:
                logger.info(
                    f"Gemini model changed ({cached_name} -> {settings.GEMINI_MODEL_NAME}); reloading instance"
                )
                del ModelFactory._instances[model_type]

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
