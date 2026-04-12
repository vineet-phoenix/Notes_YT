import google.genai as genai
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
