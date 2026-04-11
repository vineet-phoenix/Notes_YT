import torch
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
            prompt = f"Summarize the following in bullet points:\n\n{chunk}"
            summary = self.generate(prompt, max_length=150)
            summaries.append(summary)
        
        return summaries
