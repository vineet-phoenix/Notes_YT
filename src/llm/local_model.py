import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from src.logger import get_logger
from src.config import settings
from src.utils.error_handler import ModelError, handle_exception

logger = get_logger(__name__)

class LocalLLM:
    """Qwen3 0.6B local model for efficient text-only generation."""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = settings.LOCAL_MODEL_NAME
        
        self.device = settings.DEVICE
        self.model_name = model_name
        
        try:
            logger.info(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Load model with appropriate device handling
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                # CPU: don't use device_map, just load and move to device
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                ).to(self.device)
            
            self.model.eval()
            logger.info(f"Loaded local model: {model_name} on device: {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {str(e)}. Falling back to gpt2...")
            try:
                # Fallback to gpt2 if Qwen model fails
                self.model_name = "gpt2"
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                
                if self.device == "cuda":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        "gpt2",
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
                
                self.model.eval()
                logger.info(f"Loaded fallback model: gpt2 on device: {self.device}")
            except Exception as fallback_error:
                raise ModelError(f"Failed to load both {model_name} and fallback gpt2: {str(fallback_error)}")
    
    @handle_exception
    def generate(self, prompt: str, max_length: int = 200, 
                temperature: float = 0.7) -> str:
        """Generate text using Qwen3."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                               max_length=2048).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.1
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    @handle_exception
    def summarize_chunks(self, chunks: List[str]) -> List[str]:
        """Summarize multiple text chunks."""
        summaries = []
        
        for chunk in chunks:
            prompt = f"Summarize the following in bullet points:\n\n{chunk}\n\nSummary:"
            summary = self.generate(prompt, max_length=150)
            summaries.append(summary.strip())
        
        return summaries
