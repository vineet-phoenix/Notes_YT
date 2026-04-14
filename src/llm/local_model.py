import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from src.logger import get_logger
from src.config import settings
from src.utils.error_handler import ModelError, handle_exception

logger = get_logger(__name__)

class LocalLLM:
    """Qwen3 0.6B local model for efficient text-only generation."""

    def _load_model(self, model_name: str, force_download: bool = False):
        """Load tokenizer and model, optionally forcing a fresh download."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            force_download=force_download,
            local_files_only=False,
        )

        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.float16,
                device_map="auto",
                force_download=force_download,
                local_files_only=False,
            )
        else:
            # CPU: avoid device_map dependency path
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.float32,
                force_download=force_download,
                local_files_only=False,
            ).to(self.device)

        self.model.eval()
        
        # Set up terminators for instruction-tuned models
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ]
        # Filter out Nones if token doesn't exist
        self.terminators = [t for t in self.terminators if t is not None]
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = settings.LOCAL_MODEL_NAME
        
        self.device = settings.DEVICE
        self.model_name = model_name
        
        try:
            logger.info(f"Loading model: {model_name}")
            self._load_model(model_name, force_download=False)
            logger.info(f"Loaded local model: {model_name} on device: {self.device}")
        except Exception as first_error:
            logger.warning(
                f"Initial load failed for {model_name}: {str(first_error)}. "
                "Retrying with force_download=True..."
            )
            try:
                self._load_model(model_name, force_download=True)
                logger.info(
                    f"Loaded local model after forced download: {model_name} on device: {self.device}"
                )
            except Exception as second_error:
                logger.warning(
                    f"Forced download also failed for {model_name}: {str(second_error)}. "
                    "Falling back to gpt2..."
                )
                try:
                    self.model_name = "gpt2"
                    self._load_model("gpt2", force_download=False)
                    logger.info(f"Loaded fallback model: gpt2 on device: {self.device}")
                except Exception as fallback_error:
                    raise ModelError(
                        f"Failed to load {model_name} (normal + forced download) and fallback gpt2: {str(fallback_error)}"
                    )
    
    @handle_exception
    def generate(self, prompt: str, max_new_tokens: int = 200, 
                temperature: float = 0.7) -> str:
        """Generate text using Qwen3."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                               max_length=2048).to(self.device)
        input_len = inputs["input_ids"].shape[1]
        do_sample = temperature > 0
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=do_sample,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.terminators if hasattr(self, 'terminators') else self.tokenizer.eos_token_id,
            )

        # Causal LM returns prompt + generated text. Keep only new tokens.
        generated_ids = outputs[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    @handle_exception
    def summarize_chunks(self, chunks: List[str]) -> List[str]:
        """Summarize multiple text chunks."""
        summaries = []
        
        for chunk in chunks:
            messages = [
                {"role": "system", "content": "You are an expert summarizer. Extract the most important points from the video transcript chunk. Output ONLY bullet points."},
                {"role": "user", "content": f"Create concise bullet-point notes from this transcript chunk:\n\n{chunk}"}
            ]
            
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = (
                    "Create concise bullet-point notes from this transcript chunk. "
                    "Return only bullet points. Do not repeat the prompt.\n\n"
                    f"Chunk:\n{chunk}\n\n"
                    "Bullet Notes:"
                )
                
            summary = self.generate(prompt, max_new_tokens=256, temperature=0.2)

            # Last-resort cleanup in case a model still echoes prompt text.
            cleaned = summary.replace("Summarize the following in bullet points:", "").strip()
            summaries.append(cleaned)
        
        return summaries
