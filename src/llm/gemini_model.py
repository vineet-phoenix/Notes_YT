import google.genai as genai
from typing import List, Dict
from src.logger import get_logger
from src.config import settings
from src.utils.error_handler import ModelError, handle_exception

logger = get_logger(__name__)

class GeminiLLM:
    """Google Gemini-3 API for multimodal generation."""
    
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = settings.GEMINI_API_KEY
        
        if not api_key:
            raise ModelError("GEMINI_API_KEY not set in environment variables")
        
        try:
            self.client = genai.Client(api_key=api_key)
            self.model_name = settings.GEMINI_MODEL_NAME
            logger.info(f"Initialized Gemini API with model: {settings.GEMINI_MODEL_NAME}")
        except Exception as e:
            raise ModelError(f"Failed to initialize Gemini API: {str(e)}")
    
    @handle_exception
    def generate(self, prompt: str, images: List = None) -> str:
        """Generate text with optional image inputs."""
        
        if images:
            content = [prompt] + images
        else:
            content = prompt
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=content
        )
        return response.text

    @handle_exception
    def summarize_with_images(self, text: str, images: List = None) -> str:
        """Generate high-quality summary with multimodal context."""
        
        prompt = f"""Analyze the following transcript and any provided images to create 
comprehensive, well-structured notes. Focus on:
- Key concepts and main ideas
- Important definitions and explanations
- Visual insights from images
- Timestamps and scene descriptions

Transcript:
{text}

Please provide detailed, organized notes."""
        
        return self.generate(prompt, images)
    
    @handle_exception
    def summarize_chunks(self, chunks: List[str]) -> List[str]:
        """Summarize multiple text chunks."""
        summaries = []
        
        for chunk in chunks:
            prompt = f"Summarize the following in bullet points:\n\n{chunk}\n\nSummary:"
            summary = self.generate(prompt)
            summaries.append(summary.strip())
        
        return summaries
    
    @handle_exception
    def summarize_multimodal(self, captions: str, audio_transcript: str, scenes: List[Dict] = None) -> str:
        """Generate comprehensive summary from captions, audio transcript, and scene info."""
        
        scene_info = ""
        if scenes:
            scene_info = "\n\nDetected Scenes:\n"
            for i, scene in enumerate(scenes[:10]):  # Top 10 scenes
                scene_info += f"- Scene {i+1} at {scene.get('timestamp', 'N/A')}s (Change: {scene.get('difference', 0):.2f})\n"
        
        prompt = f"""Analyze the following multimodal content to create comprehensive, well-structured notes.

CAPTIONS:
{captions}

AUDIO TRANSCRIPT:
{audio_transcript}
{scene_info}

Please provide detailed, organized notes that:
1. Highlight key concepts from both sources
2. Note where captions and audio differ
3. Reference important scene changes
4. Structure the information logically
5. Include timestamps where relevant

COMPREHENSIVE NOTES:"""
        
        return self.generate(prompt)