import whisper
import numpy as np
from typing import List, Dict
from src.logger import get_logger
from src.config import settings
from src.utils.error_handler import TranscriptError, handle_exception

logger = get_logger(__name__)

class AudioTranscriber:
    """Transcribes audio using OpenAI Whisper model."""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = settings.WHISPER_MODEL
        
        try:
            self.model = whisper.load_model(model_name)
            logger.info(f"Loaded Whisper model: {model_name}")
        except Exception as e:
            raise TranscriptError(f"Failed to load Whisper model: {str(e)}")
    
    @handle_exception
    def transcribe_audio(self, audio_path: str) -> List[Dict]:
        """Transcribe audio file to text with timestamps."""
        try:
            logger.info(f"Transcribing audio from: {audio_path}")
            result = self.model.transcribe(audio_path)
            
            # Extract segments with timestamps
            segments = []
            for segment in result['segments']:
                segments.append({
                    'text': segment['text'].strip(),
                    'start': segment['start'],
                    'end': segment['end'],
                    'duration': segment['end'] - segment['start']
                })
            
            logger.info(f"Successfully transcribed {len(segments)} segments")
            return segments
        except Exception as e:
            raise TranscriptError(f"Failed to transcribe audio: {str(e)}")
    
    @handle_exception
    def transcribe_from_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> List[Dict]:
        """Transcribe audio from numpy array with timestamps."""
        try:
            logger.info(f"Transcribing audio array (sr={sample_rate})")
            result = self.model.transcribe(audio_array.astype(np.float32))
            
            segments = []
            for segment in result['segments']:
                segments.append({
                    'text': segment['text'].strip(),
                    'start': segment['start'],
                    'end': segment['end'],
                    'duration': segment['end'] - segment['start']
                })
            
            logger.info(f"Successfully transcribed {len(segments)} segments from array")
            return segments
        except Exception as e:
            raise TranscriptError(f"Failed to transcribe audio array: {str(e)}")
