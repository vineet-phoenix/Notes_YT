from typing import List, Dict
from youtube_transcript_api import YouTubeTranscriptApi
from src.logger import get_logger
from src.config import settings
from src.utils.error_handler import TranscriptError, handle_exception

logger = get_logger(__name__)

class TranscriptProcessor:
    """Handles transcript extraction and chunking."""
    
    @handle_exception
    def get_transcript_with_timestamps(self, video_id: str) -> List[Dict]:
        """Fetch transcript with timestamp information."""
        try:
            api = YouTubeTranscriptApi()
            transcript = api.fetch_transcript(video_id)
            logger.info(f"Successfully fetched transcript for video {video_id}")
            return transcript
        except Exception as e:
            raise TranscriptError(f"Failed to fetch transcript: {str(e)}")
    
    @handle_exception
    def chunk_transcript(self, transcript: List[Dict], chunk_size: int = None) -> List[Dict]:
        """Chunk transcript by character size while preserving timestamps."""
        if chunk_size is None:
            chunk_size = settings.YOUTUBE_CHUNK_SIZE
        
        chunks = []
        current_chunk = {
            'text': '',
            'start': None,
            'end': None,
            'duration': 0
        }
        
        for entry in transcript:
            text = entry['text']
            
            if not current_chunk['text']:
                current_chunk['start'] = entry['start']
            
            if len(current_chunk['text']) + len(text) < chunk_size:
                current_chunk['text'] += ' ' + text
                current_chunk['end'] = entry['start'] + entry['duration']
                current_chunk['duration'] = current_chunk['end'] - current_chunk['start']
            else:
                if current_chunk['text']:
                    chunks.append(current_chunk)
                
                current_chunk = {
                    'text': text,
                    'start': entry['start'],
                    'end': entry['start'] + entry['duration'],
                    'duration': entry['duration']
                }
        
        if current_chunk['text']:
            chunks.append(current_chunk)
        
        logger.info(f"Created {len(chunks)} chunks from transcript")
        return chunks
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
