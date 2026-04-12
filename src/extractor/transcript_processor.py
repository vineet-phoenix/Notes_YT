import cv2
import numpy as np
from typing import List, Dict, Tuple
from youtube_transcript_api import YouTubeTranscriptApi
from src.logger import get_logger
from src.config import settings
from src.utils.error_handler import TranscriptError, handle_exception

logger = get_logger(__name__)

class TranscriptProcessor:
    """Handles transcript extraction, chunking, and scene detection."""
    
    @handle_exception
    def get_transcript_with_timestamps(self, video_id: str) -> List[Dict]:
        """Fetch transcript with timestamp information."""
        try:
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id)
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
            # Handle both dict and FetchedTranscriptSnippet objects
            if isinstance(entry, dict):
                text = entry['text']
                start = entry['start']
                duration = entry['duration']
            else:
                # FetchedTranscriptSnippet object - use attribute access
                text = entry.text
                start = entry.start
                duration = entry.duration
            
            if not current_chunk['text']:
                current_chunk['start'] = start
            
            if len(current_chunk['text']) + len(text) < chunk_size:
                current_chunk['text'] += ' ' + text
                current_chunk['end'] = start + duration
                current_chunk['duration'] = current_chunk['end'] - current_chunk['start']
            else:
                if current_chunk['text']:
                    chunks.append(current_chunk)
                
                current_chunk = {
                    'text': text,
                    'start': start,
                    'end': start + duration,
                    'duration': duration
                }
        
        if current_chunk['text']:
            chunks.append(current_chunk)
        
        logger.info(f"Created {len(chunks)} chunks from transcript")
        return chunks
    
    @handle_exception
    def detect_scenes(self, video_path: str, threshold: float = None) -> List[Dict]:
        """Detect scene changes using OpenCV."""
        if threshold is None:
            threshold = settings.SCENE_DETECTION_THRESHOLD
        
        cap = cv2.VideoCapture(video_path)
        scenes = []
        
        ret, frame1 = cap.read()
        if not ret:
            raise TranscriptError("Could not read video file")
        
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame_count = 0
        
        while ret:
            ret, frame2 = cap.read()
            
            if not ret:
                break
            
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram difference
            hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
            
            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            
            if diff > threshold:
                fps = cap.get(cv2.CAP_PROP_FPS)
                timestamp = frame_count / fps if fps > 0 else 0
                scenes.append({
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'difference': diff
                })
            
            frame1 = frame2
            frame_count += 1
        
        cap.release()
        logger.info(f"Detected {len(scenes)} scenes")
        return scenes
    
    @handle_exception
    def merge_chunks_with_scenes(self, chunks: List[Dict], scenes: List[Dict]) -> List[Dict]:
        """Merge transcript chunks with detected scenes."""
        merged = []
        
        for chunk in chunks:
            chunk['scenes'] = [s for s in scenes if chunk['start'] <= s['timestamp'] <= chunk['end']]
            merged.append(chunk)
        
        return merged
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"