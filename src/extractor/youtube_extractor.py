import yt_dlp
import re
from typing import Optional, Dict
from src.config import settings
from src.logger import get_logger
from src.utils.error_handler import VideoExtractionError, handle_exception

logger = get_logger(__name__)

class YouTubeExtractor:
    """Extracts audio, video, and captions from YouTube URLs."""
    
    def __init__(self):
        self.video_download_dir = settings.VIDEO_DOWNLOAD_DIR
        
    @handle_exception
    def extract_video_id(self, url: str) -> str:
        """Extract 11-character video ID from YouTube URL."""
        patterns = [
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})",
            r"youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise VideoExtractionError(f"Could not extract video ID from URL: {url}")
    
    @handle_exception
    def get_video_metadata(self, url: str) -> Dict:
        """Get video metadata (title, duration, description)."""
        ydl_opts = {'quiet': True}
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Fetching metadata for {url}")
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title'),
                'duration': info.get('duration'),
                'description': info.get('description'),
                'uploader': info.get('uploader'),
                'upload_date': info.get('upload_date'),
            }
