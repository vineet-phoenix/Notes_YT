import re
from src.logger import get_logger
from src.utils.error_handler import VideoExtractionError

logger = get_logger(__name__)

class URLValidator:
    """Validates YouTube URLs."""
    
    @staticmethod
    def is_valid_youtube_url(url: str) -> bool:
        """Check if URL is a valid YouTube URL."""
        patterns = [
            r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
            r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
            r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
            r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
        ]
        
        for pattern in patterns:
            if re.search(pattern, url):
                return True
        
        return False

class TextValidator:
    """Validates text inputs."""
    
    @staticmethod
    def is_valid_query(query: str, min_length: int = 3) -> bool:
        """Check if query is valid."""
        if not query or len(query.strip()) < min_length:
            return False
        return True