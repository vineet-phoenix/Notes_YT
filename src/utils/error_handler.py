from src.logger import get_logger

logger = get_logger(__name__)

class NotesYTException(Exception):
    """Base exception for Notes YT."""
    pass

class VideoExtractionError(NotesYTException):
    """Raised when video extraction fails."""
    pass

class TranscriptError(NotesYTException):
    """Raised when transcript fetching fails."""
    pass

class ModelError(NotesYTException):
    """Raised when LLM inference fails."""
    pass

class EmbeddingError(NotesYTException):
    """Raised when embedding generation fails."""
    pass

class VectorStoreError(NotesYTException):
    """Raised when vector database operations fail."""
    pass

class PDFGenerationError(NotesYTException):
    """Raised when PDF generation fails."""
    pass

def handle_exception(func):
    """Decorator for robust error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise NotesYTException(f"{func.__name__} failed: {str(e)}")
    return wrapper
