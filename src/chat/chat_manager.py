import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from src.logger import get_logger
from src.config import settings
from src.utils.error_handler import handle_exception

logger = get_logger(__name__)

class ChatManager:
    """Manages chat history persistence."""
    
    def __init__(self):
        self.chats_dir = settings.CHATS_DIR
    
    @handle_exception
    def save_chat(self, video_id: str, chat_history: List[Dict]) -> Path:
        """Save chat history to file."""
        
        filename = f"{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.chats_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(chat_history, f, indent=2)
        
        logger.info(f"Saved chat to {filepath}")
        return filepath
    
    @handle_exception
    def load_chats(self, video_id: str) -> List[Dict]:
        """Load all chats for a video."""
        
        chats = []
        for file in self.chats_dir.glob(f"{video_id}_*.json"):
            with open(file, 'r') as f:
                chats.append(json.load(f))
        
        return chats
    
    @handle_exception
    def get_chat_list(self) -> List[Dict]:
        """Get list of all saved chats."""
        
        chats = []
        for file in self.chats_dir.glob("*.json"):
            with open(file, 'r') as f:
                chat_data = json.load(f)
                chats.append({
                    'filename': file.name,
                    'created': file.stat().st_mtime,
                    'messages_count': len(chat_data)
                })
        
        return sorted(chats, key=lambda x: x['created'], reverse=True)
