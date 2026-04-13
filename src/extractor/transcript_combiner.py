from typing import List, Dict, Tuple
from src.logger import get_logger
from src.utils.error_handler import TranscriptError, handle_exception

logger = get_logger(__name__)

class TranscriptCombiner:
    """Combines captions, audio transcripts, and scene information."""
    
    @staticmethod
    @handle_exception
    def align_transcripts(captions: List[Dict], audio_transcript: List[Dict]) -> List[Dict]:
        """Align captions with audio transcript by timestamp."""
        aligned = []
        
        caption_idx = 0
        for audio_seg in audio_transcript:
            audio_start = audio_seg['start']
            audio_end = audio_seg['end']
            
            matching_captions = []
            while caption_idx < len(captions):
                cap = captions[caption_idx]
                cap_start = cap['start']
                cap_end = cap['end']
                
                # Check if caption overlaps with audio segment
                if cap_end <= audio_start:
                    caption_idx += 1
                    continue
                elif cap_start >= audio_end:
                    break
                else:
                    matching_captions.append(cap['text'])
                    caption_idx += 1
            
            aligned.append({
                'start': audio_start,
                'end': audio_end,
                'duration': audio_end - audio_start,
                'audio_text': audio_seg['text'],
                'caption_text': ' '.join(matching_captions) if matching_captions else '',
                'combined_text': f"{audio_seg['text']} {' '.join(matching_captions)}".strip()
            })
        
        logger.info(f"Aligned {len(aligned)} transcript segments")
        return aligned
    
    @staticmethod
    @handle_exception
    def merge_with_scenes(aligned_transcripts: List[Dict], scenes: List[Dict]) -> List[Dict]:
        """Merge transcript segments with scene detection markers."""
        merged = []
        
        for segment in aligned_transcripts:
            segment_start = segment['start']
            segment_end = segment['end']
            
            # Find scenes within this segment
            related_scenes = [
                s for s in scenes 
                if segment_start <= s['timestamp'] <= segment_end
            ]
            
            segment['scenes'] = related_scenes
            segment['scene_count'] = len(related_scenes)
            merged.append(segment)
        
        logger.info(f"Merged {len(merged)} transcript segments with scene data")
        return merged
    
    @staticmethod
    @handle_exception
    def chunk_combined_transcripts(combined: List[Dict], chunk_size: int = 1200) -> List[Dict]:
        """Chunk combined transcript by character size."""
        chunks = []
        current_chunk = {
            'text': '',
            'combined_text': '',
            'start': None,
            'end': None,
            'duration': 0,
            'scenes': []
        }
        
        for segment in combined:
            text = segment['combined_text']
            audio_text = segment.get('audio_text', '')
            caption_text = segment.get('caption_text', '')
            
            if not current_chunk['text']:
                current_chunk['start'] = segment['start']
                current_chunk['text'] = text
                current_chunk['combined_text'] = text
            else:
                if len(current_chunk['text']) + len(text) < chunk_size:
                    current_chunk['text'] += ' ' + text
                    current_chunk['combined_text'] += '\n\n' + text
                    current_chunk['end'] = segment['end']
                    current_chunk['duration'] = current_chunk['end'] - current_chunk['start']
                    current_chunk['scenes'].extend(segment.get('scenes', []))
                else:
                    if current_chunk['text']:
                        chunks.append(current_chunk)
                    
                    current_chunk = {
                        'text': text,
                        'combined_text': text,
                        'start': segment['start'],
                        'end': segment['end'],
                        'duration': segment.get('duration', 0),
                        'scenes': segment.get('scenes', [])
                    }
        
        if current_chunk['text']:
            chunks.append(current_chunk)
        
        logger.info(f"Created {len(chunks)} combined chunks")
        return chunks
    
    @staticmethod
    @handle_exception
    def format_for_gemini(combined_chunks: List[Dict]) -> Tuple[str, str, List[Dict]]:
        """Format combined chunks for Gemini multimodal summarization.
        
        Returns:
            Tuple of (captions_text, audio_text, scenes_list)
        """
        captions_parts = []
        audio_parts = []
        all_scenes = []
        
        for chunk in combined_chunks:
            if chunk.get('text'):
                captions_parts.append(chunk['text'])
            if chunk.get('combined_text'):
                audio_parts.append(chunk['combined_text'])
            all_scenes.extend(chunk.get('scenes', []))
        
        captions_text = '\n\n'.join(captions_parts)
        audio_text = '\n\n'.join(audio_parts)
        
        # Deduplicate scenes
        unique_scenes = []
        seen_timestamps = set()
        for scene in all_scenes:
            ts = scene.get('timestamp')
            if ts not in seen_timestamps:
                unique_scenes.append(scene)
                seen_timestamps.add(ts)
        
        return captions_text, audio_text, unique_scenes
