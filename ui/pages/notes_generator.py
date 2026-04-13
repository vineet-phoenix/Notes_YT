import streamlit as st
from pathlib import Path
from src.extractor.youtube_extractor import YouTubeExtractor
from src.extractor.transcript_processor import TranscriptProcessor
from src.extractor.audio_transcriber import AudioTranscriber
from src.extractor.transcript_combiner import TranscriptCombiner
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import VectorStore
from src.llm.model_factory import ModelFactory, ModelType
from src.pdf_export.pdf_generator import PDFGenerator
from src.logger import get_logger

logger = get_logger(__name__)

def render():
    """Render notes generator page."""
    
    st.header("🎬 Generate Notes from YouTube")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url = st.text_input("Paste YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    with col2:
        generate_btn = st.button("Generate", type="primary", use_container_width=True)
    
    if generate_btn and url:
        with st.spinner("🔄 Processing video..."):
            try:
                # Extract video info
                st.info("📺 Extracting video information...")
                extractor = YouTubeExtractor()
                video_id = extractor.extract_video_id(url)
                metadata = extractor.get_video_metadata(url)
                
                st.session_state.video_id = video_id
                
                # Display video metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Title", metadata['title'][:30] + "..." if len(str(metadata.get('title', ''))) > 30 else metadata.get('title', 'N/A'))
                with col2:
                    duration = metadata.get('duration', 0)
                    st.metric("Duration", f"{int(duration)//60} mins" if duration else "N/A")
                with col3:
                    uploader = metadata.get('uploader', 'N/A')
                    st.metric("Uploader", str(uploader)[:20] + "..." if len(str(uploader)) > 20 else uploader)
                
                st.divider()
                
                # Get captions
                st.info("📝 Extracting captions...")
                processor = TranscriptProcessor()
                caption_data = processor.get_transcript_with_timestamps(video_id)
                caption_chunks = processor.chunk_transcript(caption_data)
                
                # Stream and extract audio
                st.info("🎵 Streaming video and extracting audio...")
                video_path = extractor.stream_video_to_file(url)
                
                # Transcribe audio using Whisper
                st.info("🎤 Transcribing audio with Whisper...")
                transcriber = AudioTranscriber()
                audio_transcript = transcriber.transcribe_audio(video_path)
                
                # Detect scenes
                st.info("🎬 Detecting scene changes...")
                scenes = processor.detect_scenes(video_path)
                
                st.divider()
                st.success("✅ Content extracted successfully!")
                
                # Combine transcripts
                st.info("🔗 Aligning captions with audio transcript...")
                combined_segments = TranscriptCombiner.align_transcripts(caption_chunks, audio_transcript)
                combined_with_scenes = TranscriptCombiner.merge_with_scenes(combined_segments, scenes)
                combined_chunks = TranscriptCombiner.chunk_combined_transcripts(combined_with_scenes)
                
                # Generate embeddings for combined content
                st.info("📊 Generating embeddings...")
                embedding_gen = EmbeddingGenerator()
                chunk_texts = [c['combined_text'] for c in combined_chunks]
                embeddings = embedding_gen.generate_embeddings(chunk_texts)
                
                # Store embeddings in vector DB
                vector_store = VectorStore(st.session_state.get('vector_db', 'chromadb'))
                vector_store.add_embeddings(embeddings, combined_chunks, [f"chunk_{i}" for i in range(len(combined_chunks))])
                
                st.divider()
                
                # Generate notes based on selected model
                model_type = ModelType.LOCAL if st.session_state.get('model_type', 'local') == 'local' else ModelType.GEMINI
                
                if model_type == ModelType.LOCAL:
                    # Use Qwen3 for local summarization
                    st.info("📝 Generating notes with Qwen3 (Local)...")
                    llm = ModelFactory.create_model(model_type)
                    summaries = llm.summarize_chunks(chunk_texts)
                    full_notes = "\n\n".join(summaries)
                else:
                    # Use Gemini for multimodal summarization
                    st.info("✨ Generating comprehensive notes with Gemini-3...")
                    llm = ModelFactory.create_model(model_type)
                    
                    # Format content for Gemini
                    captions_text, audio_text, scene_list = TranscriptCombiner.format_for_gemini(combined_chunks)
                    
                    # Get multimodal summary
                    full_notes = llm.summarize_multimodal(captions_text, audio_text, scene_list)
                
                st.session_state.notes = full_notes
                
                # Display notes
                st.success("✅ Notes generated successfully!")
                
                with st.expander("📝 View Notes", expanded=True):
                    st.markdown(full_notes)
                
                # Download options
                col1, col2 = st.columns(2)
                
                with col1:
                    pdf_gen = PDFGenerator()
                    pdf_path = pdf_gen.generate_notes_pdf(metadata['title'], full_notes, metadata)
                    
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_file,
                            file_name=pdf_path.name,
                            mime="application/pdf"
                        )
                
                with col2:
                    st.download_button(
                        label="📄 Download Notes (TXT)",
                        data=full_notes,
                        file_name=f"{metadata['title']}_notes.txt",
                        mime="text/plain"
                    )
                
                # Debug info
                with st.expander("📊 Processing Statistics"):
                    st.write(f"- Captions extracted: {len(caption_chunks)} segments")
                    st.write(f"- Audio segments transcribed: {len(audio_transcript)} segments")
                    st.write(f"- Scenes detected: {len(scenes)}")
                    st.write(f"- Combined segments: {len(combined_segments)}")
                    st.write(f"- Final chunks for summarization: {len(combined_chunks)}")
                    st.write(f"- Vector DB type: {st.session_state.get('vector_db', 'chromadb')}")
                    st.write(f"- Model used: {model_type.value}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Error in notes generation: {str(e)}", exc_info=True)
    
    else:
        st.info("👈 Paste a YouTube URL to generate notes")