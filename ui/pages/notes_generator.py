import streamlit as st
from src.extractor.youtube_extractor import YouTubeExtractor
from src.extractor.transcript_processor import TranscriptProcessor
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
                
                # Get transcript
                processor = TranscriptProcessor()
                transcript_data = processor.get_transcript_with_timestamps(video_id)
                
                # Chunk transcript
                chunks = processor.chunk_transcript(transcript_data)
                
                # Generate embeddings
                embedding_gen = EmbeddingGenerator()
                chunk_texts = [c['text'] for c in chunks]
                embeddings = embedding_gen.generate_embeddings(chunk_texts)
                
                # Store embeddings
                vector_store = VectorStore(st.session_state.get('vector_db', 'chromadb'))
                vector_store.add_embeddings(embeddings, chunks, [f"chunk_{i}" for i in range(len(chunks))])
                
                # Generate notes
                model_type = ModelType.LOCAL if st.session_state.get('model_type', 'local') == 'local' else ModelType.GEMINI
                llm = ModelFactory.create_model(model_type)
                
                summaries = llm.summarize_chunks(chunk_texts)
                full_notes = "\n\n".join(summaries)
                
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
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Error in notes generation: {str(e)}")
    
    else:
        st.info("👈 Paste a YouTube URL to generate notes")