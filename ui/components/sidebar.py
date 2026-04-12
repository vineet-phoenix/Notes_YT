import streamlit as st

def render():
    """Render sidebar navigation."""
    
    with st.sidebar:
        st.header("⚙️ Navigation")
        
        page = st.radio(
            "Select a page:",
            ["Notes Generator", "Chatbot"],
            index=0
        )
        
        st.divider()
        
        st.header("📊 Settings")
        
        model_type = st.selectbox(
            "Select LLM Mode:",
            ["Local (Fast)", "Gemini (Quality)"],
            help="Local: Uses Flan-T5 (text-only, faster)\nGemini: Uses multimodal API (slower, higher quality)"
        )
        
        st.session_state.model_type = "local" if model_type == "Local (Fast)" else "gemini"
        
        vector_db = st.selectbox(
            "Vector Database:",
            ["ChromaDB", "FAISS"],
            help="ChromaDB: Better for production\nFAISS: Lightweight and fast"
        )
        
        st.session_state.vector_db = vector_db.lower()
        
        st.divider()
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        - Paste YouTube URLs (any format)
        - Notes generate automatically
        - Ask questions in the chatbot
        - Download notes as PDF
        """)
    
    return page