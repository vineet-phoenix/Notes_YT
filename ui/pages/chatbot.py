import streamlit as st
from src.rag.retriever import RAGRetriever
from src.llm.model_factory import ModelType
from src.chat.chat_manager import ChatManager
from src.logger import get_logger

logger = get_logger(__name__)

def render():
    """Render chatbot page."""
    
    st.header("💬 Ask Questions About the Video")
    
    if not st.session_state.get('notes'):
        st.warning("⚠️ Please generate notes first from a YouTube video!")
        return
    
    # Chat display
    st.markdown("### Chat History")
    
    for msg in st.session_state.get('chat_history', []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask something about the video..."):
        # Add user message
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    model_type = ModelType.LOCAL if st.session_state.get('model_type', 'local') == 'local' else ModelType.GEMINI
                    retriever = RAGRetriever(model_type)
                    
                    result = retriever.answer_question(user_input, top_k=5)
                    response = result['answer']
                    
                    st.markdown(response)
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Error in chatbot: {str(e)}")
    
    # Save chat option
    if st.session_state.get('chat_history'):
        if st.button("💾 Save Chat"):
            chat_mgr = ChatManager()
            chat_mgr.save_chat(st.session_state.get('video_id', 'unknown'), st.session_state.chat_history)
            st.success("✅ Chat saved!")