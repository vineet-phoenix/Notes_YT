import streamlit as st
from src.extractor import extract_video_id, get_transcript
from src.nlp_model import VideoAIAssistant
import base64

# --- HELPER FUNCTION FOR TITLE ICON ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- PAGE CONFIGURATION ---
# We use the PNG file as the browser tab icon (page_icon)
st.set_page_config(
    page_title="Notes YT", 
    page_icon="logo.png", 
    layout="centered"
)

# --- APP LOGO ---
# This adds the logo to the top of the sidebar automatically
st.logo("logo.png", icon_image="logo.png")

# --- CUSTOM TITLE WITH ICON ---
# We use HTML to place the logo inside the title line
img_base64 = get_base64_image("logo.png")
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{img_base64}" width="50" style="margin-right: 15px;">
        <h1 style="margin: 0;">Notes YT</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("Generate smart notes from any YouTube video and ask questions about it!")

# --- LOAD MODEL (CACHED) ---
@st.cache_resource
def load_ai_model():
    """Caches the model so it only loads into GPU/CPU memory once."""
    return VideoAIAssistant()

ai_assistant = load_ai_model()

# --- SESSION STATE INITIALIZATION ---
# We use session state to remember variables across user interactions (like asking a new question)
if "summary_notes" not in st.session_state:
    st.session_state.summary_notes = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR: VIDEO INPUT ---
with st.sidebar:
    st.header("Enter Video Details")
    url_input = st.text_input("Paste YouTube URL here:")
    process_btn = st.button("Generate Notes", type="primary")

    if process_btn and url_input:
        with st.spinner("Fetching transcript and generating notes (this might take a minute)..."):
            video_id = extract_video_id(url_input)
            if not video_id:
                st.error("Invalid YouTube URL.")
            else:
                transcript = get_transcript(video_id)
                if transcript.startswith("Error"):
                    st.error(transcript)
                else:
                    # Generate notes and save to session state
                    notes = ai_assistant.summarize_video(transcript)
                    st.session_state.summary_notes = "- " + notes
                    # Clear chat history for a new video
                    st.session_state.chat_history = []
                    st.success("Notes generated successfully!")

# --- MAIN WINDOW: NOTES DISPLAY ---
if st.session_state.summary_notes:
    with st.expander("📝 View AI Generated Notes", expanded=False):
        st.markdown(st.session_state.summary_notes)

    st.divider()
    st.subheader("💬 Ask questions about this video")
    
    # --- CHAT INTERFACE ---
    # Render previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input box at the bottom
    if user_question := st.chat_input("Ask something like: 'What was the main conclusion?'"):
        # 1. Display User Message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # 2. Add to history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # 3. Generate AI Answer using the summary as Context (RAG approach)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ai_assistant.answer_question(
                    context=st.session_state.summary_notes, 
                    question=user_question
                )
                st.markdown(answer)
                
        # 4. Add AI answer to history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Paste a URL in the sidebar and click 'Generate Notes' to begin.")
