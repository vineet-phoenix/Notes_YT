import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pages import notes_generator, chatbot
from components import header, sidebar, style

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Notes YT",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
style.apply_custom_css()

# --- SESSION STATE INIT ---
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "notes" not in st.session_state:
    st.session_state.notes = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_type" not in st.session_state:
    st.session_state.model_type = "local"
if "vector_db" not in st.session_state:
    st.session_state.vector_db = "chromadb"

# --- HEADER ---
header.render()

# --- SIDEBAR ---
page = sidebar.render()

# --- PAGE ROUTING ---
if page == "Notes Generator":
    notes_generator.render()
elif page == "Chatbot":
    chatbot.render()