import streamlit as st
import base64
from pathlib import Path

def get_base64_image(image_path):
    """Convert image to base64 for embedding."""
    if not Path(image_path).exists():
        return None
    
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def render():
    """Render custom header with logo."""
    
    logo_path = Path(__file__).parent.parent.parent / "logo.png"
    
    if logo_path.exists():
        img_base64 = get_base64_image(str(logo_path))
        if img_base64:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 30px;">
                    <img src="data:image/png;base64,{img_base64}" width="60" style="margin-right: 20px;">
                    <div>
                        <h1 style="margin: 0; color: #1f77b4;">📝 Notes YT</h1>
                        <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">
                            AI-Powered Video Summarization & Q&A
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.title("Notes YT")
            st.markdown("*AI-Powered Video Summarization & Q&A*")
    else:
        st.title("Notes YT")
        st.markdown("*AI-Powered Video Summarization & Q&A*")