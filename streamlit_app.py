# streamlit_app.py — ChatGPT Style UI
import streamlit as st
from pathlib import Path
from rag_engine_streamlit import RagEngine

st.set_page_config(page_title="GUVI RAG Chatbot", layout="wide")

# Load Engine
@st.cache_resource
def get_engine():
    return RagEngine()

engine = get_engine()

# Session State
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts {role, text, sources}
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# -----------------------
# Chat Message Renderer
# -----------------------
def render_message(role, text):
    if role == "user":
        st.markdown(
            f"""
            <div style='display:flex; justify-content:flex-end; margin:8px 0;'>
                <div style='background:#0B93F6; color:white; padding:12px 16px; 
                            border-radius:16px 16px 4px 16px; max-width:70%; 
                            font-size:15px; line-height:1.4;'>
                    {text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='display:flex; justify-content:flex-start; margin:8px 0;'>
                <div style='background:#F1F1F1; color:#222; padding:12px 16px; 
                            border-radius:16px 16px 16px 4px; max-width:70%; 
                            font-size:15px; line-height:1.4;'>
                    {text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------
# INPUT PROCESSOR
# -----------------------
def process_input():
    user_query = st.session_state.user_input.strip()

    if user_query:
        # Add user message
        st.session_state.history.append({"role": "user", "text": user_query})

        # Generate bot reply
        with st.spinner("Bot is thinking..."):
            bot_reply = engine.answer(user_query)

        st.session_state.history.append({"role": "bot", "text": bot_reply})

    # Clear box safely
    st.session_state.user_input = ""


# -----------------------
# MAIN UI
# -----------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🤖 GUVI RAG Chatbot</h1>
    <p style='text-align:center; font-size:16px; color:gray;'>
        Ask questions about GUVI courses, blogs, and FAQs — answered using real GUVI data.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Chat area
chat_container = st.container()

with chat_container:
    for msg in st.session_state.history:
        render_message(msg["role"], msg["text"])

# Input Box (Fixed at bottom)
st.text_input(
    "Type your message…",
    key="user_input",
    on_change=process_input,
    placeholder="Ask me anything about GUVI…",
)

# Sidebar Info
with st.sidebar:
    st.markdown("## System Status")
    st.write(f"FAISS index: `{Path('faiss_store/index.faiss').exists()}`")
    st.write(f"Text chunks: `{Path('faiss_store/texts.json').exists()}`")
    st.write(f"Metadata: `{Path('faiss_store/meta.json').exists()}`")
    st.write(f"GGUF model folder: `{Path('models').exists()}`")

    st.markdown("## Tips")
    st.write("- Press **ENTER** to send your message.")
    st.write("- Refresh engine if you replace your GGUF model.")

    if st.button("Clear Chat"):
        st.session_state.history = []
        engine.memory = []
        st.success("Chat cleared.")
