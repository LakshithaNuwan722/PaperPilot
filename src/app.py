"""
✈️ PaperPilot: Smart Agentic UI (Production Ready)
==================================================
A modern, AI-centric interface with custom styling and smart reasoning display.
"""

import os
import tempfile
import json
import time
from datetime import datetime
import streamlit as st
from src.config import DATA_DIR
from src.logger import setup_logger
from src.step1_load_and_chunk import load_pdf, chunk_documents
from src.step2_build_vectorstore import build_vectorstore
from src.step5_agent import build_agent

# --- Initialization ---
logger = setup_logger("Streamlit_UI")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
LOG_FILE = "logs/interactions.jsonl"
os.makedirs("logs", exist_ok=True)

# --- Page Config & Styling ---
st.set_page_config(
    page_title="PaperPilot AI ✈️",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Smart AI" look
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E1E1E;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Chat Input Styling */
    .stChatFloatingInputContainer {
        padding-bottom: 2rem;
    }

    /* Status Box (Reasoning) */
    .stStatusWidget {
        border-radius: 10px;
        border: 1px solid #d1d1d1;
        background-color: #ffffff;
    }

    /* Custom Message Styling */
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Logic: Agent Initialization ---
@st.cache_resource(show_spinner=False)
def init_system(file_bytes):
    with st.spinner("🧠 Processing document with AI..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            path = tmp.name
        
        docs = load_pdf(path)
        chunks = chunk_documents(docs)
        build_vectorstore(chunks)
        return build_agent()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2807/2807213.png", width=100)
    st.markdown("## **PaperPilot AI**")
    st.caption("v2.0 - Agentic RAG Assistant")
    
    st.divider()
    
    uploaded = st.file_uploader("📂 Upload Research PDF", type="pdf", help="Upload a document to start the AI analysis.")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.divider()
    st.markdown("### 🛠️ Capabilities")
    st.markdown("""
    - **Deep Document Search**
    - **Math & Calculations**
    - **Real-time Web Search**
    - **Smart Summarization**
    """)

# --- Main App Interface ---
if not uploaded:
    # Welcome Hero Section
    st.markdown('<div class="main-header">✈️ PaperPilot AI</div>', unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #555;'>Your intelligent partner in document research.</h4>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🔍 **Analyze**\nUpload complex PDFs and get instant answers.")
    with col2:
        st.success("🧮 **Calculate**\nHandle mathematical data within documents.")
    with col3:
        st.warning("🌐 **Explore**\nGo beyond the PDF with live web search.")
    
    st.image("https://img.freepik.com/free-vector/ai-powered-content-creation-isometric-composition-with-robot-character-creating-content-computer-screen-vector-illustration_1284-82441.jpg?t=st=1716383321~exp=1716386921~hmac=6b7c8e9b6e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e", use_column_width=True)

else:
    # Initialize System
    st.session_state.agent = init_system(uploaded.getvalue())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            start_time = time.time()
            
            # Smart Reasoning Display (Agent Thoughts)
            with st.status("🚀 **Agent is thinking...**", expanded=True) as status:
                try:
                    full_response = ""
                    for step in st.session_state.agent.stream({"messages": [("user", prompt)]}, stream_mode="values"):
                        msg = step["messages"][-1]
                        
                        # Displaying Tool Calls specifically
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                if tc['name'] == 'search_documents':
                                    st.write("🔍 *Searching document context...*")
                                elif tc['name'] == 'calculator':
                                    st.write("🧮 *Performing mathematical calculations...*")
                                elif tc['name'] == 'web_search':
                                    st.write("🌐 *Searching the live web for updates...*")
                                elif tc['name'] == 'summarize_document':
                                    st.write("📝 *Synthesizing a summary...*")
                        
                        full_response = msg.content
                    
                    status.update(label="✅ **Analysis Complete**", state="complete", expanded=False)
                    st.markdown(full_response)
                    
                    # Log interaction
                    duration = time.time() - start_time
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    log_data = {
                        "timestamp": datetime.now().isoformat(),
                        "question": prompt,
                        "answer": full_response,
                        "latency": round(duration, 2)
                    }
                    with open(LOG_FILE, "a") as f:
                        f.write(json.dumps(log_data) + "\n")
                        
                except Exception as e:
                    status.update(label="❌ **Error Occurred**", state="error")
                    st.error(f"Agent encountered an issue: {e}")
                    logger.error(f"UI Error: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 PaperPilot AI. Built for Smart Research.")
