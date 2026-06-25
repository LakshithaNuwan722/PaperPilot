"""
Final Polished Web UI (Production Ready)
"""

import os
import tempfile
import streamlit as st
from src.config import DATA_DIR
from src.logger import setup_logger
from src.step1_load_and_chunk import load_pdf, chunk_documents
from src.step2_build_vectorstore import build_vectorstore
from src.step5_agent import build_agent

logger = setup_logger("Streamlit_UI")
os.environ["ANONYMIZED_TELEMETRY"] = "False"

st.set_page_config(page_title="PaperPilot Agent ✈️", layout="wide")
st.title("✈️ PaperPilot")

# Sidebar for PDF Processing
with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Upload PDF", type="pdf")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource(show_spinner="Processing document...")
def init_agent(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    
    logger.info("Initializing system from UI")
    docs = load_pdf(path)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    return build_agent()

if uploaded:
    agent = init_agent(uploaded.getvalue())
    
    # Show History
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Thinking...", expanded=False) as status:
                try:
                    # Collect reasoning steps
                    full_response = ""
                    for step in agent.stream({"messages": [("user", prompt)]}, stream_mode="values"):
                        msg = step["messages"][-1]
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                st.write(f"🔧 Using tool: {tc['name']}")
                        full_response = msg.content
                    
                    status.update(label="Complete!", state="complete")
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    logger.error(f"UI Error: {e}")
                    st.error(f"Error: {e}")
else:
    st.info("Upload a PDF to start.")
