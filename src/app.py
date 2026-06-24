"""
========================================================================
STEP 6 (Day 10): AGENTIC WEB UI
========================================================================
This is the final polished Web UI that connects the Step 5 Agent 
to a professional Streamlit interface.

Features:
- PDF Uploading & Processing
- Agentic Reasoning (Calculator, Search, Summarizer, Web)
- Displays "Thought Steps" (Tool calls)
- Clean Chat Interface
========================================================================
"""

import os
import sys
import tempfile
from pathlib import Path

# Turn OFF ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import streamlit as st
from dotenv import load_dotenv

# Add src to path so we can import our steps
sys.path.append(str(Path(__file__).resolve().parent))

from step1_load_and_chunk import load_pdf, chunk_documents
from step2_build_vectorstore import build_vectorstore
from step5_agent import build_agent

load_dotenv()

# ---------- Page Configuration ----------
st.set_page_config(page_title="PaperPilot Agent ✈️", page_icon="✈️", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .stChatFloatingInputContainer { bottom: 20px; }
    .tool-call { 
        color: #555; 
        font-style: italic; 
        font-size: 0.85rem;
        margin-bottom: 10px;
        padding-left: 20px;
        border-left: 2px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✈️ PaperPilot: Agentic Research Assistant")
st.caption("I can search your PDF, do math, and browse the web to help you.")

# ---------- Sidebar: File Upload & Settings ----------
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("### Example Questions")
    st.info("- Summarize the main points\n- What is the definition of X?\n- Calculate (25 * 4) + 15\n- Search the web for latest news on Y")

# ---------- Session State Initialization ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

# ---------- PDF Processing Logic ----------
@st.cache_resource(show_spinner="Processing your PDF...")
def initialize_system(file_bytes):
    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    # Process PDF (Reuse Step 1 & 2)
    docs = load_pdf(tmp_path)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    
    # Reload the retriever in step4_tools so the tools use the newly built DB!
    import step4_tools
    from step2_build_vectorstore import load_vectorstore
    step4_tools._retriever = load_vectorstore().as_retriever(search_kwargs={"k": 3})
    
    # Build the Agent (Step 5)
    return build_agent()

# ---------- Main UI Interaction ----------
if uploaded_file:
    # Initialize agent if not already done
    st.session_state.agent = initialize_system(uploaded_file.getvalue())
    
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask PaperPilot anything..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Agent Response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            thought_container = st.container()
            
            final_answer = ""
            
            try:
                # Stream the agent's thoughts and steps
                inputs = {"messages": [("user", prompt)]}
                
                for step in st.session_state.agent.stream(inputs, stream_mode="values"):
                    msg = step["messages"][-1]
                    
                    # If it's a tool call, display it in the thought container
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            thought_container.markdown(f"*🔧 calling tool: {tc['name']}...*")
                    
                    final_answer = msg.content
                
                response_placeholder.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
            except Exception as e:
                error_msg = f"⚠️ Oops! I ran into an error: {str(e)}"
                st.error(error_msg)

else:
    st.info("👆 Please upload a PDF in the sidebar to start chatting!")

# Footer
st.divider()
st.caption("Powered by LangGraph, Groq, and Streamlit.")
