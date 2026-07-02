"""
✈️ PaperPilot — Agentic RAG Assistant
======================================
Production-ready UI for Hugging Face Spaces deployment.
Dark glassmorphism design with animated gradients.
"""

import os
import sys
import tempfile
import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage

# --- Path setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DATA_DIR
from src.logger import setup_logger
from src.step1_load_and_chunk import load_pdf, chunk_documents
from src.step2_build_vectorstore import build_vectorstore
from src.step5_agent import build_agent

logger = setup_logger("Streamlit_UI")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
LOG_FILE = "logs/interactions.jsonl"
os.makedirs("logs", exist_ok=True)

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="PaperPilot AI — Agentic RAG",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://huggingface.co",
        "About": "# PaperPilot AI\nAn Agentic RAG assistant powered by LangGraph + Groq.",
    },
)

# ─────────────────────────────────────────
# GLOBAL CSS — Dark AI / HuggingFace Style
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:    #0f1117;
    --bg-secondary:  #1a1d27;
    --bg-card:       rgba(255,255,255,0.05);
    --accent:        #ff9d00;
    --accent-soft:   rgba(255,157,0,0.15);
    --accent2:       #a855f7;
    --text-primary:  #f1f5f9;
    --text-muted:    #94a3b8;
    --border:        rgba(255,255,255,0.08);
    --success:       #22c55e;
    --error:         #ef4444;
    --radius:        14px;
    --shadow:        0 8px 32px rgba(0,0,0,0.4);
}

/* ── Base ── */
html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #0f1117 0%, #1a1d27 50%, #0f1117 100%) !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1100px; }

/* ── Animated gradient background blobs ── */
.stApp::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background:
        radial-gradient(circle at 20% 20%, rgba(255,157,0,0.06) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(168,85,247,0.06) 0%, transparent 50%);
    animation: bgPulse 8s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
@keyframes bgPulse {
    0%   { opacity: 0.6; transform: scale(1); }
    100% { opacity: 1;   transform: scale(1.05); }
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: var(--accent-soft) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--accent) !important;
    color: #000 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(255,157,0,0.3);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed rgba(255,157,0,0.4) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    backdrop-filter: blur(10px) !important;
    margin-bottom: 0.75rem !important;
    color: var(--text-primary) !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: var(--text-primary) !important;
}

/* User message accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid var(--accent) !important;
}
/* Assistant message accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid var(--accent2) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--bg-secondary) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 16px !important;
    color: var(--text-primary) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(255,157,0,0.15) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
}
[data-testid="stChatInput"] * { color: var(--text-primary) !important; }

/* ── Status widget (agent thinking) ── */
[data-testid="stStatus"],
.stStatus {
    background: rgba(168,85,247,0.08) !important;
    border: 1px solid rgba(168,85,247,0.25) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
}
[data-testid="stStatus"] p,
[data-testid="stStatus"] span,
[data-testid="stStatus"] div { color: var(--text-primary) !important; }

/* ── Markdown text ── */
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown strong, .stMarkdown em, .stMarkdown code {
    color: var(--text-primary) !important;
}
.stMarkdown code {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 5px;
    padding: 2px 6px;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Custom hero classes ── */
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #ff9d00, #a855f7, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.5rem;
    animation: gradShift 4s ease infinite alternate;
    background-size: 200%;
}
@keyframes gradShift {
    0%   { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
}

.hero-sub {
    text-align: center;
    color: var(--text-muted);
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.badge {
    display: inline-block;
    background: var(--accent-soft);
    border: 1px solid rgba(255,157,0,0.3);
    color: var(--accent);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 3px;
}
.badge-purple {
    background: rgba(168,85,247,0.1);
    border-color: rgba(168,85,247,0.3);
    color: #c084fc;
}
.badge-blue {
    background: rgba(6,182,212,0.1);
    border-color: rgba(6,182,212,0.3);
    color: #22d3ee;
}
.badge-green {
    background: rgba(34,197,94,0.1);
    border-color: rgba(34,197,94,0.3);
    color: #4ade80;
}

.feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s, transform 0.2s;
    text-align: center;
}
.feature-card:hover {
    border-color: rgba(255,157,0,0.4);
    transform: translateY(-2px);
}
.feature-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.feature-title { font-weight: 700; color: var(--text-primary); margin-bottom: 0.25rem; }
.feature-desc { font-size: 0.85rem; color: var(--text-muted); }

.step-indicator {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: var(--text-muted);
    text-align: center;
}

.doc-status {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    color: #4ade80;
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 0.5rem;
}

.sidebar-logo {
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ff9d00, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Spinner override */
[data-testid="stSpinner"] p { color: var(--text-muted) !important; }

/* General text fallback */
p, span, div, li, label { color: var(--text-primary); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# AGENT INIT
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_system(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    docs = load_pdf(path)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    return build_agent()


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🧭 PaperPilot AI</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;font-size:0.8rem;margin-top:2px;">Agentic RAG · v2.0</p>', unsafe_allow_html=True)

    st.divider()

    # Upload
    uploaded = st.file_uploader(
        "**📄 Upload PDF**",
        type="pdf",
        help="Upload any research paper, report, or document.",
    )

    if uploaded:
        st.markdown(
            f'<div class="doc-status">✅ &nbsp;<strong>{uploaded.name}</strong> ready</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Capabilities badges
    st.markdown("**⚡ Agent Capabilities**")
    st.markdown("""
    <div style="line-height:2.2;">
        <span class="badge">🔍 Document RAG</span>
        <span class="badge badge-purple">🧮 Calculator</span>
        <span class="badge badge-blue">🌐 Web Search</span>
        <span class="badge badge-green">📝 Summarizer</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Model info
    st.markdown("**🤖 Powered By**")
    st.markdown("""
    <div style="font-size:0.82rem;color:#64748b;line-height:1.9;">
        🤖 &nbsp;Groq · GPT-OSS 120B<br>
        🔗 &nbsp;LangGraph ReAct Agent<br>
        🗃️ &nbsp;ChromaDB Vector Store<br>
        🚀 &nbsp;HuggingFace Spaces
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.72rem;color:#334155;text-align:center;">© 2026 PaperPilot AI</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────
# MAIN CONTENT — LANDING
# ─────────────────────────────────────────
if not uploaded:
    st.markdown('<div class="hero-title">PaperPilot AI 🧭</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Drop a PDF. Ask anything. Let the agent do the heavy lifting.</p>',
        unsafe_allow_html=True,
    )

    # Feature cards
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("🔍", "Deep Search", "Semantic retrieval across your entire document."),
        ("🧮", "Calculator", "Solve equations and numeric problems instantly."),
        ("🌐", "Web Search", "Go beyond the PDF with live Tavily search."),
        ("📝", "Summarizer", "Get concise, intelligent document summaries."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f"""<div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # How to use
    st.markdown("### 🚀 How to Get Started")
    col_a, col_b, col_c = st.columns(3)
    steps = [
        ("01", "Upload PDF", "Click **Upload PDF** in the sidebar and select your document."),
        ("02", "Ask Questions", "Type any question about your document in the chat box."),
        ("03", "Get Insights", "The AI agent searches, calculates, and synthesizes answers."),
    ]
    for col, (num, title, desc) in zip([col_a, col_b, col_c], steps):
        with col:
            st.markdown(
                f"""<div class="step-indicator">
                    <span style="font-size:1.6rem;font-weight:800;color:#ff9d00;">{num}</span><br>
                    <strong style="color:#f1f5f9;">{title}</strong><br>
                    <span style="font-size:0.82rem;">{desc}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#475569;font-size:0.85rem;">⬅️ Upload a PDF from the sidebar to begin your research session</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────
# MAIN CONTENT — CHAT
# ─────────────────────────────────────────
else:
    # Init agent & session state
    with st.spinner("🧠 &nbsp;Embedding document and initialising agent…"):
        st.session_state.agent = init_system(uploaded.getvalue())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show a subtle header when in chat mode
    st.markdown(
        f'<p style="text-align:center;color:#64748b;font-size:0.85rem;margin-bottom:1rem;">'
        f'📄 &nbsp;<strong style="color:#f1f5f9;">{uploaded.name}</strong> &nbsp;·&nbsp; '
        f'{len(st.session_state.messages)//2} exchange(s)</p>',
        unsafe_allow_html=True,
    )

    # Replay chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── Chat input ──
    if prompt := st.chat_input("Ask anything about your document…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            start_time = time.time()
            error_occurred = False
            full_response = ""

            # ── Agent streaming with reasoning panel ──
            with st.status("🤖 &nbsp;**Agent reasoning…**", expanded=True) as status:
                try:
                    for step in st.session_state.agent.stream(
                        {"messages": [("user", prompt)]},
                        stream_mode="values",
                    ):
                        msg = step["messages"][-1]

                        # Tool call hints
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_map = {
                                    "search_documents": "🔍 &nbsp;Searching document index…",
                                    "calculator":       "🧮 &nbsp;Running calculations…",
                                    "web_search":       "🌐 &nbsp;Fetching live web results…",
                                    "summarize_document": "📝 &nbsp;Synthesising document summary…",
                                }
                                hint = tool_map.get(tc["name"], f"🔧 &nbsp;Calling `{tc['name']}`…")
                                st.markdown(
                                    f'<p style="color:#94a3b8;font-size:0.9rem;">{hint}</p>',
                                    unsafe_allow_html=True,
                                )

                        # Capture final AI response
                        if isinstance(msg, AIMessage) and not msg.tool_calls:
                            content = msg.content
                            if isinstance(content, list):
                                text_parts = [
                                    c.get("text", "")
                                    for c in content
                                    if isinstance(c, dict) and c.get("type") == "text"
                                ]
                                candidate = " ".join(text_parts).strip()
                            else:
                                candidate = str(content).strip()
                            if candidate:
                                full_response = candidate

                    duration = round(time.time() - start_time, 2)
                    status.update(
                        label=f"✅ &nbsp;**Done** &nbsp;·&nbsp; {duration}s",
                        state="complete",
                        expanded=False,
                    )

                except Exception as e:
                    error_occurred = True
                    status.update(label="❌ &nbsp;**Agent error**", state="error")
                    st.error(f"**Error:** {e}")
                    logger.error(f"UI Error: {e}")

            # ── Render final answer outside status box ──
            if not error_occurred and full_response:
                st.markdown(full_response)

                # Persist to session & log
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                with open(LOG_FILE, "a") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now().isoformat(),
                        "question":  prompt,
                        "answer":    full_response,
                        "latency":   duration,
                    }) + "\n")

            elif not error_occurred and not full_response:
                st.warning("⚠️ The agent returned an empty response. Try rephrasing your question.")
