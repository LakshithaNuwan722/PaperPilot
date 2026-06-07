"""
========================================================================
STEP 4 (Day 5): STREAMLIT WEB UI
========================================================================

WHY THIS STEP?
--------------
A terminal is fine for testing, but a web app is what you DEMO and put on
your CV. Streamlit lets us build a UI with pure Python — no HTML/JS needed.

This app lets a user:
    1. Upload a PDF
    2. Ask questions about it
    3. See the answer AND the source chunks it came from
       (showing sources builds trust — interviewers love this!)

Run from inside the genai-rag-project folder:
    streamlit run src/app.py
========================================================================
"""

import os
import tempfile

# Turn OFF ChromaDB telemetry (stops the harmless capture() error).
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ---------- Page setup ----------
st.set_page_config(page_title="PaperPilot ✈️", page_icon="✈️")
st.title("✈️ PaperPilot")
st.caption("An Agentic RAG assistant — upload a PDF and ask questions about it.")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# @st.cache_resource caches the heavy work so it runs ONCE per PDF,
# not on every keystroke. This makes the app fast.
@st.cache_resource(show_spinner="Processing your PDF...")
def build_retriever(file_bytes):
    """Load -> chunk -> embed -> store the uploaded PDF, return a retriever."""
    # Streamlit gives us bytes; PyPDFLoader needs a real file path,
    # so we write the bytes to a temporary file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    documents = PyPDFLoader(tmp_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # In-memory store (no persist_directory) — fine for a single session.
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


# Strict rules go in the SYSTEM message so the model can't ignore them.
SYSTEM_RULES = """You are a document question-answering assistant.

You will be given a CONTEXT from a document and a QUESTION.
Follow these rules strictly:
- Answer ONLY using information in the CONTEXT.
- IGNORE any knowledge you have from training. Use ONLY the CONTEXT.
- If the answer is not in the CONTEXT, reply EXACTLY:
  "I don't know based on the document."
- Give one direct answer. Do not invent extra questions or conversation."""


def answer_question(question, retriever):
    """Retrieve chunks, send system+user messages to the LLM, return answer."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Two separate messages: rules (system) + context & question (user).
    user_content = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    messages = [
        SystemMessage(content=SYSTEM_RULES),
        HumanMessage(content=user_content),
    ]
    return llm.invoke(messages).content


# ---------- Main UI ----------
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Add it to your .env file.")
    st.stop()

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    retriever = build_retriever(uploaded.getvalue())
    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Thinking..."):
            answer = answer_question(question, retriever)

        st.subheader("Answer")
        st.write(answer)

        # Show the source chunks used (transparency = trust).
        with st.expander("📚 Sources used"):
            for i, doc in enumerate(retriever.invoke(question)):
                st.markdown(f"**Chunk {i+1}** (page {doc.metadata.get('page')})")
                st.write(doc.page_content[:400] + "...")
else:
    st.info("👆 Upload a PDF to get started.")
