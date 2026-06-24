"""
========================================================================
STEP 2 (Day 3): EMBEDDINGS + VECTOR STORE
========================================================================

WHY THIS STEP?
--------------
Computers can't search text by "meaning" directly. So we convert each
chunk into a list of numbers called an EMBEDDING (a vector). Chunks with
similar meaning end up with similar numbers (close together in space).

We store all these vectors in a VECTOR DATABASE (ChromaDB). Then, when a
user asks a question, we embed the question too and find the chunks whose
vectors are CLOSEST to it. That's "semantic search" — search by meaning,
not by exact keywords.

Run:
    python src/step2_build_vectorstore.py
========================================================================
"""

import os
import sys
import shutil
from pathlib import Path

# Turn OFF ChromaDB's telemetry (stops the harmless but annoying
# "capture() takes 1 positional argument" error message).
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Make sure Python can find step1 even when you run from the project root.
# (We add this file's own folder, src/, to the import search path.)
sys.path.append(str(Path(__file__).resolve().parent))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Reuse the loading/chunking we wrote in step 1.
from step1_load_and_chunk import load_pdf, chunk_documents, DATA_DIR, PROJECT_ROOT


# Where ChromaDB saves the vectors on disk (absolute path = run from anywhere).
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")


def get_embedding_model():
    """
    Load a FREE, local embedding model. No API key needed.
    'all-MiniLM-L6-v2' is small, fast, and good enough for learning.
    It turns each chunk into a 384-number vector.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_vectorstore(chunks):
    """
    Take the text chunks -> embed them -> store in ChromaDB on disk.
    Returns the vector store so we can search it.

    IMPORTANT: We DELETE any existing chroma_db first. Otherwise Chroma ADDS
    the new chunks to the OLD ones, so an old PDF's content stays mixed in
    with your new PDF. Deleting guarantees a clean, fresh index each build.
    """
    # Clear Chroma's internal system cache so it doesn't get confused
    # when we delete its underlying files! This prevents the "default_tenant" error.
    import chromadb
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    # Wipe the old database so we start fresh (no leftover chunks).
    if os.path.exists(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR)
            print(f"🧹 Removed old vector store at '{CHROMA_DIR}'")
        except Exception as e:
            print(f"⚠️ Could not remove old vector store: {e}")

    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,   # saved to disk here
    )
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_DIR}'")
    return vectorstore


def load_vectorstore():
    """
    Load an EXISTING ChromaDB from disk (no need to rebuild every run).
    """
    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


if __name__ == "__main__":
    PDF_PATH = str(DATA_DIR / "sample.pdf")

    if not Path(PDF_PATH).exists():
        print(f"❌ PDF not found at: {PDF_PATH}")
        print("👉 Put a PDF named 'sample.pdf' inside the data/ folder.")
        raise SystemExit

    # Build the vector store from the PDF.
    docs = load_pdf(PDF_PATH)
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)

    # 🔎 Test semantic search: find the 3 chunks closest to our question.
    question = "What is this document about?"
    results = vectorstore.similarity_search(question, k=3)

    print(f"\n--- Top 3 chunks for: '{question}' ---")
    for i, doc in enumerate(results):
        print(f"\n[Result {i}] (page {doc.metadata.get('page')})")
        print(doc.page_content[:250], "...")
