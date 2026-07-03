"""
STEP 2 (Day 3): EMBEDDINGS + VECTOR STORE (Production Version)
==============================================================
This version includes the production config, logging, and the 
fix for Chroma's SharedSystemClient cache.
"""

import os
import shutil
import sys
from pathlib import Path

# Import centralized production config and logger
from src.config import CHROMA_DIR, EMBEDDING_MODEL_NAME, DATA_DIR
from src.logger import setup_logger

# Make sure Python can find step1
sys.path.append(str(Path(__file__).resolve().parent))
from src.step1_load_and_chunk import load_pdf, chunk_documents

logger = setup_logger("Step2_VectorStore")

# Turn OFF ChromaDB's telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def get_embedding_model():
    """Load the embedding model specified in config."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def build_vectorstore(chunks):
    """
    Build a fresh vector database with production-grade logging and cache clearing.
    """
    # 🛠️ FIX: Clear Chroma's internal system cache to prevent the "default_tenant" error.
    try:
        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception as e:
        logger.warning(f"Could not clear Chroma cache: {e}")

    # Wipe the old database so we start fresh (no leftover chunks).
    if os.path.exists(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR)
            logger.info(f"🧹 Removed old vector store at '{CHROMA_DIR}'")
        except Exception as e:
            logger.error(f"⚠️ Could not remove old vector store: {e}")

    embeddings = get_embedding_model()
    
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        logger.info(f"✅ Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_DIR}'")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to build vectorstore: {e}")
        raise

def load_vectorstore():
    """
    Load an EXISTING ChromaDB from disk (no need to rebuild every run).
    """
    if not os.path.exists(CHROMA_DIR):
        logger.warning(f"No vector store found at {CHROMA_DIR}")
        return None
        
    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

if __name__ == "__main__":
    PDF_PATH = str(DATA_DIR / "sample.pdf")

    if not Path(PDF_PATH).exists():
        logger.error(f"❌ PDF not found at: {PDF_PATH}")
        print(f"👉 Put a PDF named 'sample.pdf' inside the {DATA_DIR} folder.")
        sys.exit(1)

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
