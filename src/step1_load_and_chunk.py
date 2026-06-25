"""
STEP 1: LOAD a PDF and CHUNK it into pieces (Production Version)
"""

import re
import unicodedata
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import centralized config and logger
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR
from src.logger import setup_logger

logger = setup_logger("Step1_Ingestion")

def clean_text(text: str) -> str:
    """Clean messy text extracted from PDFs."""
    text = text.replace("\x00", "")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"(?m)^[A-Z0-9]{1,3}\s*$", "", text)
    text = re.sub(r"(?<!\w)[0-9][A-Z](?!\w)", "", text)
    text = re.sub(r"(?<!\w)[A-Z][0-9](?!\w)", "", text)
    text = re.sub(r"(?<=[A-Za-z])0(?=[A-Za-z])", "o", text)
    text = re.sub(r"\b([A-Z])\s([a-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_pdf(pdf_path: str):
    """Load a PDF and clean its text."""
    logger.info(f"Loading PDF from {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    logger.info(f"Loaded {len(documents)} pages")
    return documents

def chunk_documents(documents):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
    PDF_PATH = str(DATA_DIR / "sample.pdf")
    if Path(PDF_PATH).exists():
        docs = load_pdf(PDF_PATH)
        chunks = chunk_documents(docs)
    else:
        logger.error(f"PDF not found at {PDF_PATH}")
