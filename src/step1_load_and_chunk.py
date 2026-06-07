"""
========================================================================
STEP 1 (Day 2): LOAD a PDF and CHUNK it into pieces
========================================================================

WHY THIS STEP?
--------------
An LLM cannot read a 100-page PDF all at once (it has a limited "context
window"). So the RAG process is:
    1. LOAD  the PDF text
    2. CHUNK it into small pieces
Later we will turn each chunk into numbers (embeddings) so we can search them.

Run this file on its own to SEE what chunks look like:
    python src/step1_load_and_chunk.py
========================================================================
"""

from pathlib import Path
import re
import unicodedata

# PyPDFLoader reads a PDF and gives us its text, page by page.
from langchain_community.document_loaders import PyPDFLoader

# This splitter cuts long text into smaller overlapping chunks.
# (In newer LangChain this lives in its own package: langchain-text-splitters)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------------------------------------------
# PROJECT_ROOT = the genai-rag-project folder (one level above src/).
# We build all file paths from here so the scripts work NO MATTER which
# folder you run them from (root, src/, anywhere). This is why earlier
# "data/sample.pdf" failed: it was relative to your current folder.
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def clean_text(text: str) -> str:
    """
    Clean messy text extracted from PDFs.

    WHY THIS MATTERS (we proved this with a test!):
    PDFs often extract text with broken characters — ligatures like 'ﬁ',
    leftover icon/symbol junk ('5R', '÷'), and odd spacing ('T raditional').
    This "noise" confuses the LLM and makes it produce random, wrong answers.
    Cleaning the text fixes the RAG quality completely.
    """
    # 0) Strip NULL bytes (\x00) — the PDF embeds them between characters,
    #    producing garbled text like '\x00W\x00h\x00a\x00t'. This is the #1
    #    cause of the LLM hallucinating random answers.
    text = text.replace("\x00", "")
    # 1) Normalize unicode: turn ligatures (ﬁ -> fi, ﬂ -> fl) into plain letters.
    text = unicodedata.normalize("NFKD", text)
    # 2) Drop any leftover non-ASCII characters (symbols, icon junk).
    text = text.encode("ascii", "ignore").decode("ascii")
    # 3) Remove stray short alphanumeric codes left by PDF icons/bullets
    #    (e.g. '5R', '5S', single junk chars on their own line).
    text = re.sub(r"(?m)^[A-Z0-9]{1,3}\s*$", "", text)          # full-line junk
    text = re.sub(r"(?<!\w)[0-9][A-Z](?!\w)", "", text)          # inline like '5R'
    text = re.sub(r"(?<!\w)[A-Z][0-9](?!\w)", "", text)          # inline like 'R5'
    # 4) Fix digits inside words from bad OCR (De0finition -> Definition).
    text = re.sub(r"(?<=[A-Za-z])0(?=[A-Za-z])", "o", text)
    # 5) Fix broken spacing in words (T raditional -> Traditional).
    text = re.sub(r"\b([A-Z])\s([a-z])", r"\1\2", text)
    # 6) Collapse multiple spaces/newlines into single ones (tidy spacing).
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(pdf_path: str):
    """
    Load a PDF file and return a list of 'Document' objects.
    Each Document = the text of ONE page + metadata (file name, page number).
    We CLEAN each page's text right after loading.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()          # one Document per page

    # Clean every page so chunks (and the LLM) only see tidy text.
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    print(f"✅ Loaded '{pdf_path}' — {len(documents)} pages (text cleaned)")
    return documents


def chunk_documents(documents):
    """
    Split the page-documents into smaller chunks.

    chunk_size=1000     -> each chunk is ~1000 characters
    chunk_overlap=200   -> each chunk shares 200 chars with the next one.
                           Overlap keeps sentences from being cut in a way
                           that loses meaning at the boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


# This block runs ONLY when you execute this file directly.
if __name__ == "__main__":
    # 👉 Put a PDF named "sample.pdf" in the data/ folder.
    #    DATA_DIR is an absolute path, so this works from any folder.
    PDF_PATH = str(DATA_DIR / "sample.pdf")

    # Friendly check: tell the user clearly if the PDF is missing.
    if not Path(PDF_PATH).exists():
        print(f"❌ PDF not found at: {PDF_PATH}")
        print("👉 Put a PDF named 'sample.pdf' inside the data/ folder.")
        raise SystemExit

    docs = load_pdf(PDF_PATH)
    chunks = chunk_documents(docs)

    # Print the first 2 chunks so you can SEE what RAG actually works with.
    print("\n--- First 2 chunks preview ---")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n[Chunk {i}] (from page {chunk.metadata.get('page')})")
        print(chunk.page_content[:300], "...")
