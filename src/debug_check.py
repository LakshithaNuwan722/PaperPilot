"""
========================================================================
DEBUG TOOL — find out WHY you get wrong answers
========================================================================
This inspects what is ACTUALLY in your vector store and what gets
retrieved. Run it AFTER step2:

    python src/debug_check.py
========================================================================
"""
import os
import sys
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"
sys.path.append(str(Path(__file__).resolve().parent))

from step2_build_vectorstore import load_vectorstore, CHROMA_DIR
from step1_load_and_chunk import DATA_DIR

print("=" * 60)
print("1) WHICH PDF FILES ARE IN data/ ?")
print("=" * 60)
pdfs = list(Path(DATA_DIR).glob("*.pdf"))
if not pdfs:
    print("❌ NO PDFs found in data/ !")
for p in pdfs:
    print(f"   📄 {p.name}  ({p.stat().st_size // 1024} KB)")

print("\n" + "=" * 60)
print(f"2) DOES THE VECTOR DB EXIST?  ({CHROMA_DIR})")
print("=" * 60)
print("   ✅ exists" if os.path.exists(CHROMA_DIR) else "   ❌ MISSING — run step2 first!")

print("\n" + "=" * 60)
print("3) HOW MANY CHUNKS ARE STORED, AND WHAT DO THEY SAY?")
print("=" * 60)
vs = load_vectorstore()
data = vs.get()  # returns all stored docs
docs = data.get("documents", [])
print(f"   Total chunks in DB: {len(docs)}")
print("\n   --- First 3 stored chunks (this is what RAG searches!) ---")
for i, d in enumerate(docs[:3]):
    print(f"\n   [Chunk {i}] {d[:200]}...")

print("\n" + "=" * 60)
print("4) WHAT GETS RETRIEVED FOR YOUR QUESTION?")
print("=" * 60)
question = "What is Machine Learning?"
results = vs.similarity_search(question, k=3)
print(f"   Question: {question}")
for i, r in enumerate(results):
    print(f"\n   [Retrieved {i}] {r.page_content[:200]}...")

print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("=" * 60)
print("""
- If section 3 shows Sales/Marketing/IT text  -> the DB still has the OLD pdf.
- If section 3 shows ML text but answers wrong -> prompt/LLM issue.
- If section 1 shows a file NOT named sample.pdf -> step2 loaded the wrong file.
""")
