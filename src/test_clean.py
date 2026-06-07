"""
========================================================================
DECISIVE TEST — is the messy PDF text the problem?
We send (1) a hand-clean context, then (2) the real (messy) context,
to the SAME model. If clean works and messy fails -> PDF text is the cause.

    python src/test_clean.py
========================================================================
"""
import os
import sys
import unicodedata
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"
sys.path.append(str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from step2_build_vectorstore import load_vectorstore

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
SYS = ('Answer ONLY using the CONTEXT. If not in context, say '
       '"I don\'t know based on the document." Give one short answer.')


def ask(context, question):
    msgs = [SystemMessage(content=SYS),
            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {question}")]
    return llm.invoke(msgs).content


def clean_text(text):
    """Normalize unicode, drop weird control chars, fix spacing."""
    # Turn ligatures (ﬁ, ﬂ) and odd forms into plain ASCII equivalents.
    text = unicodedata.normalize("NFKD", text)
    # Keep only printable ASCII + common whitespace.
    text = "".join(c for c in text if c.isprintable() or c in "\n\t ")
    # Remove non-ascii leftovers.
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


Q = "What is Machine Learning?"

# Get the REAL messy context from the vector store.
retriever = load_vectorstore().as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke(Q)
messy = "\n\n".join(d.page_content for d in docs)

print("=" * 60)
print("TEST 1: clean hand-written context")
print("=" * 60)
clean_ctx = ("Machine Learning is a subset of Artificial Intelligence that "
             "lets computers learn from data without being explicitly programmed.")
print("Answer:", ask(clean_ctx, Q))

print("\n" + "=" * 60)
print("TEST 2: REAL messy context (raw from PDF)")
print("=" * 60)
print("Answer:", ask(messy, Q))

print("\n" + "=" * 60)
print("TEST 3: REAL context but CLEANED")
print("=" * 60)
print("Answer:", ask(clean_text(messy), Q))

print("\n" + "=" * 60)
print("DIAGNOSIS")
print("=" * 60)
print("""
- TEST1 ok, TEST2 weird, TEST3 ok  -> messy PDF text is the cause (fix: clean text)
- TEST1 ok, TEST2 ok,    TEST3 ok  -> problem was elsewhere; try step3 again
- ALL weird                        -> deeper issue (tell me!)
""")
