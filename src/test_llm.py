"""
========================================================================
MINIMAL LLM TEST — no RAG, no retrieval, no ChromaDB.
Just talk to the LLM directly to see if the LLM itself is the problem.

    python src/test_llm.py
========================================================================
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Show what key is loaded (first/last 4 chars only, for safety).
key = os.getenv("GROQ_API_KEY", "")
print(f"API key loaded: {key[:4]}...{key[-4:]}  (length {len(key)})")
print(f"Key starts with 'gsk_': {key.startswith('gsk_')}\n")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# TEST 1: trivial question, no context at all.
print("=" * 60)
print("TEST 1: Ask '2+2=?' (no context)")
print("=" * 60)
r1 = llm.invoke([HumanMessage(content="What is 2+2? Reply with just the number.")])
print("Response:", r1.content)

# TEST 2: give a tiny made-up context and ask about it.
print("\n" + "=" * 60)
print("TEST 2: Tiny context test")
print("=" * 60)
msgs = [
    SystemMessage(content="Answer ONLY from the context. Nothing else."),
    HumanMessage(content="CONTEXT: The sky is green today.\n\nQUESTION: What color is the sky?"),
]
r2 = llm.invoke(msgs)
print("Response:", r2.content)
print("\n(Expected: 'green'. If it says 'blue' or talks about a company,")
print(" the model is ignoring context = wrong model/key/endpoint.)")
