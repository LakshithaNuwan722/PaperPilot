"""
========================================================================
FULL RAG TEST — runs the EXACT same flow as step3, step by step,
printing everything so we can see precisely where it breaks.

    python src/test_rag_full.py
========================================================================
"""
import os
import sys
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"
sys.path.append(str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from step2_build_vectorstore import load_vectorstore

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

QUESTION = "What is Machine Learning?"

print("=" * 60)
print("STEP A: load vector store + retrieve")
print("=" * 60)
retriever = load_vectorstore().as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke(QUESTION)
print(f"Retrieved {len(docs)} chunks.")
context = "\n\n".join(d.page_content for d in docs)
print("\n--- CONTEXT (first 500 chars) ---")
print(context[:500])

print("\n" + "=" * 60)
print("STEP B: build the EXACT messages we send")
print("=" * 60)
SYSTEM_RULES = """You are a document question-answering assistant.
Answer ONLY using the CONTEXT. Ignore your training knowledge.
If the answer is not in the CONTEXT, say "I don't know based on the document."
Give one direct answer."""

user_content = f"CONTEXT:\n{context}\n\nQUESTION: {QUESTION}"

print("\n--- SYSTEM MESSAGE ---")
print(SYSTEM_RULES)
print("\n--- USER MESSAGE (first 500 chars) ---")
print(user_content[:500])

print("\n" + "=" * 60)
print("STEP C: call the LLM with these messages")
print("=" * 60)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
messages = [SystemMessage(content=SYSTEM_RULES), HumanMessage(content=user_content)]
response = llm.invoke(messages)

print("\n--- LLM RESPONSE ---")
print(response.content)

print("\n" + "=" * 60)
print("VERDICT:")
print("=" * 60)
ans = response.content.lower()
if "machine learning" in ans or "artificial intelligence" in ans:
    print("✅ CORRECT! RAG is working. The answer is about ML.")
elif "company" in ans or "department" in ans or "employee" in ans:
    print("❌ Model ignored context (company/department answer).")
    print("   But TEST 2 worked... so something differs here. Check the")
    print("   model name printed below and the context above.")
else:
    print("🤔 Unexpected answer. Read the response above.")

print(f"\nModel used: llama-3.3-70b-versatile")
