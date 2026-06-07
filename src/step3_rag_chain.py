"""
========================================================================
STEP 3 (Day 4): CONNECT THE LLM  ->  this is the REAL RAG
========================================================================
We send a clear SYSTEM message + USER message so the LLM cannot ignore
the context. This fixes the bug where the model answered from its own
training memory instead of from our document.

    question -> RETRIEVE chunks -> system+user messages -> LLM -> answer

Run (set GROQ_API_KEY in .env first):
    python src/step3_rag_chain.py
========================================================================
"""

import os
import sys
from pathlib import Path

# Turn OFF ChromaDB telemetry (stops the harmless capture() error).
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Make sure Python can find step1/step2 even when run from the project root.
sys.path.append(str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from step2_build_vectorstore import load_vectorstore

# Load GROQ_API_KEY from the .env file in the project root.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# The SYSTEM message sets strict rules. Models follow system messages strongly.
SYSTEM_RULES = """You are a document question-answering assistant.
You will receive CONTEXT (excerpts from a document) and a QUESTION.

Rules you MUST follow:
1. Answer ONLY using facts stated in the CONTEXT.
2. Do NOT use any knowledge from your training data.
3. If the CONTEXT does not contain the answer, reply EXACTLY: "I don't know based on the document."
4. Give ONE short, direct answer.
5. After answering, STOP. Do NOT generate follow-up questions, examples, or additional text.
6. Do NOT continue the conversation beyond the single answer."""


def answer_question(question, retriever, llm, show_context=False):
    """Retrieve chunks, send system+user messages to the LLM, return answer."""
    # 1) Retrieve the most relevant chunks for this question.
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 2) DEBUG: show exactly what context we retrieved.
    if show_context:
        print("\n" + "-" * 60)
        print("CONTEXT BEING SENT TO THE LLM:")
        print("-" * 60)
        print(context[:800], "...\n")

    # 3) Build TWO separate messages. This is the key fix:
    #    - SystemMessage = the rules
    #    - HumanMessage  = the actual context + question
    #    Keeping them separate stops the model from blending in its own memory.
    user_content = f"""Use ONLY the context below to answer the question.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}

Answer (one concise paragraph, then stop):"""

    messages = [
        SystemMessage(content=SYSTEM_RULES),
        HumanMessage(content=user_content),
    ]

    # 4) Call the LLM with the message list (not a plain string).
    response = llm.invoke(messages)
    # Strip any trailing noise the model may have appended.
    answer = response.content.strip()
    # Cut off at the first sign of a self-generated question (safety net).
    for marker in ["\nQuestion:", "\nQUESTION:", "\nQ:", "\n\n---"]:
        if marker in answer:
            answer = answer[:answer.index(marker)].strip()
    return answer


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found. Create a .env file (see .env.example).")
        raise SystemExit

    retriever = load_vectorstore().as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=512,
        stop=["\nQuestion:", "\nQUESTION:", "\nQ:", "\n\n---"],
    )

    print("✅ RAG chain ready. Ask questions about your PDF! (type 'quit' to exit)")
    print("   (context is shown so you can verify RAG is working)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        answer = answer_question(q, retriever, llm, show_context=True)
        print(f"🤖 {answer}\n")
