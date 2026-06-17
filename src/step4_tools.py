"""
========================================================================
STEP 4 (Day 7-8): TOOLS for the Agent
========================================================================

WHAT IS A TOOL?
---------------
A "tool" is just a normal Python function that we expose to the LLM.
The LLM can DECIDE to call it (with arguments) when it thinks it helps
answer the question. This is the heart of "agentic AI".

We use the @tool decorator from LangChain. The function's NAME and its
DOCSTRING are sent to the LLM — that's how the model knows what each tool
does and when to use it. So write clear docstrings!

We define 3 tools:
    1. search_documents  -> our RAG retriever (search the PDF)
    2. calculator        -> safe math evaluation
    3. web_search        -> live internet search (optional, needs Tavily key)
    4. summarize_document -> get the gist of the document

========================================================================
"""

import os
import sys
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"
sys.path.append(str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from langchain_core.tools import tool

from step2_build_vectorstore import load_vectorstore

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ----------------------------------------------------------------------
# TOOL 1: Search the PDF (this is our Tier 1 RAG, now wrapped as a tool)
# ----------------------------------------------------------------------
# We load the retriever once at import time so every call reuses it.
# Note: This requires a vector store to exist.
try:
    _retriever = load_vectorstore().as_retriever(search_kwargs={"k": 3})
except Exception:
    _retriever = None

@tool
def search_documents(query: str) -> str:
    """Search the uploaded PDF document for information about a topic.
    Use this for any question about the document's content (e.g. definitions,
    concepts, facts that would be inside the PDF). Input: a search query."""
    if _retriever is None:
        return "Error: Vector store not found. Please run step 2 first."
    docs = _retriever.invoke(query)
    if not docs:
        return "No relevant information found in the document."
    # Return the retrieved chunks as text for the LLM to read.
    return "\n\n".join([doc.page_content for doc in docs])


import math

# ----------------------------------------------------------------------
# TOOL 2: Calculator (LLMs are bad at exact math; give them a real one)
# ----------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result.
    Use this for any arithmetic or calculation (e.g. '25 * 4', 'sqrt(256)', '2**8').
    Input: a math expression as a string."""
    # Allow numbers, basic operators, brackets, and math functions
    allowed_chars = set("0123456789+-*/(). %*")
    # Clean the expression for common LLM mistakes (like ^ for power)
    expression = expression.replace("^", "**")
    
    if not set(expression).issubset(allowed_chars) and "sqrt" not in expression:
        return "Error: only numbers, + - * / ( ) % ** and sqrt are allowed."
    
    try:
        # Provide math functions to eval safely
        safe_dict = {"sqrt": math.sqrt, "pow": math.pow, "__builtins__": {}}
        result = eval(expression, safe_dict, {})
        return str(result)
    except Exception as e:
        return f"Error: could not evaluate '{expression}' ({e})"


# ----------------------------------------------------------------------
# TOOL 3: Web search (optional — only works if TAVILY_API_KEY is set)
# ----------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Search the live internet for current/up-to-date information.
    Use this ONLY when the answer is not in the document and needs recent or
    general world knowledge (e.g. today's news, current events). Input: a query."""
    if not os.getenv("TAVILY_API_KEY"):
        return "Web search is unavailable (no TAVILY_API_KEY set)."
    try:
        from langchain_tavily import TavilySearchResults
        search = TavilySearchResults(max_results=3)
        return str(search.invoke(query))
    except Exception as e:
        return f"Web search error: {e}"


# ----------------------------------------------------------------------
# TOOL 4: Summarizer (Get the gist of the document)
# ----------------------------------------------------------------------
@tool
def summarize_document(topic: str = "the entire document") -> str:
    """Provides a concise summary of the document or a specific topic within it.
    Use this when the user asks for a 'summary', 'overview', or 'key points'.
    Input: the specific topic to summarize (default is the whole document)."""
    if _retriever is None:
        return "Error: Vector store not found. Please run step 2 first."
    
    # We retrieve more chunks (k=10) to get a broader overview for the summary.
    docs = _retriever.vectorstore.similarity_search(f"Give me a comprehensive overview of {topic}", k=10)
    if not docs:
        return "No relevant information found to summarize."
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # We use a simple prompt for the LLM to summarize the retrieved chunks.
    # Note: In a real agent, the LLM calls this tool and then summarizes the output itself.
    # Here we return the most relevant text for the agent to process.
    return f"Here are the key excerpts from the document related to '{topic}':\n\n{context}"

# A list we can import elsewhere to give the agent all its tools.
ALL_TOOLS = [search_documents, calculator, web_search, summarize_document]


if __name__ == "__main__":
    # Quick test of each tool on its own (we call .invoke on the tool).
    print("=== TOOL 1: search_documents ===")
    print(search_documents.invoke("What is Machine Learning?")[:300], "...\n")

    print("=== TOOL 2: calculator ===")
    print("25 * 4 =", calculator.invoke("25*4"))
    print("(100-20)/4 =", calculator.invoke("(100-20)/4"), "\n")

    print("=== TOOL 3: web_search ===")
    print(web_search.invoke("latest AI news")[:200], "...")

