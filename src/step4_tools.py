"""
STEP 4: TOOLS for the Agent (Production Version)
"""

import os
import math
from langchain_core.tools import tool
from src.config import TAVILY_API_KEY, SUMMARY_SEARCH_K
from src.logger import setup_logger
from src.step2_build_vectorstore import load_vectorstore

logger = setup_logger("Step4_Tools")

@tool
def search_documents(query: str) -> str:
    """Search the uploaded PDF for information."""
    db = load_vectorstore()
    if not db: return "Error: Document database not found."
    docs = db.as_retriever(search_kwargs={"k": 3}).invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    allowed_chars = set("0123456789+-*/(). %*")
    expression = expression.replace("^", "**")
    if not set(expression).issubset(allowed_chars) and "sqrt" not in expression:
        return "Error: Invalid math expression."
    try:
        safe_dict = {"sqrt": math.sqrt, "pow": math.pow, "__builtins__": {}}
        return str(eval(expression, safe_dict, {}))
    except Exception as e:
        return f"Math Error: {e}"

@tool
def web_search(query: str) -> str:
    """Search the internet using Tavily."""
    if not TAVILY_API_KEY:
        return "Web search is disabled (no API key)."
    try:
        from langchain_tavily import TavilySearchResults
        search = TavilySearchResults(max_results=3)
        return str(search.invoke(query))
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Web Search Error: {e}"

@tool
def summarize_document(topic: str = "the entire document") -> str:
    """Get a summary or key points from the document."""
    db = load_vectorstore()
    if not db: return "Error: Document database not found."
    docs = db.similarity_search(f"Overview of {topic}", k=SUMMARY_SEARCH_K)
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"Key context for summary of '{topic}':\n\n{context}"

ALL_TOOLS = [search_documents, calculator, web_search, summarize_document]
