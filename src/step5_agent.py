"""
STEP 5: THE AGENT (Production Version)
"""

import os
import sys
import warnings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from src.config import PRIMARY_MODEL, FALLBACK_MODEL
from src.logger import setup_logger
from src.step4_tools import ALL_TOOLS

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = setup_logger("Step5_Agent")

AGENT_INSTRUCTIONS = """You are PaperPilot, an expert research assistant that answers questions \
strictly from an uploaded document. You have four tools available:

  1. search_documents  – Semantic search over the uploaded PDF (PRIMARY source)
  2. summarize_document – Retrieve broad context / summaries from the PDF
  3. calculator        – Evaluate math expressions
  4. web_search        – Search the live internet (LAST RESORT only)

## STRICT TOOL-USE RULES — follow these in order every time:

STEP 1 — ALWAYS start by calling `search_documents` with the user's question.
STEP 2 — Read the retrieved chunks carefully.
  • If they contain enough information → answer directly from them. STOP. Do NOT call web_search.
  • If the result is too short or says "no relevant content" → call `search_documents` once more \
with a rephrased query before giving up.
STEP 3 — If after two document searches the document truly lacks the information, ONLY THEN call \
`web_search`. You MUST state clearly in your answer: \
"(This information was not found in the document; sourced from the web.)"
STEP 4 — For questions involving numbers or calculations, use `calculator` in addition to document search.
STEP 5 — For broad summarization requests, use `summarize_document`.

## ABSOLUTE PROHIBITIONS:
- NEVER answer from your own training knowledge without first searching the document.
- NEVER skip `search_documents` for any factual question.
- NEVER call `web_search` if the document already contains sufficient information.

## OUTPUT FORMAT:
- Be concise and precise.
- Do NOT narrate your reasoning steps in the final answer.
- Do NOT say "According to the tool..." — just give the answer.
- If using web results, always mention it at the end of your response.
"""

def build_agent(model_name=None):
    """Build the LangGraph ReAct agent."""
    model = model_name or PRIMARY_MODEL
    llm = ChatGroq(model=model, temperature=0)
    return create_react_agent(llm, ALL_TOOLS, prompt=AGENT_INSTRUCTIONS)

def ask_agent(agent, question, max_retries=2):
    """Send a question to the agent with fallback retry logic."""
    inputs = {"messages": [HumanMessage(content=question)]}
    current_agent = agent
    
    for attempt in range(max_retries + 1):
        try:
            response = current_agent.invoke(inputs)
            return response["messages"][-1].content
        except Exception as e:
            if "tool_use_failed" in str(e) and attempt < max_retries:
                logger.warning(f"Malformed tool call. Retrying {attempt+1}/{max_retries}")
                if attempt > 0:
                    current_agent = build_agent(FALLBACK_MODEL)
                continue
            logger.error(f"Final Agent Error: {e}")
            return f"⚠️ Agent Error: {e}"

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Missing GROQ_API_KEY")
        sys.exit(1)
    
    agent = build_agent()
    print("✅ Production Agent Ready.")
    while True:
        try:
            q = input("You: ").strip()
            if q.lower() in ["quit", "exit"]: break
            print(f"🤖 {ask_agent(agent, q)}")
        except EOFError: break
