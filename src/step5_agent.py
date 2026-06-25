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

AGENT_INSTRUCTIONS = """You are PaperPilot, a research assistant.
Use your tools to answer questions precisely. 
Do NOT narrate actions. If no tool is needed, provide the final answer directly."""

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
