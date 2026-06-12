"""
========================================================================
STEP 5 (Day 9): THE AGENT  ->  this is "Agentic AI"
========================================================================

WHAT IS AN AGENT?
-----------------
In Tier 1 (RAG) we ALWAYS did: retrieve -> answer. That's fixed.

An AGENT is different: we give the LLM a set of TOOLS and let IT decide,
step by step, which tool(s) to call to answer the question. It can:
    - call search_documents for PDF questions
    - call calculator for math
    - call web_search for live info
    - call several tools in sequence, then write a final answer

This loop of "think -> act (call a tool) -> observe result -> think again"
is the ReAct pattern (Reasoning + Acting).

We use LangGraph's prebuilt `create_react_agent`, which implements this loop
for us. Under the hood it relies on the LLM's "tool calling" ability.

Run (set GROQ_API_KEY in .env first):
    python src/step5_agent.py
========================================================================
"""

import os
import sys
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"
sys.path.append(str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langgraph.prebuilt import create_react_agent

from step4_tools import ALL_TOOLS

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# The agent's "personality" + rules.
# NOTE: Do NOT include manual tool-call syntax (e.g. 'call: func(arg=...)').
# Llama models interpret that as instructions to emit XML/text-style calls
# instead of the proper JSON tool-call format that Groq's API expects.
# LangGraph already tells the LLM about available tools via bind_tools().
AGENT_INSTRUCTIONS = """You are PaperPilot, a research assistant.

You have access to tools for:
- Searching uploaded PDF documents for information
- Evaluating math expressions
- Searching the live internet

Use the appropriate tool when it would help answer the user's question.
Do NOT narrate your actions. Do NOT say "I will now search for...".
If no tool is needed, provide the final answer directly."""


# Primary and fallback models. If the primary model consistently produces
# malformed tool calls, the agent will retry with the fallback model.
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"


def build_agent(model_name=None):
    """Create a ReAct agent: an LLM + tools + the reasoning loop."""
    model = model_name or PRIMARY_MODEL
    # temperature=0 keeps tool calls precise and deterministic.
    llm = ChatGroq(model=model, temperature=0)

    # create_react_agent wires the LLM to the tools and the think/act loop.
    # We pass the system prompt via `prompt` so LangGraph manages it properly.
    agent = create_react_agent(llm, ALL_TOOLS, prompt=AGENT_INSTRUCTIONS)
    return agent


def ask_agent(agent, question, show_steps=True, max_retries=2):
    """Send a question to the agent and return its final answer.

    Includes automatic retry logic for the known Llama+Groq malformed
    tool call issue. On the first retry it reuses the same agent; on
    subsequent retries it rebuilds the agent with a fallback model.
    """
    # The system message is now handled by the agent's `prompt` parameter,
    # so we only pass the user's question here.
    inputs = {
        "messages": [
            HumanMessage(content=question),
        ]
    }

    current_agent = agent
    for attempt in range(max_retries + 1):
        final_answer = ""
        try:
            # .stream lets us watch each step (tool calls + results) as it happens.
            for step in current_agent.stream(inputs, stream_mode="values"):
                msg = step["messages"][-1]
                if show_steps:
                    # Show tool calls the agent decided to make.
                    tool_calls = getattr(msg, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            print(f"    🔧 calling tool: {tc['name']}({tc['args']})")
                final_answer = msg.content
            return final_answer
        except Exception as e:
            # Llama sometimes emits a malformed tool call ('tool_use_failed').
            if "tool_use_failed" in str(e) and attempt < max_retries:
                if attempt == 0:
                    # First retry: same model, maybe it was transient.
                    print(f"    ⚠️ Malformed tool call — retrying ({attempt + 1}/{max_retries})...")
                else:
                    # Later retries: switch to the fallback model.
                    print(f"    ⚠️ Retrying with fallback model ({FALLBACK_MODEL})...")
                    current_agent = build_agent(model_name=FALLBACK_MODEL)
                continue
            elif "tool_use_failed" in str(e):
                return ("⚠️ The model produced a malformed tool call (a known "
                        "Llama+Groq hiccup). Retried {} time(s) but it persisted. "
                        "Please rephrase the question slightly.".format(max_retries))
            return f"⚠️ Agent error: {e}"
    return final_answer


if __name__ == "__main__":
    # Fix unicode encoding issues on Windows terminals
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found. Create a .env file (see .env.example).")
        raise SystemExit

    agent = build_agent()
    print("✅ Agent ready! It will pick tools automatically. (type 'quit' to exit)")
    print("   Try: 'What is Machine Learning?'  or  'What is 25 * 4?'\n")

    while True:
        try:
            q = input("You: ").strip()
        except EOFError:
            break
        if q.lower() in {"quit", "exit"}:
            break
        print("   (agent is thinking...)")
        answer = ask_agent(agent, q)
        print(f"\n🤖 {answer}\n")
