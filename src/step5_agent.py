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
from langchain_core.messages import SystemMessage, HumanMessage

# create_react_agent moved in newer LangGraph. Try the new import first,
# fall back to the older one so this works on multiple versions.
try:
    from langchain.agents import create_agent as create_react_agent
except ImportError:
    from langgraph.prebuilt import create_react_agent

from step4_tools import ALL_TOOLS

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# The agent's "personality" + rules. This guides WHEN to use each tool.
AGENT_INSTRUCTIONS = """You are PaperPilot, a helpful research assistant.

You have these tools:
- search_documents: search the uploaded PDF. Use this FIRST for any question
  that could be answered by the document.
- calculator: for any arithmetic or math.
- web_search: ONLY for current/world info not in the document.

Rules:
- Prefer search_documents for document questions.
- If the document does not contain the answer, say so. Only use web_search if
  the user clearly wants outside/current information.
- Always give a clear, concise final answer based on the tool results."""


def build_agent():
    """Create a ReAct agent: an LLM + tools + the reasoning loop."""
    # NOTE on model choice:
    # Agents need RELIABLE "tool calling". Some Llama models on Groq
    # occasionally emit a malformed tool call (the '<function=...>' error),
    # which crashes the request. Groq's openai/gpt-oss-20b and the
    # llama-3.3-70b both support tool calling; if 70b misbehaves, switching
    # models is the standard fix. We keep 70b but you can swap it here.
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    # create_react_agent wires the LLM to the tools and the think/act loop.
    agent = create_react_agent(llm, ALL_TOOLS)
    return agent


def ask_agent(agent, question, show_steps=True):
    """Send a question to the agent and return its final answer."""
    # We pass a system message (rules) + the user's question.
    inputs = {
        "messages": [
            SystemMessage(content=AGENT_INSTRUCTIONS),
            HumanMessage(content=question),
        ]
    }

    final_answer = ""
    try:
        # .stream lets us watch each step (tool calls + results) as it happens.
        for step in agent.stream(inputs, stream_mode="values"):
            msg = step["messages"][-1]
            if show_steps:
                # Show tool calls the agent decided to make.
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        print(f"   🔧 calling tool: {tc['name']}({tc['args']})")
            final_answer = msg.content
    except Exception as e:
        # Llama sometimes emits a malformed tool call ('tool_use_failed').
        # Instead of crashing, tell the user and let them retry.
        if "tool_use_failed" in str(e):
            return ("⚠️ The model produced a malformed tool call (a known "
                    "Llama+Groq hiccup). Please ask again, or rephrase the "
                    "question slightly.")
        return f"⚠️ Agent error: {e}"
    return final_answer


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found. Create a .env file (see .env.example).")
        raise SystemExit

    agent = build_agent()
    print("✅ Agent ready! It will pick tools automatically. (type 'quit' to exit)")
    print("   Try: 'What is Machine Learning?'  or  'What is 25 * 4?'\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        print("   (agent is thinking...)")
        answer = ask_agent(agent, q)
        print(f"\n🤖 {answer}\n")
