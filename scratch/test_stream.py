import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("d:/genai-rag-project").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.step5_agent import build_agent
from langchain_core.messages import HumanMessage

agent = build_agent()
print("Agent ready")
for step in agent.stream({"messages": [("user", "What is 2+2?")]}, stream_mode="values"):
    msg = step["messages"][-1]
    print(f"TYPE: {type(msg).__name__}")
    if hasattr(msg, "tool_calls"):
        print(f"TOOL CALLS: {msg.tool_calls}")
    print(f"CONTENT: {msg.content!r}")
    print("---")
