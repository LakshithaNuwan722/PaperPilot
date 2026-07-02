"""
Configuration settings for the PaperPilot project.
Centralizing these values makes it easier to update the app without touching the core logic.
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Model Configurations
# As of July 2026, Groq's recommended production models with tool-use support:
#   Primary  : openai/gpt-oss-120b  (120B params, ~500 tps, full tool-calling)
#   Fallback : openai/gpt-oss-20b   (20B params, faster, lighter)
# See: https://console.groq.com/docs/models
PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# RAG Configurations
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_SEARCH_K = 3
SUMMARY_SEARCH_K = 10

# API Keys (Loaded from .env)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
