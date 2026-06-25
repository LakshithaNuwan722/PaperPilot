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
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
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
