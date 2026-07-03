---
title: PaperPilot AI
emoji: 🧭
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
---

# ✈️ PaperPilot: Agentic RAG Research Assistant

PaperPilot is a sophisticated GenAI application that goes beyond simple RAG (Retrieval-Augmented Generation). It is an **Agentic AI** capable of reasoning, using tools, and providing grounded answers from PDF documents.

Built as a 3-Tier project, it demonstrates the evolution from a basic RAG pipeline to a production-ready, containerized agent.

## 🌟 Key Features

- **Tier 1 (Core RAG):** Intelligent PDF text extraction, cleaning, and semantic search using ChromaDB.
- **Tier 2 (Agentic AI):** A ReAct agent built with **LangGraph** that chooses between:
    - `search_documents`: Contextual search within uploaded PDFs.
    - `calculator`: Precise mathematical computations.
    - `web_search`: Real-time internet access via Tavily API.
    - `summarize_document`: High-level overviews of document content.
- **Tier 3 (MLOps & Production):** 
    - **Dockerized:** Fully containerized for consistent deployment.
    - **Monitoring:** Interaction logging with latency tracking.
    - **Robustness:** Fallback model logic (Llama 3.3 70B -> Llama 3.1 8B) for reliable tool calling.
    - **Clean Architecture:** Modular code with centralized configuration and logging.

## 🏗️ Architecture

1. **Ingestion:** PDF -> Text Cleaning -> Recursive Chunking -> HuggingFace Embeddings.
2. **Retrieval:** ChromaDB Vector Store for semantic similarity search.
3. **Reasoning:** LangGraph ReAct loop powered by Groq (Llama 3.3 70B).
4. **UI:** Streamlit-based chat interface with "Thought Step" visibility.

## 🛠️ Tech Stack

- **Framework:** LangChain & LangGraph
- **LLM:** Groq (Llama 3.3 70B / 3.1 8B)
- **Vector DB:** ChromaDB
- **Embeddings:** HuggingFace (sentence-transformers)
- **UI:** Streamlit
- **DevOps:** Docker, Python Logging, JSONL Monitoring

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Groq API Key (Free)
- Tavily API Key (Free - for web search)

### 2. Installation
```bash
git clone https://github.com/yourusername/paperpilot.git
cd paperpilot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

### 4. Running the App
```bash
streamlit run src/app.py
```

### 5. Running with Docker
```bash
docker build -t paperpilot .
docker run -p 7860:7860 --env-file .env paperpilot
```

## 📊 Monitoring
Interactions are logged to `logs/interactions.jsonl`, capturing:
- User Question
- AI Answer
- Latency (Time taken)
- Timestamp

---
*Developed as a GenAI Capstone Project for Portfolio.*
