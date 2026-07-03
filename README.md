
---
title: PaperPilot AI 🧭
emoji: 🧭
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
---

# ✈️ PaperPilot AI: Production-Grade Agentic RAG Assistant

> **"Beyond simple RAG. An autonomous agent that reasons, calculates, and browses."**

PaperPilot AI is a sophisticated GenAI application built with a 3-tier production mindset. It transforms static PDF documents into interactive intelligence by combining **Retrieval-Augmented Generation (RAG)** with **Agentic Reasoning**. Powered by **LangGraph** and **Groq's Llama 3.3 70B**, it doesn't just answer questions—it thinks about which tools to use to provide the most accurate, grounded response.

---

## 🔗 Live Demo & Repository
- **Live on HuggingFace Spaces:** [PaperPilot AI 🧭](https://huggingface.co/spaces/lakshitha722/paperpilot-ai)
- **GitHub Repository:** [GitHub Link](https://github.com/lakshitha722/paperpilot-ai) *(Replace with your actual link)*

---

## 🏗️ Evolution Architecture (3-Tier Roadmap)

### 🟢 Tier 1: Core RAG Pipeline
*   **Ingestion:** Intelligent PDF text extraction using `PyPDFLoader` with custom data cleaning.
*   **Chunking:** `RecursiveCharacterTextSplitter` with semantic overlap.
*   **Vector Database:** Local persistence with `ChromaDB` for high-speed semantic retrieval.
*   **Embeddings:** Open-source `sentence-transformers` running locally.

### 🟡 Tier 2: Agentic Intelligence (ReAct Pattern)
*   **Orchestration:** Built with **LangGraph** for structured state management.
*   **Multi-Tool Reasoning:** The LLM autonomously chooses between:
    - `search_documents`: Deep semantic retrieval from the uploaded PDF.
    - `calculator`: Precise mathematical computations using Python's math engine.
    - `web_search`: Live internet grounding via **Tavily API**.
    - `summarize_document`: High-context broad synthesis of document content.
*   **Strict Logic:** Multi-step reasoning ensuring document context is prioritized over training data.

### 🔴 Tier 3: Production & MLOps
*   **Resiliency:** Dual-model fallback logic (Llama 3.3 70B -> Llama 3.1 8B) for 99% tool-calling reliability.
*   **Containerization:** Fully dockerized with specialized non-root user permissions for secure deployment.
*   **Monitoring:** JSONL-based interaction logging with latency tracking for performance evaluation.
*   **Modern UI:** Glassmorphism-inspired Streamlit interface with real-time "Thought Step" visibility.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **LLM Engine** | Groq (Llama 3.3 70B / 3.1 8B) |
| **Frameworks** | LangChain, LangGraph |
| **Vector DB** | ChromaDB |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) |
| **Interface** | Streamlit (Custom CSS/Glassmorphism) |
| **Tools** | Tavily Web Search, Python Math |
| **Infrastructure** | Docker, HuggingFace Spaces |

---
=======
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
<<<<<<< HEAD
- [Groq API Key](https://console.groq.com/)
- [Tavily API Key](https://tavily.com/)

### 2. Local Installation
```bash
# Clone the repository
git clone https://github.com/lakshitha722/paperpilot-ai.git
cd paperpilot-ai

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### 3.  Setup Environment Variables
Create a .env file:
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key

### 4. Run the Application
streamlit run src/app.py

---

## 🐳 Docker Deployment
To run the production container locally:
docker build -t paperpilot .
docker run -p 7860:7860 --env-file .env paperpilot

---

## 📈 Monitoring & Evaluation
All user interactions are logged in logs/interactions.jsonl.
Fields captured for MLOps analysis:

timestamp: ISO 8601 formatted time.
question: Raw user input.
answer: Generated agentic response.
latency: End-to-end response time in seconds.

---

## 👤 Author
Lakshitha Wijekoon

LinkedIn: Your LinkedIn Profile (Replace with your actual link)
GitHub: @lakshitha722
