---
title: PaperPilot AI 🧭
emoji: 🧭
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
---

# ✈️ PaperPilot AI: Production-Grade Agentic RAG Assistant

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/lakshitha722/paperpilot-ai)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2-green?style=for-the-badge)](https://langchain.com/)

> **"Beyond simple RAG. An autonomous agent that reasons, calculates, and browses."**

PaperPilot AI is a sophisticated GenAI application built with a 3-tier production mindset. It transforms static PDF documents into interactive intelligence by combining **Retrieval-Augmented Generation (RAG)** with **Agentic Reasoning**. Powered by **LangGraph** and **Groq's Llama 3.3 70B**, it doesn't just answer questions—it thinks about which tools to use to provide the most accurate, grounded response.

---

## 🔗 Live Demo & Repository
- **Live on HuggingFace Spaces:** [PaperPilot AI 🧭](https://huggingface.co/spaces/lakshitha722/paperpilot-ai)
- **GitHub Repository:** [GitHub Link](https://github.com/lakshitha722/paperpilot-ai) *(Replace with your actual link if different)*

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

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
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

### 3. Setup Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 4. Run the Application
```bash
streamlit run src/app.py
```

---

## 🐳 Docker Deployment
To run the production container locally:
```bash
docker build -t paperpilot .
docker run -p 7860:7860 --env-file .env paperpilot
```

---

## 📈 Monitoring & Evaluation
All user interactions are logged in `logs/interactions.jsonl`.
Fields captured for MLOps analysis:
- **timestamp**: ISO 8601 formatted time.
- **question**: Raw user input.
- **answer**: Generated agentic response.
- **latency**: End-to-end response time in seconds.

---

## 👤 Author
**Lakshitha Wijekoon**
- **GitHub:** [@lakshitha722](https://github.com/lakshitha722)
- **LinkedIn:** [Lakshitha Wijekoon](https://www.linkedin.com/in/lakshitha-wijekoon-592657252/)
