# ✈️ PaperPilot

> An **Agentic RAG** assistant that chats with your documents — upload a PDF, ask questions, get grounded answers with sources.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What is PaperPilot?

PaperPilot lets you **"chat" with any PDF** — lecture notes, research papers,
manuals. Instead of reading 100 pages, just ask a question and get an accurate
answer drawn directly from the document, with the **source chunks shown** for trust.

Built as a learning project covering the full GenAI stack:
**RAG → Agentic AI → Deployment + MLOps.**

---

## ✨ Features

- 📄 **Chat with your PDF** — upload and ask natural-language questions
- 🔍 **Semantic search** — finds answers by *meaning*, not keywords
- 🛡️ **Grounded answers** — uses only the document, says "I don't know" otherwise (reduces hallucination)
- 📚 **Source transparency** — shows the exact chunks each answer came from
- 🤖 **Agentic mode** *(Tier 2)* — tools: retrieval, calculator, web search
- 🚀 **Deployable** *(Tier 3)* — Docker + live hosting + monitoring

---

## 🏗️ Architecture

```
                         ┌──────────────────────────────────┐
   📄 PDF  ─────────────▶│ LOAD → CHUNK → EMBED → ChromaDB   │  (indexing, once)
                         └──────────────────────────────────┘
                                                  │
   ❓ Question ──────────▶  RETRIEVE top-k chunks ─┘
                                  │
                           AUGMENT into prompt
                                  │
                           🤖 LLM (Groq / Llama 3)
                                  │
                           ✅ Grounded Answer + Sources
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Framework | LangChain (+ LangGraph for agents) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, local & free) |
| Vector DB | ChromaDB |
| LLM | Groq API (Llama 3.3, free tier) |
| UI | Streamlit |
| Deploy | Docker + HuggingFace Spaces / Streamlit Cloud |

---

## 🚀 Getting Started

```bash
# 1. Clone & enter
git clone https://github.com/<your-username>/paperpilot.git
cd paperpilot

# 2. Virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your free Groq API key (get one at https://console.groq.com)
cp .env.example .env              # then edit .env and paste your key

# 5. Run the web app
streamlit run src/app.py
```

---

## 📚 Learning the code (step by step)

Each file teaches one concept. Run them in order to understand RAG:

```bash
cd src
python step1_load_and_chunk.py     # PDF → chunks
python step2_build_vectorstore.py  # chunks → embeddings → semantic search
python step3_rag_chain.py          # full RAG: chat in the terminal
```

📖 Full explanations: [`LEARNING_GUIDE.md`](LEARNING_GUIDE.md) (English) ·
[`LEARNING_GUIDE_SI.md`](LEARNING_GUIDE_SI.md) (සිංහල)

---

## 🗺️ Roadmap

- [x] **Tier 1 — RAG pipeline** (load, chunk, embed, retrieve, generate, UI)
- [ ] **Tier 2 — Agentic AI** (tools, multi-step reasoning, LangGraph)
- [ ] **Tier 3 — Deploy + MLOps** (Docker, monitoring, live hosting)

---

## 🧠 What I learned

- How RAG reduces hallucination by grounding the LLM in real documents
- Embeddings & semantic search with a vector database
- Chunking strategies and their effect on retrieval quality
- Building agents that use tools and reason in multiple steps
- Containerizing and deploying a GenAI app with basic monitoring

---

## 📄 License

MIT — free to use and learn from.

---

*Built with ✈️ as a GenAI internship portfolio project.*
