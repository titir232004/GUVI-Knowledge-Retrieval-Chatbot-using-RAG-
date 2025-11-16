# 🤖 GUVI Knowledge Retrieval Chatbot ( RAG System)

The **GUVI Knowledge Retrieval Chatbot** is an offline AI assistant built using a **Retrieval-Augmented Generation (RAG)** pipeline.  
It can answer questions using **real GUVI blogs, FAQs, and course information**, all processed, indexed, and queried locally — **no API keys, no internet, no external dependencies**.

This project uses:

- **FAISS** for semantic retrieval  
- **Sentence-Transformers** for text embeddings  
- **TinyLlama GGUF** (via llama.cpp) for offline LLM inference  **Download Link:** https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF/tree/main
- **Streamlit** for a clean ChatGPT-style chat UI  

---

## ✨ Features

- 🔍 Scrapes and cleans **GUVI Blogs & FAQs**  
- ✂️ Splits content into high-quality text chunks  
- 🧠 Embeds content using **all-MiniLM-L6-v2**  
- ⚡ Fast similarity search using **FAISS vector DB**  
- 🤖 Offline large language model (TinyLlama) for response generation  
- 💬 Beautiful **ChatGPT-style UI** built in Streamlit  
- 🔐 100% offline — All data stays on your machine  
- 🗂️ Modular & production-ready code  

## 📁 Project Structure

GUVI_KNOWLEDGE_RETRIEVAL_CHATBOT
```
├── raw/                           # Raw scraped HTML & extracted paragraphs
├── processed/                     # Cleaned text + generated chunks
├── faiss_store/                   # Vector index + embeddings metadata
│
├── models/
│   └── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf   # Local LLM (not included in GitHub)
│
├── scrape_and_clean.py            # Step 1: Scrape GUVI blogs & FAQs
├── chunk_text.py                  # Step 2: Chunk cleaned text
├── build_faiss_index.py           # Step 3: Build FAISS vector database
├── faiss_retriever.py             # Retrieval testing script (optional)
│
├── rag_engine_streamlit.py        # Core RAG engine (retriever + generator)
├── streamlit_app.py               # ChatGPT-style Streamlit UI
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```
## ⚙️ Requirements

### 🔹 Python Version  
**Python 3.10+** recommended

### 🔹 Install Dependencies
After cloning repo, run:

```bash
pip install -r requirements.txt
