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

---

## 📁 Project Architecture

User Query
│
▼
[Streamlit Chat UI]
│
▼
[Embedding Model] → Convert query to vector
│
▼
[FAISS Vector Store] → Retrieve top-k similar chunks
│
▼
[Local LLM (TinyLlama GGUF)]
│
▼
Generate final answer based on context
│
▼
ChatGPT-style Response to User



---

## ⚙️ Requirements

### 🔹 Python Version  
**Python 3.10+** recommended

### 🔹 Install Dependencies
After cloning your repo, run:

```bash
pip install -r requirements.txt
