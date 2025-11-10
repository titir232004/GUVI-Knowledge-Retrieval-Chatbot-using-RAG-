from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
import os

INPUT_FILE = "output/guvi_chunks.json"
FAISS_DIR = "output/guvi_faiss_index"

# Load chunks
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]
metadatas = [{"url": c.get("url",""), "title": c.get("title",""), "chunk_id": c.get("chunk_id","")} for c in chunks]

# Use the SAME embedding model as in retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

# Save FAISS index
os.makedirs(FAISS_DIR, exist_ok=True)
vectorstore.save_local(FAISS_DIR)
print("✅ Rebuilt FAISS vectorstore successfully")
