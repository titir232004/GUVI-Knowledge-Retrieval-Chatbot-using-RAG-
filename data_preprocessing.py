import os
import json
import re
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FILE = "output/guvi_full_data.json"
OUTPUT_FILE_JSON = "output/guvi_chunks.json"
OUTPUT_FILE_CSV = "output/guvi_chunks.csv"

# Create output directory
os.makedirs("output", exist_ok=True)

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,        # smaller chunk size for precise retrieval
        chunk_overlap=100,
        separators=["\n\n", ".", "!", "?", ";", ",", " "]
    )

    docs = []
    for item in data:
        if "content" not in item or not item["content"].strip():
            continue
        content = clean_text(item["content"])
        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            docs.append({
                "url": item.get("url"),
                "title": item.get("title"),
                "chunk_id": f"{item.get('url')}_part_{i}",
                "text": chunk
            })
    return docs

def save_output(docs):
    with open(OUTPUT_FILE_JSON, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    pd.DataFrame(docs).to_csv(OUTPUT_FILE_CSV, index=False)
    print(f"✅ Saved {len(docs)} chunks to {OUTPUT_FILE_JSON} and {OUTPUT_FILE_CSV}")

if __name__ == "__main__":
    print("🧹 Preprocessing and chunking GUVI data...")
    data = load_data(INPUT_FILE)
    chunks = chunk_documents(data)
    save_output(chunks)
