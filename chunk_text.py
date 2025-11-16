import json
import os
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FILE = "processed/guvi_clean_text.json"
OUTPUT_FILE = "processed/guvi_chunks.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def chunk_documents(documents: dict):
    """
    documents = { url: long_cleaned_text }
    Returns list of chunks: [{"text": ..., "metadata": {...}}, ...]
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        length_function=len
    )

    chunks = []

    for url, doc_text in tqdm(documents.items(), desc="Chunking documents"):
        if not doc_text or len(doc_text.strip()) < 40:
            continue

        small_chunks = splitter.split_text(doc_text)
        for ch in small_chunks:
            chunks.append({
                "text": ch,
                "metadata": {"source": url}
            })

    return chunks


def main():
    print(f"📘 Loading cleaned text from {INPUT_FILE}")
    data = load_json(INPUT_FILE)
    print(f"Loaded {len(data)} documents.")

    print("🔪 Splitting documents into chunks...")
    chunks = chunk_documents(data)
    print(f"Created {len(chunks)} total chunks.")

    print(f"💾 Saving to {OUTPUT_FILE}")
    save_json(chunks, OUTPUT_FILE)
    print("🎉 Chunking complete!")


if __name__ == "__main__":
    main()
