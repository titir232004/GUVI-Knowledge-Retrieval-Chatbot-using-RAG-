import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CHUNK_FILE = "processed/guvi_chunks.json"
OUTPUT_DIR = "faiss_store"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_chunks(path=CHUNK_FILE):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    metas = []
    for d in data:
        text = d.get("text")
        if not text:
            continue
        texts.append(text)
        meta = d.get("metadata", {})
        metas.append(meta)
    return texts, metas

def build_faiss_index(texts, model_name=MODEL_NAME, out_dir=OUTPUT_DIR, batch_size=64):
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(texts)} chunks...")
    # produce normalized embeddings (so we can use IndexFlatIP as cosine)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=batch_size,
        normalize_embeddings=True
    )

    embeddings = np.ascontiguousarray(embeddings).astype("float32")
    dim = embeddings.shape[1]
    print(f"Embedding dimension = {dim}")

    # Build IndexFlatIP (inner-product on normalized vectors -> cosine similarity)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "index.faiss")
    faiss.write_index(index, index_path)
    print(f"FAISS index saved at: {index_path}")

    return index

def save_metadata(texts, metas, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    print(f"Saved texts.json and meta.json in {out_dir}")

def main():
    print(f"Loading chunks from {CHUNK_FILE}")
    texts, metas = load_chunks()
    if len(texts) == 0:
        print("No chunks found. Run chunk_text.py first.")
        return
    print("Building FAISS index...")
    build_faiss_index(texts)
    print("Saving metadata...")
    save_metadata(texts, metas)
    print("FAISS store created.")

if __name__ == "__main__":
    main()
