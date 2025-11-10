# vector_store.py

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def create_vector_store(chunk_file="guvi_chunks.json", index_file="guvi_faiss.index", mapping_file="chunk_mapping.json"):
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} text chunks.")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    faiss.write_index(index, index_file)
    print(f"FAISS index created with {index.ntotal} vectors, saved to {index_file}")

    chunk_mapping = {i: chunk for i, chunk in enumerate(chunks)}
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(chunk_mapping, f, ensure_ascii=False, indent=4)
    print(f"Chunk mapping saved to {mapping_file}")

    return index, chunk_mapping

if __name__ == "__main__":
    create_vector_store()
