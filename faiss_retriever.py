import os
import json
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

FAISS_DIR = "faiss_store"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
TEXTS_PATH = os.path.join(FAISS_DIR, "texts.json")
META_PATH = os.path.join(FAISS_DIR, "meta.json")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class FaissRetriever:
    def __init__(self, index_path=INDEX_PATH, texts_path=TEXTS_PATH, meta_path=META_PATH,
                 model_name=EMBED_MODEL):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not os.path.exists(texts_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("texts.json / meta.json missing in faiss_store")

        self.index = faiss.read_index(index_path)
        with open(texts_path, "r", encoding="utf-8") as f:
            self.texts = json.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metas = json.load(f)

        self.model = SentenceTransformer(model_name)

        if len(self.texts) != self.index.ntotal:
            print("Warning: number of texts != index.ntotal "
                  f"({len(self.texts)} != {self.index.ntotal})")

    def embed_query(self, query: str) -> np.ndarray:
        # Produce normalized embedding (consistent with build step normalize_embeddings=True)
        emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        emb = np.ascontiguousarray(emb).astype("float32")
        return emb

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qv = self.embed_query(query)
        D, I = self.index.search(qv, top_k)
        distances = D[0].tolist()
        indices = I[0].tolist()

        results = []
        for idx, dist in zip(indices, distances):
            if idx < 0 or idx >= len(self.texts):
                continue
            # dist is inner-product on normalized vectors -> in [-1,1], map to 0-1 confidence
            conf = float(max(-1.0, min(1.0, dist)))
            conf_pct = round((conf + 1.0) / 2.0 * 100.0, 1)  # map -1..1 -> 0..100
            results.append({
                "index": int(idx),
                "score": float(dist),
                "confidence": conf_pct,
                "text": self.texts[idx],
                "metadata": self.metas[idx] if idx < len(self.metas) else {}
            })
        return results

if __name__ == "__main__":
    r = FaissRetriever()
    q = input("Query: ")
    for riter in r.retrieve(q, top_k=5):
        print(f"{riter['confidence']}%  idx={riter['index']}  src={riter['metadata'].get('source')}")
        print(riter['text'][:300].replace("\n", " "), "...\n")
