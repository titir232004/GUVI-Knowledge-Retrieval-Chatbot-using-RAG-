import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

FAISS_DIR = "faiss_store"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
TEXTS_PATH = os.path.join(FAISS_DIR, "texts.json")
META_PATH = os.path.join(FAISS_DIR, "meta.json")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

SYSTEM_PROMPT = (
    "You are a helpful assistant for the GUVI platform. "
    "Use the retrieved context to answer questions. "
    "If the answer is not in the context, say you don't know."
)

TOP_K = 4


class RagEngine:
    def __init__(self):
        self.index = faiss.read_index(INDEX_PATH)

        with open(TEXTS_PATH, "r", encoding="utf-8") as f:
            self.texts = json.load(f)
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.metas = json.load(f)

        self.embedder = SentenceTransformer(EMBED_MODEL)

        self.llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=6,
            temperature=0.2,
        )

        self.memory = []

    def embed(self, text):
        vec = self.embedder.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(vec)
        return vec.astype("float32")

    def retrieve(self, query):
        qv = self.embed(query)
        D, I = self.index.search(qv, TOP_K)
        docs = []

        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.texts):
                docs.append(self.texts[idx])

        return "\n\n".join(docs)

    def format_prompt(self, user_query, context):
        """
        ChatML format required by TinyLlama
        """
        prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|context|>\n{context}\n"
            f"<|user|>\n{user_query}\n"
            f"<|assistant|>\n"
        )
        return prompt

    def answer(self, query):
        ctx = self.retrieve(query)
        prompt = self.format_prompt(query, ctx)

        response = self.llm(
            prompt,
            max_tokens=256,
            stop=["<|user|>", "<|system|>", "<|context|>"],
        )

        reply = response["choices"][0]["text"].strip()

        # Strip leftover tokens if any
        for bad in ["<|assistant|>", "<|system|>", "<|user|>", "<|context|>"]:
            reply = reply.replace(bad, "")

        return reply
