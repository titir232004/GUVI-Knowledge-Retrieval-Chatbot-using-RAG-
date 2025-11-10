import os
import json
import re
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------------- CONFIG ----------------
CHUNKS_FILE = "output/guvi_chunks.json"
FAISS_DIR = "output/guvi_faiss_index"
EMB_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
GEN_MODEL = "google/flan-t5-large"
TOP_K = 6
MAX_CONTEXT_TOKENS = 400


# ---------------- UTILITY FUNCTIONS ----------------
def load_chunks(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing chunks file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", "", text)
    return text.strip()


def filter_chunks(chunks):
    """Remove noisy or irrelevant chunks (URLs, author lines, very short text)."""
    filtered = []
    for c in chunks:
        text = c.strip()
        if len(text) < 50:
            continue
        if "http" in text and len(text) < 200:
            continue
        if "Blog Author" in text or text.startswith("By "):
            continue
        filtered.append(text)
    return filtered


# ---------------- BUILD OR LOAD FAISS ----------------
def build_or_load_faiss(chunks):
    os.makedirs(FAISS_DIR, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL_NAME)

    try:
        vectorstore = FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Loaded existing FAISS index")
        return vectorstore
    except:
        print("⚙️ Building FAISS index from scratch...")
        texts = [c["text"] for c in chunks]
        metadatas = [{"url": c.get("url", ""), "title": c.get("title", ""), "chunk_id": c.get("chunk_id", "")} for c in
                     chunks]
        vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        vectorstore.save_local(FAISS_DIR)
        print("✅ FAISS index built and saved")
        return vectorstore


# ---------------- RETRIEVE ----------------
def retrieve_chunks(vectorstore, query, max_context_tokens=MAX_CONTEXT_TOKENS, k=TOP_K):
    results = vectorstore.similarity_search(query, k=k)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    selected_chunks = []
    total_tokens = 0
    for r in results:
        chunk_text = clean_text(r.page_content)
        filtered = filter_chunks([chunk_text])
        if not filtered:
            continue
        chunk_text = filtered[0]
        chunk_tokens = len(tokenizer(chunk_text)["input_ids"])
        if total_tokens + chunk_tokens > max_context_tokens:
            break
        selected_chunks.append(chunk_text)
        total_tokens += chunk_tokens

    # Debug: show retrieved chunks
    print("\n📝 Retrieved Chunks (Filtered):")
    for i, c in enumerate(selected_chunks):
        print(f"Chunk {i + 1}: {c[:200]}...\n")
    return selected_chunks


# ---------------- PROMPT ----------------
def make_prompt(retrieved_chunks, user_query):
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = (
        f"You are a helpful assistant. Answer the question **ONLY using the information provided in the context below**. "
        f"If the answer is not in the context, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    )
    return prompt


# ---------------- GENERATOR ----------------
def load_generator():
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    print(f"Device set to use {'GPU' if device >= 0 else 'CPU'}")
    return gen


# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("⚙️ Loading chunks and FAISS vectorstore...")
    chunks = load_chunks(CHUNKS_FILE)
    vectorstore = build_or_load_faiss(chunks)
    gen = load_generator()

    print("\n💬 GUVI RAG Chatbot ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit", "bye"):
            print("Bot: Goodbye!")
            break

        retrieved_chunks = retrieve_chunks(vectorstore, query)
        if not retrieved_chunks:
            print("Bot: I don't know\n")
            continue

        prompt = make_prompt(retrieved_chunks, query)
        output = gen(prompt, max_length=512, do_sample=False)
        answer = output[0]["generated_text"] if isinstance(output, list) else str(output)
        answer = " ".join(answer.split())
        print(f"Bot: {answer}\n")
